import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from ultimateBot.audit_hybrid_trades import market_features, read_log, reconstruct_context, replay_exit, stats
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules


class HybridTradeAuditTests(unittest.TestCase):
    def test_orders_are_joined_by_order_id_and_breakeven_is_separate(self):
        log = '''[trade-created] order_id=9 symbol=EURUSD timestamp=2024-01-01T01:00:00+00:00 signal_timestamp=2024-01-01T00:59:59+00:00 side=buy order_type=limit entry_price=100
[trade-filled] order_id=9 trade_id=2 symbol=EURUSD timestamp=2024-01-01T01:00:01+00:00 side=buy quantity=3 entry_price=100 stop_loss=99 take_profit=None trailing_activation_price=101
[trade-closed] trade_id=2 symbol=EURUSD timestamp=2024-01-01T02:00:00+00:00 side=buy quantity=3 entry_price=100 exit_price=100 pnl=0.00 reason=trailing_stop
Total closed trades: 1
'''
        for encoding in ('utf-8', 'utf-16'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'out.log'
                path.write_text(log, encoding=encoding)
                _, orders, trades = read_log(path)
            self.assertEqual(len(orders), 1)
            self.assertEqual(trades.iloc[0].branch, 'breakout-retest')
            self.assertEqual(trades.iloc[0].order_id, 9)
            self.assertEqual(trades.iloc[0].trade_id, 2)
            self.assertEqual(trades.iloc[0].outcome, 'flat')
            self.assertEqual(stats(trades)['flats'], 1)

    def row(self, direction=1):
        return SimpleNamespace(trade_id=1, symbol='EURUSD', opened_at=pd.Timestamp('2024-01-01T00:00:00Z'),
                               direction=direction, entry=100., stop=100.-direction,
                               initial_risk=1., activation=100.+direction, target=None)

    def test_entry_ratios_use_planned_price_before_future_fill(self):
        log = '''[trade-created] order_id=1 symbol=EURUSD timestamp=2024-01-01T01:00:00Z signal_timestamp=2024-01-01T00:59:59Z side=buy order_type=limit entry_price=100.1
[trade-filled] order_id=1 trade_id=1 symbol=EURUSD timestamp=2024-01-01T01:00:01Z side=buy quantity=1 entry_price=100 stop_loss=99 take_profit=102.3
[trade-closed] trade_id=1 symbol=EURUSD timestamp=2024-01-01T02:00:00Z side=buy quantity=1 entry_price=100 exit_price=102.3 pnl=2.30 reason=take_profit
Total closed trades: 1
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            path.write_text(log)
            _, _, frame = read_log(path)
        self.assertAlmostEqual(frame.iloc[0].planned_risk, 1.1)
        self.assertAlmostEqual(frame.iloc[0].planned_rr, 2.)
        self.assertAlmostEqual(frame.iloc[0].filled_rr, 2.3)

    def test_replay_honors_existing_stop_before_trailing_update(self):
        row = self.row()
        start = row.opened_at.timestamp()*1000
        result = replay_exit(row, np.array([[start+1, 98.9, 99.]]), original=True)
        self.assertAlmostEqual(result['r'], -1.1)

    def test_trailing_replay_handles_long_and_short_using_executable_side(self):
        for direction in (1, -1):
            row = self.row(direction)
            start = row.opened_at.timestamp()*1000
            prices = 100 + direction*np.array([1., 2.2, 1.7, .9])
            bid = prices if direction > 0 else prices-.1
            ask = prices+.1 if direction > 0 else prices
            result = replay_exit(row, np.column_stack([start+np.arange(1,5), bid, ask]), original=True)
            self.assertAlmostEqual(result['r'], .9)

    def test_fixed_target_and_censored_path(self):
        row = self.row()
        start = row.opened_at.timestamp()*1000
        quotes = np.array([[start+1, 101., 101.1], [start+2, 102.4, 102.5]])
        self.assertAlmostEqual(replay_exit(row, quotes, fixed_rr=2.3)['r'], 2.4)
        self.assertIsNone(replay_exit(row, quotes[:1], fixed_rr=2.3)['r'])

    def test_features_ignore_future_bars_and_quote_side_is_not_future_outcome(self):
        start = pd.Timestamp('2024-01-01T00:00:00Z')
        bars = [{'start_ms':int((start+pd.Timedelta(hours=i)).timestamp()*1000), 'mid':[100,102,99,101]}
                for i in range(80)]
        row = SimpleNamespace(signal_at=start+pd.Timedelta(hours=73, minutes=59), direction=1)
        original = market_features(bars, row, 60, 1.)
        bars[-1]['mid'] = [1000,2000,1,1000]
        self.assertEqual(original, market_features(bars, row, 60, 1.))

    def test_context_recovery_requires_the_logged_stop_and_target(self):
        start = pd.Timestamp('2024-01-01T00:00:00Z')
        bars = []
        for index in range(30):
            bid = [105, 108, 102, 105] if index < 29 else [105, 112, 104, 109]
            bars.append(dict(start_ms=int((start+pd.Timedelta(hours=index)).timestamp()*1000),
                             bid=bid, ask=[value+.1 for value in bid], mid=[value+.05 for value in bid],
                             count=4, volumes=[0,0,4,4]))
        analysis = dict(bar_count=25, atr=2., reasons=[], parallelism=.8, swing_density=.5,
                        filtered_consecutive_swing_pair_density=.1, shared_slope=0., market_volatility_ratio=.001,
                        anchored_line=dict(kind='low', intercept=100., slope=0.),
                        floating_line=dict(kind='high', intercept=110., slope=0.))
        strategy = SimpleNamespace(regression_rules=resolve_swing_regression_rules({'regression_entry_mode':'hybrid',
                                                                                  'regression_lookback_bars':25}),
                                   _evaluate_regression_pipeline=Mock(return_value=(None,analysis)))
        row = SimpleNamespace(signal_at=start+pd.Timedelta(hours=29, minutes=59), branch='failed-breakout',
                               symbol='EURUSD', side='sell', stop=112.5, target=105., initial_risk=3.5,
                               planned_risk=3.5, direction=-1)
        self.assertEqual(reconstruct_context(SimpleNamespace(strategy=strategy),bars,row,60)['context_matches'],1)
        row.stop = 112.6
        self.assertEqual(reconstruct_context(SimpleNamespace(strategy=strategy),bars,row,60)['context_matches'],0)


if __name__ == '__main__':
    unittest.main()
