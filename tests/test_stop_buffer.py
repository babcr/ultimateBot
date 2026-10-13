import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy import structure_decision_engine as structure
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy
from ultimateBot.run_backtest import build_engine, parse_args


class StopBufferTests(unittest.TestCase):
    def entry(self, source, side, buffer_atr, min_rr=0.5):
        rules = structure.resolve_structure_rules({
            "protective_level_source": source,
            "protective_stop_buffer_atr": buffer_atr,
            "entry_distance_atr": 0.5,
            "min_entry_quality": 0.55,
            "min_rr_ratio": min_rr,
        })
        candle = {"bullish_confirmation": float(side == "buy"),
                  "bearish_confirmation": float(side == "sell"),
                  "previous_close": 100.0, "low": 99.0, "high": 101.0}
        if side == "buy":
            return structure._evaluate_buy_entry(
                100.0, 2.0, candle, {"price": 99.8, "quality": 0.9}, None,
                {"price": 102.0, "quality": 0.9}, {"price": 98.8, "quality": 0.9}, rules,
            )
        return structure._evaluate_sell_entry(
            100.0, 2.0, candle, {"price": 100.2, "quality": 0.9}, None,
            {"price": 98.0, "quality": 0.9}, {"price": 101.2, "quality": 0.9}, rules,
        )

    def manager_and_tick(self):
        manager = RiskManager(risk_per_trade=0.1, minimum_candles=1, reward_risk_ratio=2.0)
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        ticks = [Tick(start, "TEST", 98.0, 98.0, 1.0, 1.0),
                 Tick(start + timedelta(seconds=30), "TEST", 100.0, 100.0, 1.0, 1.0)]
        for tick in ticks:
            manager.on_tick(tick)
        return manager, ticks[-1]

    def evaluate_entry(self, entry, side, style):
        decision = structure._decision_from_setup(side, entry, {"rule_margin": 0.2})
        decision = replace(decision, entry_style=style)
        manager, tick = self.manager_and_tick()
        signal = StructureStrategy()._build_signal(
            TickBatch(tick.symbol, [tick]), decision,
            decision.protective_level_price, decision.protective_level_quality,
        )
        self.assertEqual(signal.protective_stop_buffer, entry["protective_stop_buffer"])
        result = manager.evaluate(signal, Portfolio(1000.0), tick)
        self.assertIsNotNone(result)
        return result

    def test_buffer_moves_both_stops_and_resizes_all_order_types(self):
        for source in ("structure_level", "recent_swing"):
            for side in ("buy", "sell"):
                styles = (None, "support_bounce", "resistance_breakout") if side == "buy" else (
                    None, "resistance_rejection", "support_breakdown",
                )
                for style in styles:
                    with self.subTest(source=source, side=side, style=style):
                        original_entry = self.entry(source, side, 0.0)
                        padded_entry = self.entry(source, side, 0.25)
                        self.assertTrue(padded_entry["entry_ok"])
                        self.assertEqual(padded_entry["protective_level_source"], source)
                        original = self.evaluate_entry(original_entry, side, style)
                        padded = self.evaluate_entry(padded_entry, side, style)
                        expected_shift = -0.5 if side == "buy" else 0.5
                        self.assertAlmostEqual(padded.stop_loss - original.stop_loss, expected_shift)
                        self.assertAlmostEqual(padded.stop_distance - original.stop_distance, 0.5)
                        self.assertAlmostEqual(padded.take_profit_distance - original.take_profit_distance, 1.0)
                        self.assertLess(padded.quantity, original.quantity)
                        self.assertAlmostEqual(padded.quantity * padded.stop_distance, 1.0)
                        self.assertAlmostEqual(padded.capital_at_risk, 1.0)

    def test_rr_filter_accounts_for_buffer_for_both_stop_sources(self):
        for source, min_rr in (("structure_level", 2.0), ("recent_swing", 1.5)):
            for side in ("buy", "sell"):
                with self.subTest(source=source, side=side):
                    original = self.entry(source, side, 0.0, min_rr)
                    padded = self.entry(source, side, 0.25, min_rr)
                    self.assertTrue(original["entry_ok"])
                    self.assertFalse(padded["entry_ok"])
                    self.assertIn("poor_rr", padded["blockers"])
                    self.assertAlmostEqual(padded["stop_distance_atr"] - original["stop_distance_atr"], 0.25)

    def test_buffer_is_added_after_minimum_stop_distance(self):
        for side, level in ((SignalSide.BUY, 99.999999), (SignalSide.SELL, 100.000001)):
            with self.subTest(side=side):
                manager, tick = self.manager_and_tick()
                signal = Signal(tick.symbol, tick.timestamp, side,
                                protective_level_price=level, protective_level_quality=1.0)
                original = manager.evaluate(signal, Portfolio(1000.0), tick)
                padded = manager.evaluate(replace(signal, protective_stop_buffer=0.5), Portfolio(1000.0), tick)
                self.assertIsNotNone(original)
                self.assertIsNotNone(padded)
                self.assertAlmostEqual(original.stop_distance, 0.12)
                self.assertAlmostEqual(padded.stop_distance, 0.62)

    def test_cli_passes_buffer_to_structure_strategy(self):
        with patch.object(sys, "argv", ["run_backtest", "--structure-stop-buffer-atr", "0.75",
                                        "--structure-stop-source", "recent_swing"]):
            engine = build_engine(feed=[], args=parse_args())
        self.assertEqual(engine.strategy.structure_rules["protective_stop_buffer_atr"], 0.75)
        self.assertEqual(engine.strategy.structure_rules["protective_level_source"], "recent_swing")


if __name__ == "__main__":
    unittest.main()
