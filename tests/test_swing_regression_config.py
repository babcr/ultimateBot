import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from unittest.mock import patch

from ultimateBot.backtester.config.trading_schedule import MarketSession
from ultimateBot.backtester.optimization.bandit_core import BanditConfig
from ultimateBot.backtester.strategy.structure_decision_engine import DEFAULT_STRUCTURE_RULES
from ultimateBot.backtester.strategy.swing_regression_config import SwingRegressionConfig
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import (
    build_engine, parse_args, resolve_backtest_regression_config,
)


class SwingRegressionConfigTests(unittest.TestCase):
    def parse(self, config=None, *options):
        with patch.object(sys, "argv", ["run_backtest", *options]):
            return parse_args(regression_config=config)

    def configured(self):
        config = SwingRegressionConfig()
        return replace(
            config,
            regression=replace(config.regression, lookback_bars=12, score_threshold=0.72,
                               min_market_volatility_ratio=0),
            global_timeframe=replace(config.global_timeframe, global_timeframe_multiplier=3),
            levels=replace(config.levels, level_lookback_bars=40, level_merge_atr_ratio=0.21,
                           min_level_touches=4),
            candles=replace(config.candles, confirmation_body_ratio=0.61),
            protection=replace(config.protection, protective_stop_buffer_atr=0.7,
                               protective_level_source="recent_swing"),
            risk=replace(config.risk, risk_per_trade_percent=0.4, reward_risk_ratio=3.0,
                         max_positions=3, max_spread_ratio=0.42),
            volatility=replace(config.volatility, candle_interval=timedelta(minutes=2),
                               candle_lookback=77, minimum_candles=8, loss_shrink_ratio=0.33),
            orders=replace(config.orders, expiration_bars=7, execution_delay=timedelta(milliseconds=120)),
            execution=replace(config.execution, balance=900, batch_size=37, trailing_stop=True,
                              minimum_candles=45),
            schedule=replace(config.schedule, session_filter=True, session_entry_fraction=0.5,
                             sessions=(MarketSession("Test", "UTC", "08:00", "12:00"),)),
        )

    def test_every_inherited_rule_has_an_independent_interface(self):
        config = self.configured()
        self.assertLessEqual(set(DEFAULT_STRUCTURE_RULES), set(config.to_rules()))
        with patch.dict(DEFAULT_STRUCTURE_RULES, {
            "level_lookback_bars": 999, "level_merge_atr_ratio": 8,
            "confirmation_body_ratio": 0.99, "protective_stop_buffer_atr": 9,
        }):
            rules = resolve_swing_regression_rules(config)
        self.assertEqual(rules["level_lookback_bars"], 40)
        self.assertEqual(rules["level_merge_atr_ratio"], 0.21)
        self.assertEqual(rules["confirmation_body_ratio"], 0.61)
        self.assertEqual(rules["protective_stop_buffer_atr"], 0.7)

    def test_direct_strategy_uses_grouped_rules_and_allocates_all_windows(self):
        config = self.configured()
        strategy = SwingRegressionStrategy(config=config)
        self.assertEqual(strategy.regression_rules, resolve_swing_regression_rules(config))
        self.assertEqual(strategy.history_lookback_bars, 12)
        self.assertEqual(strategy.level_history_lookback_bars, 40)
        self.assertEqual(strategy.global_history_capacity, 36)
        self.assertEqual(strategy.minimum_candles, 45)
        self.assertEqual(strategy.analysis_interval, timedelta(minutes=2))

    def test_legacy_constructor_arguments_override_grouped_inputs(self):
        strategy = SwingRegressionStrategy(
            {"regression_score_threshold": 0.8, "level_lookback_bars": 60},
            minimum_candles=70, analysis_interval=timedelta(minutes=5), config=self.configured(),
        )
        self.assertEqual(strategy.config.regression.score_threshold, 0.8)
        self.assertEqual(strategy.config.levels.level_lookback_bars, 60)
        self.assertEqual(strategy.config.execution.minimum_candles, 70)
        self.assertEqual(strategy.config.volatility.candle_interval, timedelta(minutes=5))

    def test_config_reaches_strategy_risk_orders_schedule_and_batcher(self):
        config = self.configured()
        engine = build_engine([], self.parse(config))
        self.assertEqual(engine.strategy.regression_rules, resolve_swing_regression_rules(config))
        self.assertEqual(engine.risk_manager.risk_config.risk_per_trade_percent, 0.4)
        self.assertEqual(engine.risk_manager.risk_config.max_positions, 3)
        self.assertEqual(engine.risk_manager.risk_config.reward_risk_ratio, 3.0)
        self.assertEqual(engine.risk_manager.volatility_config, config.volatility)
        self.assertEqual(engine.order_builder.execution_delay, timedelta(milliseconds=120))
        self.assertEqual(engine.order_builder.expiration_bars, 7)
        self.assertEqual(engine.order_builder.candle_interval, timedelta(minutes=2))
        self.assertEqual(engine.market_state.candle_interval, timedelta(minutes=2))
        self.assertEqual(engine.trading_schedule.config, config.schedule)
        self.assertEqual(engine.batcher.batch_size, 37)
        self.assertEqual(engine.portfolio.balance, 900)
        self.assertTrue(engine.trade_closer.trailing_stop_enabled)
        self.assertEqual(engine.strategy.config.risk.risk_per_trade_percent, 0.4)

    def test_explicit_cli_options_take_precedence_and_snapshot_reports_them(self):
        config = self.configured()
        args = self.parse(config, "--risk-percent", "0.9", "--regression-score-threshold", "0.81",
                          "--structure-stop-buffer-atr", "0.2", "--candle-interval", "3m",
                          "--order-expiration-bars", "4", "--no-regression-level-reaction")
        engine = build_engine([], args)
        effective = engine.strategy.config
        self.assertEqual(effective.regression.score_threshold, 0.81)
        self.assertFalse(effective.regression.require_level_reaction)
        self.assertEqual(effective.protection.protective_stop_buffer_atr, 0.2)
        self.assertEqual(effective.risk.risk_per_trade_percent, 0.9)
        self.assertEqual(effective.orders.expiration_bars, 4)
        self.assertEqual(effective.volatility.candle_interval, timedelta(minutes=3))
        self.assertEqual(engine.risk_manager.volatility_config.candle_interval, timedelta(minutes=3))
        self.assertEqual(engine.order_builder.candle_interval, timedelta(minutes=3))
        self.assertEqual(config.risk.risk_per_trade_percent, 0.4)

    def test_other_strategies_keep_their_own_defaults(self):
        args = self.parse(self.configured(), "--strategy", "structure")
        plain_args = self.parse(None, "--strategy", "structure")
        self.assertEqual(vars(args), vars(plain_args))
        self.assertEqual(self.parse().strategy, "structure")

    def test_default_and_legacy_level_fallback_remain_equivalent(self):
        config = SwingRegressionConfig()
        self.assertEqual(SwingRegressionStrategy().regression_rules, resolve_swing_regression_rules(config))
        config = replace(config, regression=replace(config.regression, lookback_bars=None))
        args = self.parse(config, "--level-lookback-bars", "17")
        effective = resolve_backtest_regression_config(args)
        self.assertEqual(effective.regression.lookback_bars, 17)
        self.assertEqual(effective.levels.level_lookback_bars, 17)

    def test_adaptive_configuration_reaches_optimizer(self):
        config = self.configured()
        config = replace(config, optimization=replace(
            config.optimization, adaptive_enabled=True,
            adaptive=replace(config.optimization.adaptive, lookback_min=8, lookback_max=25,
                             lookback_step=2, seed=123, optimize_risk=False),
        ))
        engine = build_engine([], self.parse(config))
        self.assertEqual(engine.adaptive_optimizer.config, config.optimization.adaptive)
        self.assertEqual(engine.strategy.config.optimization.adaptive, config.optimization.adaptive)

    def test_bandit_parameters_and_runtime_options_are_exposed(self):
        config = self.configured()
        bandit = BanditConfig(lookback_min=6, lookback_max=30, lookback_step=3,
                             context_bars=23, profit_factor_window=12, alpha=0.7,
                             regularization=2, shadow_budget=0.1, group_main_trades=5)
        config = replace(config, optimization=replace(config.optimization, bandit=bandit,
                                                      bandit_read_only=True, bandit_run_id="config-test"))
        effective = resolve_backtest_regression_config(self.parse(config))
        self.assertEqual(effective.optimization.bandit, bandit)
        self.assertTrue(effective.optimization.bandit_read_only)
        self.assertEqual(effective.optimization.bandit_run_id, "config-test")

    def test_configs_and_rule_adapters_are_isolated(self):
        config = SwingRegressionConfig()
        with self.assertRaises(FrozenInstanceError):
            config.levels.level_lookback_bars = 99
        rules = config.to_rules()
        rules["level_lookback_bars"] = 999
        self.assertEqual(config.levels.level_lookback_bars, 33)
        first, second = SwingRegressionStrategy(), SwingRegressionStrategy()
        expected = resolve_swing_regression_rules(config)["regression_lookback_bars"]
        first.set_history_lookback_bars(10, symbol="EURUSD")
        self.assertEqual(second.get_history_lookback_bars("EURUSD"), expected)
        self.assertEqual(first.config.regression.lookback_bars, expected)


if __name__ == "__main__":
    unittest.main()
