"""Selective hybrid baseline; optional --regression-pullback-enabled adds the experiment."""
from dataclasses import replace
from datetime import timedelta
import sys

from ultimateBot.backtester.strategy.swing_regression_config import DEFAULT_SWING_REGRESSION_CONFIG
from ultimateBot.run_backtest import main


def hybrid_config():
    config = DEFAULT_SWING_REGRESSION_CONFIG
    return replace(config,
        regression=replace(config.regression,entry_mode="hybrid",lookback_bars=25,max_spread_stop_ratio=0.),
        reentry=replace(config.reentry,quality_mode="selective"),
        breakout=replace(config.breakout,confirmation_mode="confirmed"),
        pullback=replace(config.pullback,enabled=False),
        impulse=replace(config.impulse,enabled=False),
        execution=replace(config.execution,trailing_stop=True),
        optimization=replace(config.optimization,adaptive_enabled=False,bandit=None),
        volatility=replace(config.volatility,candle_interval=timedelta(hours=1),candle_lookback=150),
        risk=replace(config.risk,risk_per_trade_percent=1.,reward_risk_ratio=2.3,trailing_start_rr_ratio=1.))


if __name__ == "__main__":
    sys.argv[1:1] = ["--symbols","EURUSD","GBPUSD","USDCAD","USDCHF",
                    "--start-date","2020-01-01","--end-date","2025-12-31"]
    main(regression_config=hybrid_config())
