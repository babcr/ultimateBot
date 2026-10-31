"""Fixed breakout/retest baseline: py -m ultimateBot.run_regression_breakout."""
from dataclasses import replace
from datetime import timedelta
import sys

from ultimateBot.backtester.strategy.swing_regression_config import DEFAULT_SWING_REGRESSION_CONFIG
from ultimateBot.run_backtest import main


def breakout_config():
    config = DEFAULT_SWING_REGRESSION_CONFIG
    return replace(
        config,
        regression=replace(config.regression, entry_mode="breakout-retest", lookback_bars=15),
        execution=replace(config.execution, trailing_stop=True),
        optimization=replace(config.optimization, adaptive_enabled=False, bandit=None),
        volatility=replace(config.volatility, candle_interval=timedelta(minutes=10), candle_lookback=150),
        risk=replace(config.risk, risk_per_trade_percent=2.2, reward_risk_ratio=3,
                     trailing_start_rr_ratio=1.5),
    )


if __name__ == "__main__":
    sys.argv[1:1] = ["--symbols", "USDCHF", "GBPUSD", "USDCAD", "EURUSD",
                    "--start-date", "2020-01-01", "--end-date", "2023-12-31", "--session-filter"]
    main(regression_config=breakout_config())
