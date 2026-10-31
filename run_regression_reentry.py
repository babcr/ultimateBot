"""Fixed failed-breakout baseline. Run as py -m ultimateBot.run_regression_reentry."""
from dataclasses import replace
from datetime import timedelta
import sys

from ultimateBot.backtester.strategy.swing_regression_config import DEFAULT_SWING_REGRESSION_CONFIG
from ultimateBot.run_backtest import main


def reentry_config():
    config = DEFAULT_SWING_REGRESSION_CONFIG
    return replace(
        config,
        regression=replace(config.regression, entry_mode="failed-breakout", lookback_bars=15),
        execution=replace(config.execution, trailing_stop=False),
        optimization=replace(config.optimization, adaptive_enabled=False, bandit=None),
        volatility=replace(config.volatility, candle_interval=timedelta(minutes=10), candle_lookback=150),
        risk=replace(config.risk, risk_per_trade_percent=2.2),
    )


if __name__ == "__main__":
    # Later explicit CLI options override this reproducible starting configuration.
    sys.argv[1:1] = ["--symbols", "USDCHF", "GBPUSD", "USDCAD", "EURUSD",
                    "--start-date", "2020-01-01", "--end-date", "2023-12-31", "--session-filter"]
    main(regression_config=reentry_config())
