#!/usr/bin/env python3
"""Train a versioned XGBoost trade validator from committed SQLite trade closes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimateBot.backtester.ml.train_validator import InsufficientTrainingData, TrainingConfig, train_validator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("C:/TradingBotData/trade_embeddings.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=None, help="New artifact directory; default creates a version under C:/TradingBotData/models/trade_validators")
    parser.add_argument("--embedding-model-id", default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--completed-runs-only", action="store_true", help="Otherwise also use committed closes from running collection runs")
    parser.add_argument("--cutoff", default=None, help="Latest permitted close time, ISO-8601; naive timestamps mean UTC")
    parser.add_argument("--wait-for-data", action="store_true", help="Wait for enough compatible closed trades from the collector, then train once")
    parser.add_argument("--poll-seconds", type=float, default=30, help="Collection polling interval in wait mode, at least one second")
    defaults = TrainingConfig()
    for name in ("train_fraction", "validation_fraction", "target_r", "threshold", "learning_rate"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    for name in ("min_train_samples", "min_validation_samples", "min_test_samples", "min_class_samples",
                 "min_accepted_validation", "n_estimators", "max_depth", "early_stopping_rounds", "seed", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 3600:
        parser.error("--poll-seconds must be in [1,3600]")
    try:
        config = TrainingConfig(**{name: getattr(args, name) for name in defaults.__dataclass_fields__})
    except ValueError as error:
        parser.error(str(error))
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    while True:
        try:
            directory, metadata, report = train_validator(
                args.database, config=config, output_dir=args.output_dir, model_id=args.embedding_model_id,
                campaign_id=args.campaign_id, completed_only=args.completed_runs_only, cutoff=args.cutoff)
            break
        except InsufficientTrainingData as error:
            if not args.wait_for_data:
                raise
            print(f"[waiting-for-order-creation-data] {error}", flush=True)
            time.sleep(args.poll_seconds)
    print(f"Model: {directory}")
    print(f"Threshold: {metadata['threshold']:.3f}; target: PnL/R > {config.target_r:g}")
    print(f"Use only after: {metadata['available_after']}")
    print(json.dumps(report["partitions"]["test"], indent=2))
    print(f'Activate on a future backtest with: --trade-validator "{directory}"')


if __name__ == "__main__":
    main()
