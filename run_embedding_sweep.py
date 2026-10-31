#!/usr/bin/env python3
"""Replay the requested swing-regression configuration for an inclusive seed range."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
DATA = Path("C:/TradingBotData")
FIXED_ARGUMENTS = [
    "--strategy", "swing-regression", "--symbols", "USDCHF", "GBPUSD", "USDCAD", "EURUSD",
    "--candle-lookback", "150", "--reward-risk-ratio", "3.",
    "--level-lookback-bars", "1", "--regression-lookback-bars", "15",
    "--risk-percent", ".01", "--trailing-stop", "--trailing-start-rr-ratio", "2.99",
    "--adaptive-strategy", "--no-adaptive-tighten-search-interval",
    "--adaptive-optimization-metric", "profit_factor",
    "--no-adaptive-optimize-global-timeframe-validation", "--no-regression-global-timeframe-validation",
    "--no-adaptive-optimize-risk", "--adaptive-lookback-max", "20",
    "--no-regression-level-reaction", # "--session-filter",
    "--regression-min-market-volatility-ratio", "0.0001"
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1, help="Inclusive first adaptive seed")
    parser.add_argument("--seed-end", type=int, default=10, help="Inclusive last adaptive seed")
    parser.add_argument("--passes", type=int, default=1, help="Repeat the entire range, adding fresh runs each time")
    parser.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--trade-database", type=Path, default=DATA / "trade_embeddings.sqlite3")
    parser.add_argument("--ts2vec-model", type=Path, default=DATA / "models" / "ts2vec_order_creation.pt")
    parser.add_argument("--ts2vec-train-iters", type=int, default=200)
    parser.add_argument("--ts2vec-device", default="cpu")
    parser.add_argument("--trade-validator", type=Path, default=None, help="Optional XGBoost order-creation validator on a later period")
    parser.add_argument("--trade-validator-threshold", type=float, default=None)
    parser.add_argument("--log-dir", type=Path, default=DATA / "logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or creating data")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.seed_start > args.seed_end or args.passes < 1 or args.ts2vec_train_iters < 1:
        parser.error("Require seed-start <= seed-end, passes >= 1 and train-iters >= 1")
    if args.trade_validator_threshold is not None and (args.trade_validator is None or not 0 <= args.trade_validator_threshold <= 1):
        parser.error("--trade-validator-threshold requires --trade-validator and a threshold in [0,1]")
    try:
        start, end = datetime.fromisoformat(args.start_date).date(), datetime.fromisoformat(args.end_date).date()
        if start > end:
            raise ValueError("start-date is after end-date")
    except ValueError as error:
        parser.error(str(error))
    return args


def command_for(args, seed, campaign_id, run_id):
    command = [sys.executable, "-u", "-m", "ultimateBot.run_backtest", *FIXED_ARGUMENTS,
            "--history", str(args.history.resolve()),
            "--start-date", args.start_date, "--end-date", args.end_date,
            "--adaptive-seed", str(seed),
            "--trade-database", str(args.trade_database.resolve()),
            "--ts2vec-model", str(args.ts2vec_model.resolve()),
            "--ts2vec-train-iters", str(args.ts2vec_train_iters),
            "--ts2vec-device", args.ts2vec_device,
            "--dataset-campaign-id", campaign_id, "--dataset-run-id", run_id]
    if args.trade_validator is not None:
        command.extend(["--trade-validator", str(args.trade_validator.resolve())])
    if args.trade_validator_threshold is not None:
        command.extend(["--trade-validator-threshold", str(args.trade_validator_threshold)])
    return command


def mark_unfinished(database, run_id, status, error):
    """Also handle a killed interpreter, which cannot execute its engine's finally path."""
    if not database.exists():
        return
    with closing(sqlite3.connect(database, timeout=60)) as connection:
        with connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone():
                connection.execute("UPDATE runs SET status=?,error=?,finished_at=? WHERE run_id=? AND status='running'",
                                   (status, error, datetime.now(timezone.utc).isoformat(), run_id))


def main(argv=None):
    args = parse_args(argv)
    campaign_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    directory = args.log_dir.resolve() / campaign_id
    manifest = {"campaign_id": campaign_id, "database": str(args.trade_database.resolve()),
                "seed_start": args.seed_start, "seed_end": args.seed_end, "passes": args.passes,
                "runs": [], "status": "running"}
    if not args.dry_run:
        for symbol in ("USDCHF", "GBPUSD", "USDCAD", "EURUSD"):
            if not (args.history / symbol).is_dir():
                raise FileNotFoundError(f"History missing: {args.history / symbol}")
        directory.mkdir(parents=True, exist_ok=False)

    def save_manifest():
        if not args.dry_run:
            temporary = directory / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(directory / "manifest.json")

    print(f"Campaign: {campaign_id}\nDatabase: {args.trade_database.resolve()}\nLogs: {directory}", flush=True)
    save_manifest()
    failures = 0
    for pass_number in range(1, args.passes + 1):
        for seed in range(args.seed_start, args.seed_end + 1):
            run_id = uuid4().hex
            command = command_for(args, seed, campaign_id, run_id)
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            logfile = directory / f"pass_{pass_number:02d}_seed_{seed}_{run_id}.log"
            record = {"pass": pass_number, "seed": seed, "run_id": run_id,
                      "log": str(logfile), "command": command, "status": "running"}
            manifest["runs"].append(record)
            save_manifest()
            print(f"[sweep-start] pass={pass_number} seed={seed} run_id={run_id} log={logfile}", flush=True)
            with logfile.open("w", encoding="utf-8") as output:
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=output, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                save_manifest()
                try:
                    code = child.wait()
                except KeyboardInterrupt:
                    # The child shares the terminal and normally receives Ctrl+C too.
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait(timeout=10)
                    mark_unfinished(args.trade_database, run_id, "interrupted", "Seed sweep interrupted")
                    record["status"] = manifest["status"] = "interrupted"
                    save_manifest()
                    return 130
            record.update(returncode=code, status="completed" if code == 0 else "failed")
            save_manifest()
            print(f"[sweep-finish] seed={seed} status={record['status']} log={logfile}", flush=True)
            if code:
                failures += 1
                mark_unfinished(args.trade_database, run_id, "failed", f"Backtest interpreter exited with code {code}; see {logfile}")
                if not args.continue_on_error:
                    manifest["status"] = "failed"
                    save_manifest()
                    return 1
    manifest["status"] = "failed" if failures else "completed"
    save_manifest()
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
