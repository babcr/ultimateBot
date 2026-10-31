#!/usr/bin/env python3
"""Export one frozen embedding model's completed runs into a flat CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def export_dataset(database, output, *, model_id=None, campaign_id=None):
    database, output = Path(database).resolve(), Path(output).resolve()
    if database == output:
        raise ValueError("CSV output must not overwrite the SQLite database")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if "feature_point" not in {row[1] for row in connection.execute("PRAGMA table_info(closed_trades)")}:
            raise ValueError("This database only has legacy fill-time features; collect order-creation features before exporting")
        where, params = "run_status='completed' AND feature_point='order_creation'", []
        if campaign_id is not None:
            where += " AND campaign_id=?"
            params.append(campaign_id)
        if model_id is not None:
            where += " AND model_id=?"
            params.append(model_id)
        models = connection.execute(f"SELECT DISTINCT model_id,embedding_dims FROM training_samples WHERE {where}", params).fetchall()
        if len(models) != 1:
            raise ValueError("Export requires exactly one model with closed trades in completed runs; specify --model-id/--campaign-id")
        dimensions = models[0]["embedding_dims"]
        metadata = ["run_id", "trade_id", "adaptive_seed", "campaign_id", "model_id", "feature_point", "symbol", "order_id", "order_created_at", "feature_observed_at", "opened_at", "closed_at",
                    "context_bars", "timeframe_minutes", "strategy_configuration_id"]
        features = [f"embedding_{i:03d}" for i in range(dimensions)] + ["regression_lookback", "trailing", "side"]
        targets = ["pnl", "initial_risk", "pnl_r", "won"]
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=metadata + features + targets)
            writer.writeheader()
            for row in connection.execute(f"SELECT * FROM training_samples WHERE {where} ORDER BY opened_at,run_id,trade_id", params):
                vector = json.loads(row["embedding_json"])
                if len(vector) != dimensions:
                    raise ValueError("Embedding dimension mismatch in database")
                record = {name: row[name] for name in metadata + targets + ["regression_lookback", "trailing", "side"]}
                record.update({f"embedding_{i:03d}": value for i, value in enumerate(vector)})
                writer.writerow(record)
                count += 1
        return count
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("C:/TradingBotData/trade_embeddings.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("C:/TradingBotData/exports/trades.csv"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--campaign-id", default=None)
    args = parser.parse_args()
    count = export_dataset(args.database, args.output, model_id=args.model_id, campaign_id=args.campaign_id)
    print(f"Exported {count} trades to {args.output.resolve()}")


if __name__ == "__main__":
    main()
