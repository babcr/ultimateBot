#!/usr/bin/env python
"""Télécharge l'historique Dukascopy en parquet mensuel sous C:\\ticks.

Disposition écrite par défaut:
    C:\\ticks\\EURUSD\\2020\\01.parquet

Chaque fichier contient des quote ticks FX compressés en points entiers:
    - timestamp_ms: horodatage UTC en millisecondes
    - bid_points: bid * point_scale
    - spread_points: (ask - bid) * point_scale
    - bid_size / ask_size: tailles de quote en float32

Le module DukascopyTick émet aussi des TradeTick synthétiques dérivés du midquote.
Ils ne sont pas persistés ici pour éviter de doubler le volume disque avec une
information artificielle et reconstructible.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from DukascopyTick import DukascopyTick
from ITick import QuoteTick

DEFAULT_OUTPUT_DIR = Path(r"D:\ticks")
DEFAULT_CACHE_DIR = Path(r"D:\dukascopy_cache")
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_SYMBOLS = (
    "EURGBP",
    "EURCHF",
    "EURUSD",
    "EURCAD",
    "EURAUD",
    "EURNZD",
    "USDCHF",
    "USDCAD",
    "GBPUSD",
    "GBPCAD",
    "GBPCHF",
    "GBPAUD",
    "GBPNZD",
)
DEFAULT_BATCH_ROWS = 250_000
PARQUET_SCHEMA_VERSION = "1"
MONTH_RETRY_DELAYS = (30.0, 60.0, 120.0, 300.0)
REQUIRED_COLUMNS = (
    "timestamp_ms",
    "bid_points",
    "spread_points",
    "bid_size",
    "ask_size",
)


def _load_parquet_modules() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeError(
            "Le script nécessite pyarrow. Installer avec: pip install pyarrow"
        ) from exc
    return pa, pq


def _utc_now_full_hour() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Date invalide: {value!r}. Format attendu: YYYY-MM-DD"
        ) from exc


def _month_floor(day_value: date) -> datetime:
    return datetime(day_value.year, day_value.month, 1, tzinfo=timezone.utc)


def _exclusive_end(end_date: date | None) -> datetime:
    latest_available = _utc_now_full_hour()
    if end_date is None:
        return latest_available
    requested = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return min(requested, latest_available)


def _month_after(moment: datetime) -> datetime:
    if moment.month == 12:
        return datetime(moment.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(moment.year, moment.month + 1, 1, tzinfo=timezone.utc)


def _iter_month_windows(start_date: date, end_exclusive: datetime):
    cursor = _month_floor(start_date)
    lower_bound = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    while cursor < end_exclusive:
        next_cursor = _month_after(cursor)
        yield max(cursor, lower_bound), min(next_cursor, end_exclusive)
        cursor = next_cursor


def _point_scale(symbol: str) -> int:
    return 1_000 if symbol.endswith("JPY") else 100_000


def _timestamp_ms(moment: datetime) -> int:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Horodatage naïf inattendu")
    delta = moment.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000


def _target_file(output_dir: Path, symbol: str, month_start: datetime) -> Path:
    return output_dir / symbol / f"{month_start.year:04d}" / f"{month_start.month:02d}.parquet"


def _temp_file(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".part")


def _has_month_artifacts(output_path: Path) -> bool:
    return output_path.exists() or _temp_file(output_path).exists()


def _purge_month_artifacts(output_path: Path) -> list[Path]:
    removed: list[Path] = []
    for candidate in (output_path, _temp_file(output_path)):
        if candidate.exists():
            candidate.unlink()
            removed.append(candidate)
    return removed


def _read_edge_timestamp(column: Any, *, first: bool) -> int:
    chunks = column.chunks
    ordered_chunks = chunks if first else reversed(chunks)
    for chunk in ordered_chunks:
        values = chunk.to_pylist()
        if not values:
            continue
        return int(values[0] if first else values[-1])
    raise ValueError("Colonne timestamp_ms vide malgré un parquet non vide")


def _validate_existing_month(
    output_path: Path,
    symbol: str,
    start: datetime,
    end: datetime,
) -> str | None:
    temp_path = _temp_file(output_path)
    if temp_path.exists():
        return f"fichier temporaire résiduel présent: {temp_path}"
    if not output_path.exists():
        return "fichier absent"

    _, pq = _load_parquet_modules()
    try:
        parquet_file = pq.ParquetFile(output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"parquet illisible: {exc}"

    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows <= 0:
        return "parquet vide"
    if metadata.num_columns != len(REQUIRED_COLUMNS):
        return f"nombre de colonnes inattendu: {metadata.num_columns}"

    schema = parquet_file.schema_arrow
    if tuple(schema.names) != REQUIRED_COLUMNS:
        return f"colonnes inattendues: {schema.names}"
    schema_meta = schema.metadata or {}
    expected_symbol = symbol.encode("ascii")
    expected_scale = str(_point_scale(symbol)).encode("ascii")
    if schema_meta.get(b"symbol") != expected_symbol:
        return "metadata symbol incohérente"
    if schema_meta.get(b"point_scale") != expected_scale:
        return "metadata point_scale incohérente"
    if schema_meta.get(b"schema_version") != PARQUET_SCHEMA_VERSION.encode("ascii"):
        return "schema_version incompatible"

    try:
        first_table = parquet_file.read_row_group(0, columns=["timestamp_ms"])
        last_table = parquet_file.read_row_group(parquet_file.num_row_groups - 1, columns=["timestamp_ms"])
    except (OSError, RuntimeError, ValueError) as exc:
        return f"lecture timestamp impossible: {exc}"

    try:
        first_stamp = _read_edge_timestamp(first_table.column("timestamp_ms"), first=True)
        last_stamp = _read_edge_timestamp(last_table.column("timestamp_ms"), first=False)
    except (KeyError, ValueError, TypeError) as exc:
        return f"timestamp_ms invalide: {exc}"

    lower = _timestamp_ms(start)
    upper = _timestamp_ms(end)
    if not (lower <= first_stamp < upper):
        return "premier timestamp hors fenêtre"
    if not (lower <= last_stamp < upper):
        return "dernier timestamp hors fenêtre"
    if last_stamp < first_stamp:
        return "timestamps décroissants"
    return None


def _strict_resume_downloads(
    output_dir: Path,
    symbol: str,
    start_date: date,
    end_exclusive: datetime,
    repair_first_invalid: bool = False,
) -> list[tuple[datetime, datetime, Path]]:
    downloads: list[tuple[datetime, datetime, Path]] = []
    frontier_reason: str | None = None
    frontier_month: datetime | None = None
    repaired_month: datetime | None = None
    for month_start, month_end in _iter_month_windows(start_date, end_exclusive):
        target = _target_file(output_dir, symbol, month_start)
        problem = _validate_existing_month(target, symbol, month_start, month_end)
        has_artifacts = _has_month_artifacts(target)
        if frontier_reason is None:
            if problem is None:
                print(f"[resume-strict] ok {symbol} {month_start:%Y-%m} -> {target}")
                continue
            if repair_first_invalid and repaired_month is None and has_artifacts:
                removed = _purge_month_artifacts(target)
                repaired_month = month_start
                downloads.append((month_start, month_end, target))
                removed_text = ", ".join(str(path) for path in removed) or str(target)
                print(
                    f"[resume-strict] réparation {symbol} {month_start:%Y-%m}: {problem}; supprimé {removed_text}"
                )
                continue
            frontier_reason = problem
            frontier_month = month_start
            downloads.append((month_start, month_end, target))
            print(
                f"[resume-strict] reprise {symbol} {month_start:%Y-%m}: {problem}"
            )
            continue
        if has_artifacts:
            raise RuntimeError(
                "Mode reprise stricte: un mois existe après la première lacune ou corruption "
                f"pour {symbol}. Premier mois à reprendre: {frontier_month:%Y-%m} ({frontier_reason}). "
                f"Mois bloquant: {month_start:%Y-%m} -> {target}. Utiliser --overwrite ou corriger la série."
            )
        downloads.append((month_start, month_end, target))
    return downloads


class MonthlyQuoteWriter:
    def __init__(self, output_path: Path, symbol: str, point_scale: int):
        self.output_path = output_path
        self.temp_path = output_path.with_suffix(output_path.suffix + ".part")
        self.symbol = symbol
        self.point_scale = point_scale
        self._timestamp_ms: list[int] = []
        self._bid_points: list[int] = []
        self._spread_points: list[int] = []
        self._bid_size: list[float] = []
        self._ask_size: list[float] = []
        self.rows_written = 0
        self._writer: Any | None = None

    def append(self, quote: QuoteTick) -> None:
        bid_points = int(round(quote.bid * self.point_scale))
        ask_points = int(round(quote.ask * self.point_scale))
        self._timestamp_ms.append(_timestamp_ms(quote.timestamp))
        self._bid_points.append(bid_points)
        self._spread_points.append(ask_points - bid_points)
        self._bid_size.append(float(quote.bid_size or 0.0))
        self._ask_size.append(float(quote.ask_size or 0.0))

    def pending_rows(self) -> int:
        return len(self._timestamp_ms)

    def _schema(self, pa: Any) -> Any:
        base = pa.schema(
            [
                pa.field("timestamp_ms", pa.int64(), nullable=False),
                pa.field("bid_points", pa.int32(), nullable=False),
                pa.field("spread_points", pa.int32(), nullable=False),
                pa.field("bid_size", pa.float32(), nullable=False),
                pa.field("ask_size", pa.float32(), nullable=False),
            ]
        )
        return base.with_metadata(
            {
                b"symbol": self.symbol.encode("ascii"),
                b"point_scale": str(self.point_scale).encode("ascii"),
                b"price_encoding": b"bid_points + spread_points",
                b"schema_version": PARQUET_SCHEMA_VERSION.encode("ascii"),
                b"source": b"DukascopyTick.historical_ticks quotes-only",
            }
        )

    def flush(self, pa: Any, pq: Any) -> None:
        if not self._timestamp_ms:
            return
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.temp_path.exists():
                self.temp_path.unlink()
            self._writer = pq.ParquetWriter(
                self.temp_path,
                self._schema(pa),
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=False,
            )
        table = pa.table(
            {
                "timestamp_ms": pa.array(self._timestamp_ms, type=pa.int64()),
                "bid_points": pa.array(self._bid_points, type=pa.int32()),
                "spread_points": pa.array(self._spread_points, type=pa.int32()),
                "bid_size": pa.array(self._bid_size, type=pa.float32()),
                "ask_size": pa.array(self._ask_size, type=pa.float32()),
            },
            schema=self._schema(pa),
        )
        self._writer.write_table(table)
        self.rows_written += len(self._timestamp_ms)
        self._timestamp_ms.clear()
        self._bid_points.clear()
        self._spread_points.clear()
        self._bid_size.clear()
        self._ask_size.clear()

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.rows_written > 0 and self.temp_path.exists():
            self.temp_path.replace(self.output_path)
        elif self.temp_path.exists():
            self.temp_path.unlink()

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.temp_path.exists():
            self.temp_path.unlink()


def _write_month(
    source: DukascopyTick,
    symbol: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    batch_rows: int,
) -> int:
    pa, pq = _load_parquet_modules()
    writer = MonthlyQuoteWriter(output_path, symbol, _point_scale(symbol))
    try:
        for tick in source.historical_ticks(symbol, start, end):
            if not isinstance(tick, QuoteTick):
                continue
            writer.append(tick)
            if writer.pending_rows() >= batch_rows:
                writer.flush(pa, pq)
        writer.flush(pa, pq)
        writer.finalize()
        return writer.rows_written
    except Exception:
        writer.abort()
        raise


def _download_month(
    source: DukascopyTick,
    symbol: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    batch_rows: int,
    overwrite: bool,
    retries: int,
) -> int:
    if output_path.exists() and not overwrite:
        print(f"[skip] {symbol} {start:%Y-%m} -> {output_path}")
        return -1
    attempt = 0
    while True:
        attempt += 1
        try:
            rows = _write_month(source, symbol, start, end, output_path, batch_rows)
            print(f"[done] {symbol} {start:%Y-%m} rows={rows:,} file={output_path}")
            return rows
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            if attempt > retries:
                raise RuntimeError(
                    f"Echec sur {symbol} {start:%Y-%m} après {retries} tentative(s)"
                ) from exc
            print(f"[retry] {symbol} {start:%Y-%m} tentative {attempt}/{retries} après erreur: {exc}")
            delay = MONTH_RETRY_DELAYS[min(attempt - 1, len(MONTH_RETRY_DELAYS) - 1)]
            print(f"[retry-wait] {symbol} {start:%Y-%m} pause {delay:.0f}s avant nouvelle tentative")
            time_module.sleep(delay)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Télécharge les ticks Dukascopy historiques vers une base parquet mensuelle."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier racine de la base parquet (défaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache local des archives .bi5 (défaut: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=DEFAULT_START_DATE,
        help=f"Date de début incluse YYYY-MM-DD (défaut: {DEFAULT_START_DATE.isoformat()})",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Date de fin incluse YYYY-MM-DD. Par défaut: dernière heure UTC complète disponible.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Liste de paires séparées par des virgules.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Réécrit les fichiers parquet existants.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"Nombre de rows par flush parquet (défaut: {DEFAULT_BATCH_ROWS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Nombre de tentatives par mois avant abandon (défaut: 2).",
    )
    parser.add_argument(
        "--limit-months",
        type=int,
        default=None,
        help="Limite le nombre de mois traités pour un smoke test.",
    )
    parser.add_argument(
        "--resume-strict",
        action="store_true",
        help=(
            "Ne reprend qu'un préfixe contigu valide de mois existants. "
            "S'arrête si un mois plus tardif existe après une lacune ou un parquet invalide."
        ),
    )
    parser.add_argument(
        "--resume-repair-first-invalid",
        action="store_true",
        help=(
            "Avec --resume-strict, supprime puis réécrit automatiquement le premier mois invalide "
            "déjà présent avant de poursuivre la validation stricte."
        ),
    )
    return parser


def _normalize_symbols(symbols_arg: str) -> list[str]:
    symbols = []
    for raw in symbols_arg.split(","):
        symbol = raw.strip().replace("/", "").upper()
        if symbol:
            symbols.append(symbol)
    if not symbols:
        raise ValueError("Aucun symbole fourni")
    return symbols


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.batch_rows <= 0:
        raise ValueError("--batch-rows doit être strictement positif")
    if args.retries < 1:
        raise ValueError("--retries doit être >= 1")
    if args.resume_repair_first_invalid and not args.resume_strict:
        raise ValueError("--resume-repair-first-invalid exige --resume-strict")
    if args.resume_repair_first_invalid and args.overwrite:
        raise ValueError("--resume-repair-first-invalid est incompatible avec --overwrite")

    symbols = _normalize_symbols(args.symbols)
    end_exclusive = _exclusive_end(args.end_date)
    start_inclusive = datetime.combine(args.start_date, time.min, tzinfo=timezone.utc)
    if start_inclusive >= end_exclusive:
        raise ValueError("La fenêtre demandée est vide")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    source = DukascopyTick(
        bridge_dir=args.output_dir,
        cache_dir=args.cache_dir,
        timeout=60.0,
        history_request_interval=0.35,
    )

    months_processed = 0
    files_written = 0
    rows_written = 0
    for symbol in symbols:
        if args.resume_strict and not args.overwrite:
            month_windows = _strict_resume_downloads(
                output_dir=args.output_dir,
                symbol=symbol,
                start_date=args.start_date,
                end_exclusive=end_exclusive,
                repair_first_invalid=args.resume_repair_first_invalid,
            )
            if not month_windows:
                print(f"[resume-strict] {symbol}: rien à reprendre")
        else:
            month_windows = [
                (month_start, month_end, _target_file(args.output_dir, symbol, month_start))
                for month_start, month_end in _iter_month_windows(args.start_date, end_exclusive)
            ]

        for month_start, month_end, target in month_windows:
            if args.limit_months is not None and months_processed >= args.limit_months:
                print("[stop] limite de mois atteinte")
                print(f"[summary] files={files_written} rows={rows_written:,}")
                return 0
            result = _download_month(
                source=source,
                symbol=symbol,
                start=month_start,
                end=month_end,
                output_path=target,
                batch_rows=args.batch_rows,
                overwrite=args.overwrite,
                retries=args.retries,
            )
            months_processed += 1
            if result >= 0:
                files_written += 1
                rows_written += result

    print(f"[summary] files={files_written} rows={rows_written:,} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
