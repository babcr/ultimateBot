#!/usr/bin/env python
"""Convertit les exports historiques JForex en parquet mensuel sous C:\\ticks.

Source attendue par defaut:
    C:\\jforex_history\\EURUSD\\2020\\01\\2020-01-01.tsv

Chaque fichier journalier contient des trames START/QUOTE/STOP ecrites par
 DukascopyHistoricalTickExport.java. Les jours sans ticks existent quand meme,
 avec seulement START/STOP, afin que la reprise soit deterministe.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ITick import QuoteTick
from download_dukascopy_ticks import (  # noqa: E402
    DEFAULT_BATCH_ROWS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    MonthlyQuoteWriter,
    _iter_month_windows,
    _load_parquet_modules,
    _normalize_symbols,
    _parse_date,
    _point_scale,
    _strict_resume_downloads,
    _target_file,
)

DEFAULT_SOURCE_DIR = Path(r"C:\jforex_history")
DEFAULT_SYMBOLS = ("USDCHF", "GBPUSD", "USDCAD", "EURUSD")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FRAME_TYPES = {"START", "STOP", "HEARTBEAT", "QUOTE"}


def _day_floor_utc(moment: datetime | None = None) -> datetime:
    current = datetime.now(timezone.utc) if moment is None else moment.astimezone(timezone.utc)
    return current.replace(hour=0, minute=0, second=0, microsecond=0)


def _source_end_exclusive(end_date: date | None) -> datetime:
    latest_complete_day = _day_floor_utc()
    if end_date is None:
        return latest_complete_day
    requested = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return min(requested, latest_complete_day)


def _day_file(source_dir: Path, symbol: str, day_start: datetime) -> Path:
    return (
        source_dir
        / symbol
        / f"{day_start.year:04d}"
        / f"{day_start.month:02d}"
        / f"{day_start.year:04d}-{day_start.month:02d}-{day_start.day:02d}.tsv"
    )


def _temp_day_file(source_dir: Path, symbol: str, day_start: datetime) -> Path:
    day_path = _day_file(source_dir, symbol, day_start)
    return day_path.with_suffix(day_path.suffix + ".part")


def _iter_days(start: datetime, end: datetime):
    cursor = start
    while cursor < end:
        yield cursor
        cursor += timedelta(days=1)


def _validate_source_month(
    source_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
) -> str | None:
    for day_start in _iter_days(start, end):
        temp_path = _temp_day_file(source_dir, symbol, day_start)
        if temp_path.exists():
            return f"export temporaire residuel present: {temp_path}"
        day_path = _day_file(source_dir, symbol, day_start)
        if not day_path.exists():
            return f"jour source absent: {day_path}"
        if day_path.stat().st_size <= 0:
            return f"jour source vide: {day_path}"
    return None


def _parse_export_line(line: str, symbol: str) -> tuple[str, QuoteTick | None]:
    cells = line.rstrip("\r\n").split("\t")
    if len(cells) != 9:
        raise ValueError("trame JForex incomplete")
    kind, frame_symbol, stamp_text, session, seq_text, bid_text, ask_text, bid_size_text, ask_size_text = cells
    if not session:
        raise ValueError("session JForex absente")
    if frame_symbol != symbol:
        raise ValueError(f"symbole inattendu {frame_symbol!r}")
    if kind not in FRAME_TYPES:
        raise ValueError(f"type de trame inattendu {kind!r}")
    try:
        timestamp_ms = int(stamp_text)
        sequence = int(seq_text)
    except ValueError as exc:
        raise ValueError("horodatage ou sequence invalides") from exc
    if sequence < 0 or timestamp_ms < 0:
        raise ValueError("horodatage ou sequence negatifs")
    try:
        bid = float(bid_text)
        ask = float(ask_text)
        bid_size = float(bid_size_text)
        ask_size = float(ask_size_text)
    except ValueError as exc:
        raise ValueError("bid/ask/volumes invalides") from exc
    if not all(math.isfinite(value) for value in (bid, ask, bid_size, ask_size)):
        raise ValueError("valeurs non finies dans la trame")
    if kind != "QUOTE":
        if any(value != 0 for value in (bid, ask, bid_size, ask_size)):
            raise ValueError("valeurs de controle invalides")
        return kind, None
    if bid <= 0 or ask < bid or bid_size < 0 or ask_size < 0:
        raise ValueError("bid/ask/volumes incoherents")
    quote = QuoteTick(
        symbol,
        EPOCH + timedelta(milliseconds=timestamp_ms),
        bid,
        ask,
        bid_size=bid_size,
        ask_size=ask_size,
        source_id="dukascopy:jforex:history",
        sequence=sequence,
    )
    return kind, quote


def _iter_day_quotes(day_path: Path, symbol: str, day_start: datetime):
    """Valide START/QUOTE/STOP et les sequences avant publication du parquet."""
    lower_ms = int((day_start - EPOCH).total_seconds()) * 1000
    upper_ms = lower_ms + 86_400_000
    session: str | None = None
    sequence = 0
    previous_timestamp_ms = lower_ms
    saw_stop = False
    line_number = 0
    with day_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                if not line.endswith("\n"):
                    raise ValueError("derniere trame tronquee (fin de ligne absente)")
                if saw_stop:
                    raise ValueError("trame apres STOP")
                kind, quote = _parse_export_line(line, symbol)
                cells = line.rstrip("\r\n").split("\t")
                timestamp_ms, frame_sequence = int(cells[2]), int(cells[4])
                if session is None:
                    if kind != "START" or timestamp_ms != lower_ms or frame_sequence != 0:
                        raise ValueError("START initial invalide ou absent")
                    session = cells[3]
                    continue
                if cells[3] != session:
                    raise ValueError("session modifiee en cours de jour")
                if kind == "QUOTE":
                    if frame_sequence != sequence + 1:
                        raise ValueError("sequence manquante ou dupliquee")
                    if not lower_ms <= timestamp_ms < upper_ms:
                        raise ValueError("quote hors du jour UTC")
                    if timestamp_ms < previous_timestamp_ms:
                        raise ValueError("timestamps decroissants")
                    sequence = frame_sequence
                    previous_timestamp_ms = timestamp_ms
                    assert quote is not None
                    yield quote
                elif kind == "STOP":
                    if timestamp_ms != upper_ms - 1 or frame_sequence != sequence:
                        raise ValueError("STOP invalide")
                    saw_stop = True
                else:
                    raise ValueError("type de trame inattendu apres START")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{day_path}:{line_number}: {exc}") from exc
    if session is None or not saw_stop:
        raise ValueError(f"{day_path}:{line_number + 1}: jour source incomplet (START/STOP manquants)")


def _write_month_from_exports(
    source_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    batch_rows: int,
) -> int:
    pa, pq = _load_parquet_modules()
    writer = MonthlyQuoteWriter(output_path, symbol, _point_scale(symbol))
    try:
        for day_start in _iter_days(start, end):
            day_path = _day_file(source_dir, symbol, day_start)
            for quote in _iter_day_quotes(day_path, symbol, day_start):
                writer.append(quote)
                if writer.pending_rows() >= batch_rows:
                    writer.flush(pa, pq)

        writer.flush(pa, pq)
        writer.finalize()
        return writer.rows_written
    except Exception:
        writer.abort()
        raise


def _import_month(
    source_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    batch_rows: int,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        print(f"[skip] {symbol} {start:%Y-%m} -> {output_path}")
        return -1

    source_problem = _validate_source_month(source_dir, symbol, start, end)
    if source_problem is not None:
        raise RuntimeError(f"Source JForex incomplete pour {symbol} {start:%Y-%m}: {source_problem}")

    rows = _write_month_from_exports(source_dir, symbol, start, end, output_path, batch_rows)
    print(f"[done] {symbol} {start:%Y-%m} rows={rows:,} file={output_path}")
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convertit les exports historiques JForex en base parquet mensuelle."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Dossier racine des exports JForex journaliers (defaut: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier racine de la base parquet (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=DEFAULT_START_DATE,
        help=f"Date de debut incluse YYYY-MM-DD (defaut: {DEFAULT_START_DATE.isoformat()})",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Date de fin incluse YYYY-MM-DD. Par defaut: dernier jour UTC complet disponible.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Liste de paires separees par des virgules.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verifie tous les TSV sans creer de parquet (pyarrow non requis).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reecrit les fichiers parquet existants.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"Nombre de rows par flush parquet (defaut: {DEFAULT_BATCH_ROWS}).",
    )
    parser.add_argument(
        "--limit-months",
        type=int,
        default=None,
        help="Limite le nombre de mois traites pour un smoke test.",
    )
    parser.add_argument(
        "--resume-strict",
        action="store_true",
        help=(
            "Ne reprend qu'un prefixe contigu valide de mois existants. "
            "S'arrete si un mois plus tardif existe apres une lacune ou un parquet invalide."
        ),
    )
    parser.add_argument(
        "--resume-repair-first-invalid",
        action="store_true",
        help=(
            "Avec --resume-strict, supprime puis reecrit automatiquement le premier mois invalide "
            "deja present avant de poursuivre la validation stricte."
        ),
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.batch_rows <= 0:
        raise ValueError("--batch-rows doit etre strictement positif")
    if args.resume_repair_first_invalid and not args.resume_strict:
        raise ValueError("--resume-repair-first-invalid exige --resume-strict")
    if args.resume_repair_first_invalid and args.overwrite:
        raise ValueError("--resume-repair-first-invalid est incompatible avec --overwrite")
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Dossier source absent: {args.source_dir}")

    symbols = _normalize_symbols(args.symbols)
    end_exclusive = _source_end_exclusive(args.end_date)
    start_inclusive = datetime.combine(args.start_date, datetime.min.time(), tzinfo=timezone.utc)
    if start_inclusive >= end_exclusive:
        raise ValueError("La fenetre demandee est vide")

    if args.validate_only:
        days_checked = quotes_checked = 0
        for symbol in symbols:
            for month_start, month_end in _iter_month_windows(args.start_date, end_exclusive):
                problem = _validate_source_month(args.source_dir, symbol, month_start, month_end)
                if problem is not None:
                    raise ValueError(problem)
                month_quotes = 0
                for day_start in _iter_days(month_start, month_end):
                    month_quotes += sum(1 for _ in _iter_day_quotes(
                        _day_file(args.source_dir, symbol, day_start), symbol, day_start))
                    days_checked += 1
                quotes_checked += month_quotes
                print(f"[valid] {symbol} {month_start:%Y-%m} quotes={month_quotes:,}")
        print(f"[summary] validated_days={days_checked} validated_quotes={quotes_checked:,}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
                print(f"[resume-strict] {symbol}: rien a reprendre")
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

            result = _import_month(
                source_dir=args.source_dir,
                symbol=symbol,
                start=month_start,
                end=month_end,
                output_path=target,
                batch_rows=args.batch_rows,
                overwrite=args.overwrite,
            )
            months_processed += 1
            if result >= 0:
                files_written += 1
                rows_written += result

    print(f"[summary] files={files_written} rows={rows_written:,} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
