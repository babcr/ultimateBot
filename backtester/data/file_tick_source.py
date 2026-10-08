from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator
import warnings

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.data.tick_source import TickSource


class _MalformedTickLineError(ValueError):
    """Raised when a non-empty line cannot be parsed as a tick frame."""


_DECODE_REPLACEMENT_CHAR = "\ufffd"


def _warn_skipped_lines(file_path: Path, count: int, first_line_number: int | None, first_reason: str | None) -> None:
    if count <= 0 or first_line_number is None or first_reason is None:
        return
    warnings.warn(
        f"Skipped {count} malformed tick lines in {file_path} "
        f"(first at line {first_line_number}: {first_reason})."
    )


class JForexTickSource(TickSource):
    """Tick source that reads tick data from a JForex file."""

    def __init__(self, base_path: Path, symbol: str, start_date: date | None = None, end_date: date | None = None):
        self.base_path = Path(base_path)
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(
                f"Invalid tick-source date bounds for {self.symbol}: "
                f"{self.start_date.isoformat()} > {self.end_date.isoformat()}"
            )

    @property
    def symbol_path(self) -> Path:
        return self.base_path / self.symbol

    def __iter__(self) -> Iterator[Tick]:
        for file_path in self.__iter_files():
            yield from self.__iter_file(file_path)

    def __iter_files(self) -> Iterator[Path]:
        """
        Parcours :
        base/
           SYMBOL/
              YYYY/
                 MM/
                    YYYY-MM-DD.tsv
        dans l'ordre chronologique.

        Yields:
            Iterator[Path]: Paths to the tick data files in chronological order.
        """
        symbol_path = self.symbol_path
        if not symbol_path.is_dir():
            raise FileNotFoundError(f"Symbol directory not found: {symbol_path}")

        for year_path in sorted(symbol_path.iterdir()):
            if not year_path.is_dir():
                continue
            try:
                year = int(year_path.name)
            except ValueError:
                continue
            if self.start_date is not None and year < self.start_date.year:
                continue
            if self.end_date is not None and year > self.end_date.year:
                continue
            for month_path in sorted(year_path.iterdir()):
                if not month_path.is_dir():
                    continue
                for file_path in sorted(month_path.glob("*.tsv")):
                    file_date = self._parse_file_date(file_path)
                    if file_date is None:
                        continue
                    if self.start_date is not None and file_date < self.start_date:
                        continue
                    if self.end_date is not None and file_date > self.end_date:
                        continue
                    yield file_path

    @staticmethod
    def _parse_file_date(file_path: Path) -> date | None:
        try:
            return date.fromisoformat(file_path.stem)
        except ValueError:
            return None

    def __iter_file(self, file_path: Path) -> Iterator[Tick]:
        malformed_count = 0
        first_malformed_line: int | None = None
        first_malformed_reason: str | None = None

        with file_path.open("r", encoding="utf-8-sig", errors="replace", buffering=1024 * 1024) as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    tick = self._parse_line(line)
                except _MalformedTickLineError as error:
                    malformed_count += 1
                    if first_malformed_line is None:
                        first_malformed_line = line_number
                        first_malformed_reason = str(error)
                    continue

                if tick is not None:
                    yield tick

        _warn_skipped_lines(
            file_path=file_path,
            count=malformed_count,
            first_line_number=first_malformed_line,
            first_reason=first_malformed_reason,
        )

    def _parse_line(self, line: str) -> Tick | None:
        if _DECODE_REPLACEMENT_CHAR in line:
            raise _MalformedTickLineError("invalid utf-8 payload")

        line = line.strip()
        if not line:
            return None

        fields = line.split()
        if not fields:
            return None
        if len(fields) != 9:
            raise _MalformedTickLineError(f"expected 9 fields, got {len(fields)}")

        frame_type = fields[0].upper()
        symbol = fields[1]
        if symbol != self.symbol:
            raise _MalformedTickLineError(f"expected symbol {self.symbol}, got {symbol}")
        if frame_type in ("START", "STOP"):
            return None
        if frame_type != "QUOTE":
            raise _MalformedTickLineError(f"unexpected frame type {frame_type}")

        try:
            timestamp_ms = int(fields[2])
            bid = float(fields[5])
            ask = float(fields[6])
            bid_volume = float(fields[7])
            ask_volume = float(fields[8])
        except (ValueError, IndexError) as error:
            raise _MalformedTickLineError(f"invalid numeric payload: {error}") from error

        if bid <= 0.0 or ask <= 0.0:
            return None

        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return Tick(
            timestamp=timestamp,
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )


class FileTickSource(TickSource):
    """Tick source that reads tick data from a file."""

    def __init__(self, symbol: str, path: Path):
        self.symbol = symbol
        self.path = path
        self.number = 0
        self.frame_number = 0

    def __iter__(self) -> Iterator[Tick]:
        malformed_count = 0
        first_malformed_line: int | None = None
        first_malformed_reason: str | None = None

        with self.path.open("r", encoding="utf-8-sig", errors="replace", buffering=1024 * 1024) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    tick = self._parse_line(line)
                except _MalformedTickLineError as error:
                    malformed_count += 1
                    if first_malformed_line is None:
                        first_malformed_line = line_number
                        first_malformed_reason = str(error)
                    continue

                if tick is not None:
                    yield tick

        _warn_skipped_lines(
            file_path=self.path,
            count=malformed_count,
            first_line_number=first_malformed_line,
            first_reason=first_malformed_reason,
        )

    def _parse_line(self, line: str) -> Tick | None:
        if _DECODE_REPLACEMENT_CHAR in line:
            raise _MalformedTickLineError("invalid utf-8 payload")

        fields = line.rstrip().split()
        if not fields:
            return None
        if len(fields) != 9:
            raise _MalformedTickLineError(f"expected 9 fields, got {len(fields)}")

        frame_type, symbol, timestamp_str, _key, _rank, bid_str, ask_str, bid_volume_str, ask_volume_str = fields

        self.frame_number += 1
        if symbol != self.symbol or frame_type != "QUOTE":
            return None

        self.number += 1
        try:
            return Tick(
                timestamp=datetime.fromisoformat(timestamp_str),
                symbol=self.symbol,
                bid=float(bid_str),
                ask=float(ask_str),
                bid_volume=float(bid_volume_str),
                ask_volume=float(ask_volume_str),
            )
        except ValueError as error:
            raise _MalformedTickLineError(f"invalid tick payload: {error}") from error