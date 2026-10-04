"""Dukascopy: historique .bi5 public + cotations JForex en direct.

Le FX au comptant Dukascopy expose ici des cotations, pas les transactions
exécutées du marché. Chaque TradeTick émis est donc SYNTHÉTIQUE, au midquote,
avec quantité inconnue; ne pas l'utiliser comme volume ou transaction réelle.

Le pont Java DukascopyTickBridge.java doit tourner dans JForex (même machine ou
répertoire partagé). Il peut écrire soit un fichier legacy `live_<SYMBOL>.tsv`,
soit un flux multiplexé `live_ticks.tsv` contenant tous les symboles en ordre
d'arrivée.
"""
from __future__ import annotations

import lzma
import math
import re
import struct
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ITick import ITick, QuoteTick, Tick, TradeTick

_RECORD = struct.Struct('>IIIff')
_BASE = 'https://datafeed.dukascopy.com/datafeed'
_HOUR_MS = 3_600_000
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
_DOWNLOAD_ATTEMPTS = 5
_DOWNLOAD_RETRY_DELAYS = (1.0, 3.0, 8.0, 20.0)
_LIVE_MULTI_FILE = 'live_ticks.tsv'
_CONTROL_SYMBOL = 'SYSTEM'


class DukascopyStreamGapError(ConnectionError):
    """Le flux live a perdu le pont ou un numéro de séquence."""


def _ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Les dates doivent être conscientes du fuseau horaire')
    return int(value.timestamp() * 1000)


def _stamp(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, timezone.utc)


class DukascopyTick(ITick):
    """Cotations Dukascopy historiques et live via stratégie JForex.

    source_id synthetic:... identifie sans ambiguïté les pseudo transactions.
    Le contrat documentaire de ITick exige des vrais trades : cette source
    ne satisfait donc pas cette garantie malgré la signature compatible.
    """

    def __init__(self, *, bridge_dir: str | Path, cache_dir: str | Path | None = None,
                 point_values: dict[str, int] | None = None, timeout: float = 15.0,
                 poll_interval: float = 0.1, max_quote_age: float = 10.0,
                 max_bridge_age: float = 20.0, history_request_interval: float = 0.0):
        if min(timeout, poll_interval, max_quote_age, max_bridge_age) <= 0 or history_request_interval < 0:
            raise ValueError('Tous les délais doivent être positifs et history_request_interval >= 0')
        self.bridge_dir = Path(bridge_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.point_values = point_values or {}
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_quote_age = max_quote_age
        self.max_bridge_age = max_bridge_age
        self.history_request_interval = history_request_interval
        self._quotes: dict[str, QuoteTick] = {}
        self._next_history_request_at = 0.0

    @staticmethod
    def _symbol(symbol: str) -> str:
        result = symbol.replace('/', '').upper()
        if not re.fullmatch(r'[A-Z]{6}', result):
            raise ValueError('Symbole 6 lettres attendu, exemple : EURUSD, BTCUSD ou EUR/USD')
        return result

    def _scale(self, symbol: str) -> int:
        scale = self.point_values.get(symbol, 1_000 if symbol.endswith('JPY') else 100_000)
        if not isinstance(scale, int) or scale <= 0:
            raise ValueError('point_values doit contenir des entiers strictement positifs')
        return scale

    @staticmethod
    def _synthetic(quote: QuoteTick) -> TradeTick:
        return TradeTick(quote.symbol, quote.timestamp, (quote.bid + quote.ask) / 2,
                         source_id='synthetic:dukascopy:midquote', sequence=quote.sequence)

    def _wait_history_request_slot(self) -> None:
        if self.history_request_interval <= 0:
            return
        now = time.monotonic()
        if now < self._next_history_request_at:
            time.sleep(self._next_history_request_at - now)
        self._next_history_request_at = time.monotonic() + self.history_request_interval

    def _download_hour(self, symbol: str, hour: datetime) -> bytes | None:
        relative = f'{symbol}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5'
        cached = self.cache_dir / relative if self.cache_dir else None
        if cached is not None and cached.exists():
            payload = cached.read_bytes()
            if payload:
                return payload
            cached.unlink()
        request = Request(f'{_BASE}/{relative}', headers={'User-Agent': 'DukascopyTick/1.0'})
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            try:
                self._wait_history_request_slot()
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                if not payload:
                    raise ConnectionError(f'Téléchargement Dukascopy vide : {relative}')
                break
            except HTTPError as exc:
                if exc.code == 404:
                    return None  # Heure sans fichier, généralement marché fermé.
                if exc.code not in _TRANSIENT_HTTP_CODES or attempt + 1 >= _DOWNLOAD_ATTEMPTS:
                    raise ConnectionError(f'Dukascopy HTTP {exc.code} : {relative}') from exc
            except TimeoutError as exc:
                if attempt + 1 >= _DOWNLOAD_ATTEMPTS:
                    raise ConnectionError(f'Téléchargement Dukascopy expiré : {relative}') from exc
            except (URLError, OSError, ConnectionError) as exc:
                if attempt + 1 >= _DOWNLOAD_ATTEMPTS:
                    raise ConnectionError(f'Téléchargement Dukascopy impossible : {relative}') from exc
            time.sleep(_DOWNLOAD_RETRY_DELAYS[min(attempt, len(_DOWNLOAD_RETRY_DELAYS) - 1)])
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(cached.suffix + '.part')
            if temporary.exists():
                temporary.unlink()
            temporary.write_bytes(payload)
            temporary.replace(cached)
        return payload

    def historical_ticks(self, symbol: str, start: datetime, end: datetime) -> Iterator[Tick]:
        symbol = self._symbol(symbol)
        lower, upper = _ms(start), _ms(end)
        if lower >= upper:
            raise ValueError('start doit précéder end')
        # Les archives horaires ne sont pas un flux temps réel.
        last_full_hour = int(time.time() * 1000) // _HOUR_MS * _HOUR_MS
        if upper > last_full_hour:
            raise ValueError('L’historique ne comprend que les heures UTC terminées')
        scale = self._scale(symbol)
        cursor = lower // _HOUR_MS * _HOUR_MS
        while cursor < upper:
            hour = _stamp(cursor)
            payload = self._download_hour(symbol, hour)
            if payload:
                try:
                    raw = lzma.decompress(payload)
                except lzma.LZMAError as exc:
                    raise ValueError(f'Archive Dukascopy corrompue : {hour}') from exc
                if len(raw) % _RECORD.size:
                    raise ValueError(f'Taille des ticks Dukascopy invalide : {hour}')
                previous = -1
                for offset, ask_raw, bid_raw, ask_volume, bid_volume in _RECORD.iter_unpack(raw):
                    if offset >= _HOUR_MS or offset < previous:
                        raise ValueError(f'Horodatages Dukascopy incohérents : {hour}')
                    previous = offset
                    stamp_ms = cursor + offset
                    if lower <= stamp_ms < upper:
                        bid, ask = bid_raw / scale, ask_raw / scale
                        if bid <= 0 or ask < bid or not all(map(math.isfinite, (ask_volume, bid_volume))):
                            raise ValueError(f'Cotations Dukascopy invalides : {hour}')
                        quote = QuoteTick(symbol, _stamp(stamp_ms), bid, ask,
                                          bid_size=float(bid_volume), ask_size=float(ask_volume),
                                          source_id='dukascopy:quote')
                        yield quote
                        yield self._synthetic(quote)
            cursor += _HOUR_MS

    def _file(self, symbol: str) -> Path:
        return self.bridge_dir / f'live_{symbol}.tsv'

    def _multi_file(self) -> Path:
        return self.bridge_dir / _LIVE_MULTI_FILE

    def _wait_live_path(self, symbol: str) -> Path:
        single_path = self._file(symbol)
        multi_path = self._multi_file()
        deadline = time.monotonic() + self.timeout
        while True:
            if multi_path.exists():
                return multi_path
            if single_path.exists():
                return single_path
            if time.monotonic() > deadline:
                raise DukascopyStreamGapError(
                    f'Pont JForex absent : attendu {multi_path} ou {single_path}'
                )
            time.sleep(self.poll_interval)

    @staticmethod
    def _bridge_frame(line: str) -> tuple[str, str, str, int, int, QuoteTick | None]:
        cells = line.rstrip('\r\n').split('\t')
        if len(cells) != 9 or cells[0] not in ('START', 'HEARTBEAT', 'QUOTE', 'STOP'):
            raise DukascopyStreamGapError('Trame JForex illisible')
        kind, frame_symbol, stamp_text, session, seq_text, bid, ask, bid_size, ask_size = cells
        if not session:
            raise DukascopyStreamGapError('Session JForex absente')
        try:
            moment, seq = int(stamp_text), int(seq_text)
            quote = None
            if kind == 'QUOTE':
                normalized_symbol = DukascopyTick._symbol(frame_symbol)
                bid_v, ask_v = float(bid), float(ask)
                sizes = float(bid_size), float(ask_size)
                if bid_v <= 0 or ask_v < bid_v or not all(map(math.isfinite, (bid_v, ask_v, *sizes))):
                    raise ValueError('bid/ask ou volumes incorrects')
                quote = QuoteTick(normalized_symbol, _stamp(moment), bid_v, ask_v,
                                  bid_size=sizes[0], ask_size=sizes[1],
                                  source_id='dukascopy:jforex:quote', sequence=seq)
        except (ValueError, OverflowError) as exc:
            raise DukascopyStreamGapError('Trame JForex invalide') from exc
        if seq < 0:
            raise DukascopyStreamGapError('Séquence négative')
        return kind, frame_symbol, session, moment, seq, quote

    @staticmethod
    def _frame(line: str, symbol: str) -> tuple[str, str, int, int, QuoteTick | None]:
        kind, frame_symbol, session, moment, seq, quote = DukascopyTick._bridge_frame(line)
        if kind == 'QUOTE':
            if quote is None or quote.symbol != symbol:
                raise DukascopyStreamGapError('Trame JForex illisible ou symbole incorrect')
        elif frame_symbol not in {symbol, _CONTROL_SYMBOL}:
            raise DukascopyStreamGapError('Trame JForex illisible ou symbole incorrect')
        return kind, session, moment, seq, quote

    def _iter_live_stream(self, path: Path, requested_symbols: set[str]) -> Iterator[Tick]:
        with path.open('r', encoding='utf-8') as stream:
            stream.seek(0, 2)
            session: str | None = None
            last_seq: int | None = None
            last_seen = time.monotonic()
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    if time.monotonic() - last_seen > self.max_bridge_age:
                        raise DukascopyStreamGapError('Plus de trames JForex; vérifier la stratégie')
                    time.sleep(self.poll_interval)
                    continue
                if not line.endswith('\n'):
                    stream.seek(position)
                    time.sleep(self.poll_interval)
                    continue
                kind, _frame_symbol, current_session, moment, seq, quote = self._bridge_frame(line)
                last_seen = time.monotonic()
                if (time.time() * 1000 - moment) > self.max_bridge_age * 1000:
                    raise DukascopyStreamGapError('Pont JForex en retard')
                if session is None:
                    session, last_seq = current_session, seq if kind != 'QUOTE' else seq - 1
                elif current_session != session:
                    raise DukascopyStreamGapError('Nouvelle session JForex; rattrapage nécessaire')
                if kind == 'STOP':
                    raise DukascopyStreamGapError('Stratégie JForex arrêtée')
                if kind == 'START':
                    raise DukascopyStreamGapError('Pont JForex redémarré; rattrapage nécessaire')
                if kind == 'QUOTE':
                    if seq != last_seq + 1:
                        raise DukascopyStreamGapError('Séquence JForex manquante; rattrapage nécessaire')
                    last_seq = seq
                    assert quote is not None
                    if quote.symbol not in requested_symbols:
                        continue
                    self._quotes[quote.symbol] = quote
                    yield quote
                    yield self._synthetic(quote)
                elif seq != last_seq:
                    raise DukascopyStreamGapError('Séquence JForex incohérente')

    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        symbol = self._symbol(symbol)
        path = self._wait_live_path(symbol)
        yield from self._iter_live_stream(path, {symbol})

    def live_ticks_multi(self, symbols: Iterable[str]) -> Iterator[Tick]:
        requested_symbols = {self._symbol(symbol) for symbol in symbols}
        if not requested_symbols:
            raise ValueError('Aucun symbole fourni')
        path = self._multi_file()
        deadline = time.monotonic() + self.timeout
        while not path.exists():
            if time.monotonic() > deadline:
                raise DukascopyStreamGapError(f'Pont JForex absent : {path}')
            time.sleep(self.poll_interval)
        yield from self._iter_live_stream(path, requested_symbols)

    def latest_quote(self, symbol: str) -> QuoteTick:
        symbol = self._symbol(symbol)
        # Consultation utilisable même si live_ticks n'est pas encore consommé.
        path = self._multi_file() if self._multi_file().exists() else self._file(symbol)
        if not path.exists():
            raise LookupError(f'Pont JForex absent : {path}')
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 131_072))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        if size > 131_072 and lines:
            lines = lines[1:]
        quote = None
        for line in reversed(lines):
            try:
                kind, _, _, _, _, candidate = self._bridge_frame(line)
            except DukascopyStreamGapError:
                continue
            if kind in ('STOP', 'START'):
                break
            if candidate is not None and candidate.symbol == symbol:
                quote = candidate
                break
        if quote is None:
            raise LookupError(f'Aucune cotation JForex récente pour {symbol}')
        age = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
        if age < -2 or age > self.max_quote_age:
            raise TimeoutError(f'Cotations Dukascopy périmées : {age:.1f}s')
        self._quotes[symbol] = quote
        return quote
