"""Adaptateur ITick pour Vantage via le terminal local MetaTrader 5.

Le module officiel MetaTrader5 doit être installé dans le Python qui accède au
terminal MT5 ouvert ou lançable. Pour les CFD/forex sans transactions Last,
on émet un TradeTick SYNTHÉTIQUE au prix médian de chaque changement Bid/Ask.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from ITick import ITick, QuoteTick, Tick, TradeTick


class MT5StreamGapError(ConnectionError):
    """Un intervalle live n'est pas assuré complet (terminal, API ou retard)."""


def _utc_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, timezone.utc)


def _millis(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("La date doit avoir un fuseau horaire explicite")
    return int(value.timestamp() * 1000)


class MT5VantageTick(ITick):
    """Un adaptateur par terminal MT5 ; ne pas partager le module entre comptes."""

    def __init__(self, *, terminal_path: str | None = None, login: int | None = None,
                 password: str | None = None, server: str | None = None,
                 poll_interval: float = 0.2, max_quote_age: float = 10.0,
                 history_chunk: timedelta = timedelta(days=1),
                 max_live_backlog: timedelta = timedelta(minutes=2),
                 mt5_module: Any = None):
        if poll_interval <= 0 or max_quote_age <= 0 or history_chunk <= timedelta(0):
            raise ValueError("Intervalles strictement positifs requis")
        if max_live_backlog <= timedelta(0):
            raise ValueError("max_live_backlog doit être positif")
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module
            except ImportError as exc:
                raise RuntimeError("Installer MetaTrader5 dans le Python qui pilote le terminal") from exc
        self._mt5 = mt5_module
        self._lock = threading.RLock()
        self._poll, self._max_age = poll_interval, max_quote_age
        self._chunk, self._backlog = history_chunk, max_live_backlog
        options: dict[str, Any] = {}
        if login is not None:
            options["login"] = int(login)
        if password is not None:
            options["password"] = password
        if server is not None:
            options["server"] = server
        # Le chemin de l'EXE est le premier argument POSITIONNEL de initialize.
        ok = (self._mt5.initialize(terminal_path, **options) if terminal_path is not None
              else self._mt5.initialize(**options))
        if not ok:
            raise ConnectionError(f"MT5 initialize a échoué : {self._mt5.last_error()}")
        self._closed = False
        try:
            self._connected()
            if login is not None and self._mt5.account_info().login != int(login):
                raise ConnectionError("Le terminal est sur un autre compte MT5")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._mt5.shutdown()
                self._closed = True

    def __enter__(self) -> "MT5VantageTick":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _connected(self) -> None:
        if self._closed:
            raise ConnectionError("Connexion MT5 fermée")
        info = self._mt5.terminal_info()
        if info is None or not info.connected:
            raise MT5StreamGapError(f"Terminal MT5 hors connexion : {self._mt5.last_error()}")

    def _select(self, symbol: str) -> None:
        with self._lock:
            self._connected()
            if not self._mt5.symbol_select(symbol, True):
                raise ValueError(f"Symbole introuvable/non sélectionnable chez Vantage : {symbol}; {self._mt5.last_error()}")

    def _signature(self, row: Any) -> tuple:
        # Pour dédupliquer une fenêtre de polling qui recouvre le dernier ms.
        fields = ("time_msc", "flags", "bid", "ask", "last", "volume", "volume_real")
        names = set(row.dtype.names)
        return tuple(row[name].item() if hasattr(row[name], "item") else row[name]
                     for name in fields if name in names)

    def _events(self, symbol: str, row: Any) -> Iterator[Tick]:
        mt5 = self._mt5
        flags = int(row["flags"])
        stamp = _utc_ms(int(row["time_msc"]))
        bid, ask = float(row["bid"]), float(row["ask"])
        quote_changed = bool(flags & (mt5.TICK_FLAG_BID | mt5.TICK_FLAG_ASK))
        trade_changed = bool(flags & mt5.TICK_FLAG_LAST) and float(row["last"]) > 0
        if quote_changed and bid > 0 and ask > 0:
            yield QuoteTick(symbol, stamp, bid, ask, source_id="mt5:vantage:quote")
        if trade_changed:
            size = float(row["volume_real"]) if "volume_real" in row.dtype.names else float(row["volume"])
            yield TradeTick(symbol, stamp, float(row["last"]),
                            quantity=size if size > 0 else None,
                            source_id="mt5:vantage:last")
        elif quote_changed and bid > 0 and ask > 0:
            yield TradeTick(symbol, stamp, (bid + ask) / 2,
                            quantity=None, source_id="synthetic:mt5:vantage:midquote")

    def _range(self, symbol: str, start_ms: int, end_ms: int):
        with self._lock:
            self._connected()
            rows = self._mt5.copy_ticks_range(symbol, _utc_ms(start_ms),
                _utc_ms(end_ms), self._mt5.COPY_TICKS_ALL)
            if rows is None:
                raise MT5StreamGapError(f"Impossible d'obtenir les ticks MT5 : {self._mt5.last_error()}")
            return rows

    def historical_ticks(self, symbol: str, start: datetime, end: datetime) -> Iterator[Tick]:
        start_ms, end_ms = _millis(start), _millis(end)
        if start_ms >= end_ms:
            raise ValueError("start doit précéder end")
        self._select(symbol)
        chunk_ms = max(1, int(self._chunk.total_seconds() * 1000))
        cursor = start_ms
        while cursor < end_ms:
            upper = min(cursor + chunk_ms - 1, end_ms - 1)
            for row in self._range(symbol, cursor, upper):
                if cursor <= int(row["time_msc"]) < end_ms:
                    yield from self._events(symbol, row)
            cursor = upper + 1

    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        """Polling de l'historique récent, sans perdre les ticks d'un même ms.

        Une déconnexion lève une erreur ; le consommateur doit reconstruire sa
        période manquante par historical_ticks avant toute reprise.
        """
        self._select(symbol)
        started_ms = _millis(datetime.now(timezone.utc))
        cursor_ms = started_ms
        seen_at_cursor: Counter[tuple] = Counter()
        while True:
            now_ms = _millis(datetime.now(timezone.utc))
            if now_ms - cursor_ms > self._backlog.total_seconds() * 1000:
                raise MT5StreamGapError("Retard live supérieur au seuil ; rattrapage requis")
            rows = self._range(symbol, cursor_ms, now_ms)
            counts: Counter[tuple] = Counter()
            for row in rows:
                stamp = int(row["time_msc"])
                if stamp < cursor_ms:
                    continue
                signature = self._signature(row)
                if stamp == cursor_ms:
                    counts[signature] += 1
                    if counts[signature] <= seen_at_cursor[signature]:
                        continue
                else:
                    cursor_ms = stamp
                    seen_at_cursor = Counter()
                    counts = Counter({signature: 1})
                seen_at_cursor[signature] = counts[signature]
                if stamp >= started_ms:
                    yield from self._events(symbol, row)
            time.sleep(self._poll)

    def latest_quote(self, symbol: str) -> QuoteTick:
        self._select(symbol)
        with self._lock:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is None:
                raise LookupError(f"Pas de quote MT5 pour {symbol} : {self._mt5.last_error()}")
            quote = QuoteTick(symbol, _utc_ms(int(tick.time_msc)), float(tick.bid),
                              float(tick.ask), source_id="mt5:vantage:quote")
        if quote.bid <= 0 or quote.ask <= 0 or quote.bid > quote.ask:
            raise ValueError("Bid/Ask MT5 invalide")
        if (datetime.now(timezone.utc) - quote.timestamp).total_seconds() > self._max_age:
            raise TimeoutError("Dernier bid/ask Vantage périmé")
        return quote
