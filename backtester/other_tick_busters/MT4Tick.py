"""Lecture des ticks MT4 capturés par MT4TickBridge.mq4.

MT4 ne propose pas l'API Python MT5 ni un historique de quotes tick garanti.
Les TradeTick renvoyés sont des prix médians SYNTHÉTIQUES, non des transactions.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ITick import ITick, QuoteTick, Tick, TradeTick


class MT4StreamGapError(ConnectionError):
    """Perte du pont MT4 ; les données manquantes ne peuvent être supposées complètes."""


class MT4Tick(ITick):
    def __init__(self, data_dir: str | Path, *, max_quote_age: float = 10.0,
                 heartbeat_timeout: float = 20.0, poll_interval: float = 0.1,
                 allowed_heartbeat_gap: float = 20.0):
        self._dir = Path(data_dir)
        if min(max_quote_age, heartbeat_timeout, poll_interval, allowed_heartbeat_gap) <= 0:
            raise ValueError("Délais strictement positifs requis")
        self._max_age = max_quote_age
        self._hb_timeout = heartbeat_timeout
        self._poll = poll_interval
        self._allowed_gap = allowed_heartbeat_gap
        self._last: dict[str, QuoteTick] = {}

    def _path(self, symbol: str) -> Path:
        if not symbol or any(x in symbol for x in ("/", "\\", "..")):
            raise ValueError("Nom du symbole invalide")
        return self._dir / f"mt4_ticks_{symbol}.csv"

    @staticmethod
    def _row(line: str, symbol: str) -> tuple[str, datetime, str, int, QuoteTick | None, bool]:
        parts = line.rstrip("\r\n").split(";")
        if len(parts) != 8 or parts[1] != symbol or parts[0] not in ("START", "STOP", "QUOTE", "HEARTBEAT"):
            raise ValueError("Ligne de pont MT4 invalide ou symbole différent")
        kind, _, utc_s, session, sequence, bid, ask, connected = parts
        timestamp = datetime.fromtimestamp(int(utc_s), timezone.utc)
        quote = (QuoteTick(symbol, timestamp, float(bid), float(ask), source_id="mt4:broker:quote",
                           sequence=int(sequence)) if kind == "QUOTE" else None)
        return kind, timestamp, session, int(sequence), quote, connected == "1"

    @staticmethod
    def _emit(quote: QuoteTick) -> Iterator[Tick]:
        yield quote
        yield TradeTick(quote.symbol, quote.timestamp, (quote.bid + quote.ask) / 2.0,
                        quantity=None, source_id="synthetic:mt4:midquote", sequence=quote.sequence)

    def historical_ticks(self, symbol: str, start: datetime, end: datetime) -> Iterator[Tick]:
        if any(x.tzinfo is None or x.utcoffset() is None for x in (start, end)) or start >= end:
            raise ValueError("Intervalle [start, end) et dates avec fuseau horaire requis")
        path = self._path(symbol)
        if not path.is_file():
            raise FileNotFoundError(f"Aucune capture MT4 pour {symbol} : {path}")
        # N'accepter qu'une fenêtre entièrement incluse dans une même session
        # saine avec heartbeats réguliers. La mémoire reste bornée à la fenêtre.
        sessions: dict[str, dict] = {}
        with path.open("r", encoding="ascii") as file:
            for line in file:
                if not line.endswith("\n"):  # écriture concurrente incomplète
                    break
                kind, stamp, session, seq, quote, connected = self._row(line, symbol)
                state = sessions.get(session)
                if kind == "START":
                    state = {"start": stamp, "last_hb": stamp, "last_seq": seq,
                             "valid": connected, "events": [], "end": None}
                    sessions[session] = state
                    continue
                if state is None:
                    continue
                if seq != state["last_seq"] + 1 or stamp < state["last_hb"]:
                    state["valid"] = False
                if (stamp - state["last_hb"]).total_seconds() > self._allowed_gap or not connected:
                    state["valid"] = False
                state["last_hb"] = stamp
                state["last_seq"] = seq
                if kind == "STOP":
                    state["end"] = stamp
                if quote is not None and start <= stamp < end:
                    state["events"].append(quote)
        candidates = [state for state in sessions.values()
                      if state["valid"] and state["start"] <= start and state["last_hb"] >= end]
        if len(candidates) != 1:
            raise MT4StreamGapError("Historique non couvert intégralement par une seule capture MT4 continue")
        for quote in candidates[0]["events"]:
            yield from self._emit(quote)

    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        path = self._path(symbol)
        if not path.is_file():
            raise FileNotFoundError(f"Attacher MT4TickBridge au graphique {symbol} avant l'abonnement")
        with path.open("r", encoding="ascii") as file:
            file.seek(0, 2)  # uniquement les nouveautés ; démarrer la capture avant le robot
            last_heartbeat = time.monotonic()
            session = None
            seq = None
            while True:
                line = file.readline()
                if not line or not line.endswith("\n"):
                    if line: file.seek(file.tell() - len(line))
                    if time.monotonic() - last_heartbeat > self._hb_timeout:
                        raise MT4StreamGapError("Plus de heartbeat MT4 ; capture interrompue")
                    time.sleep(self._poll)
                    continue
                kind, stamp, current_session, number, quote, connected = self._row(line, symbol)
                if kind == "START":
                    if session is not None:
                        raise MT4StreamGapError("EA MT4 redémarré ; rattrapage nécessaire")
                    session, seq = current_session, number
                elif session is None:
                    session, seq = current_session, number - 1
                if current_session != session or number != seq + 1 or not connected or kind == "STOP":
                    raise MT4StreamGapError("Trou, déconnexion ou changement de session MT4")
                seq = number
                if kind in ("HEARTBEAT", "START"):
                    last_heartbeat = time.monotonic()
                if quote is not None:
                    last_heartbeat = time.monotonic()
                    self._last[symbol] = quote
                    yield from self._emit(quote)

    def latest_quote(self, symbol: str) -> QuoteTick:
        quote = self._last.get(symbol)
        if quote is None:
            raise LookupError("Démarrer et consommer live_ticks avant de demander la quote")
        if (datetime.now(timezone.utc) - quote.timestamp).total_seconds() > self._max_age:
            raise TimeoutError("Quote MT4 périmée")
        return quote
