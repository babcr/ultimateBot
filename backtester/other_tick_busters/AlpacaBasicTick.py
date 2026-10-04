"""ITick pour les actions US du plan Alpaca Basic (flux IEX uniquement).

Dépendance pour le direct : ``pip install websocket-client``.
L'historique provient de l'API REST, le direct du WebSocket v2/iex.
Ne pas confondre les transactions/cotations IEX avec celles de toutes les bourses US.
"""
from __future__ import annotations

import heapq
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from ITick import ITick, QuoteTick, Tick, TradeTick

REST_URL = "https://data.alpaca.markets/v2/stocks"
STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"


class AlpacaStreamGapError(ConnectionError):
    """Le stream est interrompu : son ordre et son exhaustivité ne sont plus garantis."""


def _ns(raw: str) -> int:
    """RFC3339 -> nanosecondes UTC, sans perdre la précision lors du tri."""
    base, sep, frac_zone = raw.partition(".")
    if sep:
        digits = "".join(c for c in frac_zone if c.isdigit())
        # Le champ timestamp Alpaca est en UTC et utilise le suffixe Z.
        if not frac_zone.endswith("Z") or len(digits) > 9:
            raise ValueError(f"Horodatage inattendu : {raw}")
    elif not base.endswith("Z"):
        raise ValueError(f"Horodatage inattendu : {raw}")
    instant = datetime.fromisoformat(base.removesuffix("Z") + "+00:00")
    return int(instant.timestamp()) * 1_000_000_000 + int((digits if sep else "").ljust(9, "0"))


def _datetime(ns: int) -> datetime:
    seconds, nano = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=nano // 1000)


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Utiliser une date avec fuseau horaire explicite")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class AlpacaBasicTick(ITick):
    def __init__(self, key_id: str, secret_key: str, *, max_quote_age: float = 10.0,
                 timeout: float = 20.0):
        if not key_id or not secret_key:
            raise ValueError("Clés API Alpaca manquantes")
        self._key_id, self._secret_key = key_id, secret_key
        self._timeout, self._max_quote_age = timeout, max_quote_age
        self._last_quotes: dict[str, QuoteTick] = {}
        self._lock = threading.Lock()

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = REST_URL + path + "?" + urllib.parse.urlencode({**params, "feed": "iex"})
        request = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Alpaca HTTP {exc.code} ({path}); vérifier accès IEX, dates et quota") from exc

    @staticmethod
    def _normalize(raw: dict[str, Any], symbol: str, kind: str) -> Tick:
        stamp = _datetime(_ns(raw["t"]))
        if kind == "trades":
            return TradeTick(symbol, stamp, float(raw["p"]), float(raw["s"]),
                             source_id=f"alpaca:iex:trade:{raw.get('i', '')}")
        return QuoteTick(symbol, stamp, float(raw["bp"]), float(raw["ap"]),
                         bid_size=float(raw["bs"]) if "bs" in raw else None,
                         ask_size=float(raw["as"]) if "as" in raw else None,
                         source_id="alpaca:iex:quote")

    def _pages(self, symbol: str, kind: str, start: datetime, end: datetime
               ) -> Iterator[tuple[int, Tick]]:
        path = f"/{urllib.parse.quote(symbol, safe='')}/{kind}"
        params = {"start": _rfc3339(start), "end": _rfc3339(end),
                  "sort": "asc", "limit": "10000"}
        token: str | None = None
        seen_tokens: set[str] = set()
        end_ns = int(end.timestamp() * 1_000_000_000)
        start_ns = int(start.timestamp() * 1_000_000_000)
        while True:
            response = self._get(path, {**params, **({"page_token": token} if token else {})})
            for raw in response.get(kind, []):
                stamp = _ns(raw["t"])
                if start_ns <= stamp < end_ns:  # Alpaca traite end comme inclusif.
                    yield stamp, self._normalize(raw, symbol, kind)
            token = response.get("next_page_token")
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Pagination Alpaca répétée")
            seen_tokens.add(token)

    def historical_ticks(self, symbol: str, start: datetime, end: datetime) -> Iterator[Tick]:
        if not symbol:
            raise ValueError("Symbole manquant")
        _rfc3339(start)
        _rfc3339(end)
        if start >= end:
            raise ValueError("start doit précéder end")
        # Plan Basic : ne pas masquer l'inaccessibilité de la fenêtre récente.
        if end > datetime.now(timezone.utc) - timedelta(minutes=15):
            raise ValueError("Alpaca Basic : demander un historique terminant il y a au moins 15 min")
        trades = self._pages(symbol, "trades", start, end)
        quotes = self._pages(symbol, "quotes", start, end)
        # Itérateurs paginés : pas de chargement de tout l'historique en mémoire.
        heap: list[tuple[int, int, Tick, Iterator[tuple[int, Tick]]]] = []
        for rank, source in enumerate((quotes, trades)):
            first = next(source, None)
            if first is not None:
                heapq.heappush(heap, (first[0], rank, first[1], source))
        while heap:
            _, rank, event, source = heapq.heappop(heap)
            yield event
            following = next(source, None)
            if following is not None:
                heapq.heappush(heap, (following[0], rank, following[1], source))

    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        if not symbol:
            raise ValueError("Symbole manquant")
        try:
            import websocket  # websocket-client, pas le paquet 'websocket'
        except ImportError as exc:
            raise RuntimeError("Installer websocket-client pour live_ticks") from exc
        # Une connexion live par appel ; Basic peut limiter à une seule connexion.
        try:
            ws = websocket.create_connection(STREAM_URL, timeout=self._timeout,
                header=[f"APCA-API-KEY-ID: {self._key_id}",
                        f"APCA-API-SECRET-KEY: {self._secret_key}"])
        except Exception as exc:
            raise AlpacaStreamGapError("Connexion IEX impossible") from exc
        try:
            ws.send(json.dumps({"action": "subscribe", "trades": [symbol], "quotes": [symbol]}))
            confirmed = False
            while True:
                payload = ws.recv()
                if not payload:
                    raise AlpacaStreamGapError("Connexion IEX fermée ; période non reçue à rattraper")
                for raw in json.loads(payload):
                    kind = raw.get("T")
                    if kind == "error":
                        raise RuntimeError(f"Alpaca stream : {raw.get('code')} {raw.get('msg')}")
                    if kind == "subscription":
                        if symbol not in raw.get("trades", ()) or symbol not in raw.get("quotes", ()):
                            raise RuntimeError("Alpaca n'a pas confirmé trades et quotes pour ce symbole")
                        confirmed = True
                    if kind not in ("t", "q") or raw.get("S") != symbol:
                        continue
                    if not confirmed:
                        raise RuntimeError("Données reçues avant confirmation d'abonnement")
                    event = self._normalize(raw, symbol, "trades" if kind == "t" else "quotes")
                    if isinstance(event, QuoteTick):
                        with self._lock:
                            self._last_quotes[symbol] = event
                    yield event
        except (websocket.WebSocketException, OSError, TimeoutError) as exc:
            raise AlpacaStreamGapError("Flux IEX interrompu ; utiliser un nouvel abonnement et rattraper la lacune par REST") from exc
        finally:
            ws.close()

    def latest_quote(self, symbol: str) -> QuoteTick:
        # REST fonctionne même si live_ticks n'a pas encore été appelé.
        with self._lock:
            value = self._last_quotes.get(symbol)
        if value is None:
            raw = self._get(f"/{urllib.parse.quote(symbol, safe='')}/quotes/latest", {})
            record = raw.get("quote")
            if record is None:
                raise LookupError(f"Aucune quote IEX pour {symbol}")
            value = self._normalize(record, symbol, "quotes")
        if datetime.now(timezone.utc) - value.timestamp > timedelta(seconds=self._max_quote_age):
            raise TimeoutError("Dernière quote IEX périmée ; pas de préparation d'ordre")
        return value
