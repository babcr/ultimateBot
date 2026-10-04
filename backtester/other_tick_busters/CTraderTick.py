"""Pepperstone cTrader Open API: cotations historiques et live.

Dépendance: pip install ctrader-open-api
Configuration: client_id, client_secret, access_token OAuth, account_id cTID,
et environment ('demo' ou 'live'). Aucune clé n'est conservée sur disque.
Une instance possède son réacteur Twisted; utiliser une seule instance par processus.
"""
from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timedelta, timezone
from typing import Iterator

from ITick import ITick, QuoteTick, TradeTick, Tick

log = logging.getLogger(__name__)
MS_WEEK = 604_800_000
SCALE = 100_000.0


def _ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("La date doit avoir un fuseau UTC explicite")
    return int(value.timestamp() * 1000)


def _date(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, timezone.utc)


def _decode_ticks(response) -> list[tuple[int, float]]:
    """cTrader renvoie le plus récent d'abord, puis des deltas de temps."""
    result = []
    time_ms = None
    for row in response.tickData:
        time_ms = int(row.timestamp) if time_ms is None else time_ms - int(row.timestamp)
        result.append((time_ms, row.tick / SCALE))
    result.reverse()
    return result


class CTraderTick(ITick):
    """Émet des TradeTick synthétiques avec source_id explicite.

    Une variation bid/ask ne constitue jamais une transaction de marché.
    L'usage de ces événements est réservé aux signaux de prix.
    """
    def __init__(self, *, client_id: str, client_secret: str, access_token: str,
                 account_id: int, environment: str = "demo", max_quote_age: float = 10.0,
                 timeout: float = 30.0):
        try:
            from ctrader_open_api import Client, Protobuf, TcpProtocol
            from ctrader_open_api.endpoints import EndPoints
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAApplicationAuthReq, ProtoOAApplicationAuthRes,
                ProtoOAAccountAuthReq, ProtoOAAccountAuthRes,
                ProtoOASymbolsListReq, ProtoOASymbolsListRes,
                ProtoOAGetTickDataReq, ProtoOAGetTickDataRes,
                ProtoOASubscribeSpotsReq, ProtoOASubscribeSpotsRes,
                ProtoOAUnsubscribeSpotsReq, ProtoOASpotEvent, ProtoOAErrorRes,
            )
            from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAQuoteType
            from twisted.internet import reactor
        except ImportError as exc:
            raise RuntimeError("Installer ctrader-open-api (et ses dépendances)") from exc
        if environment not in ("demo", "live"):
            raise ValueError("environment doit valoir demo ou live")
        self._reactor = reactor
        self._protobuf = Protobuf
        self._types = {cls.__name__: cls for cls in (
            ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq, ProtoOASymbolsListReq,
            ProtoOAGetTickDataReq, ProtoOASubscribeSpotsReq, ProtoOAUnsubscribeSpotsReq)}
        self._responses = {cls().payloadType: cls.__name__ for cls in (
            ProtoOAApplicationAuthRes, ProtoOAAccountAuthRes, ProtoOASymbolsListRes,
            ProtoOAGetTickDataRes, ProtoOASubscribeSpotsRes, ProtoOASpotEvent, ProtoOAErrorRes)}
        self._bid_type, self._ask_type = ProtoOAQuoteType.BID, ProtoOAQuoteType.ASK
        host = EndPoints.PROTOBUF_LIVE_HOST if environment == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client_id, self._client_secret = client_id, client_secret
        self._access_token, self._account_id = access_token, int(account_id)
        self._timeout, self._max_age = timeout, max_quote_age
        self._waiting: dict[str, queue.Queue] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self._symbols: dict[str, int] = {}
        self._names: dict[int, str] = {}
        self._quotes: dict[str, QuoteTick] = {}
        self._sides: dict[int, dict[str, tuple[int, float]]] = {}
        self._subscribers: dict[int, list[queue.Queue]] = {}
        self._broken: Exception | None = None
        self._ready = threading.Event()
        self._client.setConnectedCallback(self._connected)
        self._client.setDisconnectedCallback(self._disconnected)
        self._client.setMessageReceivedCallback(self._received)
        self._thread = threading.Thread(target=self._reactor.run, kwargs={"installSignalHandlers": False}, daemon=True)
        self._thread.start()
        self._reactor.callFromThread(self._client.startService)
        if not self._ready.wait(timeout):
            raise TimeoutError("Authentification cTrader expirée")
        if self._broken:
            raise self._broken

    def _connected(self, _client):
        request = self._types["ProtoOAApplicationAuthReq"]()
        request.clientId, request.clientSecret = self._client_id, self._client_secret
        self._client.send(request).addErrback(self._send_error)

    def _send_error(self, failure):
        self._disconnected(self._client, failure)

    def _disconnected(self, _client, reason):
        error = ConnectionError(f"cTrader déconnecté : {reason}")
        with self._lock:
            self._broken = error
            queues = list(self._waiting.values())
            queues += [q for group in self._subscribers.values() for q in group]
        for pending in queues:
            pending.put(error)
        self._ready.set()

    def _received(self, _client, envelope):
        name = self._responses.get(envelope.payloadType)
        if name is None:
            return
        msg = self._protobuf.extract(envelope)
        if name == "ProtoOAApplicationAuthRes":
            req = self._types["ProtoOAAccountAuthReq"]()
            req.ctidTraderAccountId = self._account_id
            req.accessToken = self._access_token
            self._client.send(req).addErrback(self._send_error)
        elif name == "ProtoOAAccountAuthRes":
            req = self._types["ProtoOASymbolsListReq"]()
            req.ctidTraderAccountId = self._account_id
            self._client.send(req).addErrback(self._send_error)
        elif name == "ProtoOASymbolsListRes":
            with self._lock:
                self._symbols = {s.symbolName: s.symbolId for s in msg.symbol}
                self._names = {s.symbolId: s.symbolName for s in msg.symbol}
            self._ready.set()
        elif name == "ProtoOASpotEvent":
            self._spot(msg)
        elif name == "ProtoOAErrorRes":
            ident = getattr(envelope, "clientMsgId", "")
            error = RuntimeError(f"cTrader API : {msg}")
            with self._lock:
                pending = self._waiting.get(ident)
            if pending:
                pending.put(error)
            else:
                self._disconnected(_client, error)
        else:
            ident = getattr(envelope, "clientMsgId", "")
            with self._lock:
                pending = self._waiting.get(ident)
            if pending:
                pending.put(msg)

    def _spot(self, msg):
        if msg.ctidTraderAccountId != self._account_id:
            return
        sid = msg.symbolId
        stamp = int(msg.spotTimestamp) if msg.HasField("spotTimestamp") else _ms(datetime.now(timezone.utc))
        with self._lock:
            sides = self._sides.setdefault(sid, {})
            if msg.HasField("bid"):
                sides["bid"] = stamp, msg.bid / SCALE
            if msg.HasField("ask"):
                sides["ask"] = stamp, msg.ask / SCALE
            if "bid" not in sides or "ask" not in sides:
                return
            symbol = self._names.get(sid)
            if symbol is None:
                return
            bt, bid = sides["bid"]
            at, ask = sides["ask"]
            quote = QuoteTick(symbol, _date(stamp), bid, ask, source_id="ctrader:quote")
            self._quotes[symbol] = quote
            for subscriber in self._subscribers.get(sid, ()):
                subscriber.put(quote)

    def _request(self, request):
        if self._broken:
            raise self._broken
        answer = queue.Queue(maxsize=1)
        with self._lock:
            self._counter += 1
            key = f"ticks-{self._counter}"
            self._waiting[key] = answer
        self._reactor.callFromThread(self._client.send, request, key)
        try:
            result = answer.get(timeout=self._timeout)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty as exc:
            raise TimeoutError(f"cTrader ne répond pas : {type(request).__name__}") from exc
        finally:
            with self._lock:
                self._waiting.pop(key, None)

    def _symbol_id(self, symbol: str) -> int:
        try:
            return self._symbols[symbol]
        except KeyError as exc:
            raise ValueError(f"Symbole absent du compte Pepperstone : {symbol}") from exc

    def _side_history(self, sid: int, kind: int, start_ms: int, end_ms: int):
        """Paginer vers le passé. Si une page coupe un même ms, échouer sans perte."""
        cursor = end_ms - 1
        values: list[tuple[int, float]] = []
        while cursor >= start_ms:
            req = self._types["ProtoOAGetTickDataReq"]()
            req.ctidTraderAccountId, req.symbolId, req.type = self._account_id, sid, kind
            req.fromTimestamp = max(start_ms, cursor - MS_WEEK + 1)
            req.toTimestamp = cursor
            res = self._request(req)
            page = _decode_ticks(res)
            if res.hasMore and not page:
                raise RuntimeError("Pagination cTrader vide avec hasMore=True")
            values.extend((t, v) for t, v in page if start_ms <= t <= cursor)
            if res.hasMore:
                oldest = page[0][0]
                if sum(t == oldest for t, _ in page) > 1:
                    raise RuntimeError("Page tronquée sur un timestamp partagé : historique incomplet")
                cursor = oldest - 1
            else:
                cursor = req.fromTimestamp - 1
        values.sort(key=lambda pair: pair[0])
        return values

    def historical_ticks(self, symbol: str, start: datetime, end: datetime) -> Iterator[Tick]:
        start_ms, end_ms = _ms(start), _ms(end)
        if start_ms >= end_ms:
            raise ValueError("start doit précéder end")
        sid = self._symbol_id(symbol)
        # Un état initial dans le flux peut manquer au début de la fenêtre.
        # Dans ce cas, aucune quote complète n'est émise avant les deux côtés.
        bids = self._side_history(sid, self._bid_type, start_ms, end_ms)
        asks = self._side_history(sid, self._ask_type, start_ms, end_ms)
        updates = sorted([(t, 0, v) for t, v in bids] + [(t, 1, v) for t, v in asks])
        state: dict[int, tuple[int, float]] = {}
        for time_ms, side, price in updates:
            state[side] = time_ms, price
            if len(state) == 2:
                bt, bid = state[0]
                at, ask = state[1]
                quote = QuoteTick(symbol, _date(time_ms), bid, ask, source_id="ctrader:quote")
                yield quote
                yield self._synthetic_trade(quote)

    @staticmethod
    def _synthetic_trade(quote: QuoteTick) -> TradeTick:
        return TradeTick(quote.symbol, quote.timestamp, (quote.bid + quote.ask) / 2,
                         quantity=None, source_id="synthetic:ctrader:midquote")

    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        sid = self._symbol_id(symbol)
        stream: queue.Queue = queue.Queue()
        with self._lock:
            group = self._subscribers.setdefault(sid, [])
            first = not group
            group.append(stream)
        if first:
            req = self._types["ProtoOASubscribeSpotsReq"]()
            req.ctidTraderAccountId = self._account_id
            req.symbolId.append(sid)
            req.subscribeToSpotTimestamp = True
            try:
                self._request(req)
            except Exception:
                with self._lock:
                    self._subscribers[sid].remove(stream)
                raise
        try:
            while True:
                item = stream.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                yield self._synthetic_trade(item)
        finally:
            with self._lock:
                group = self._subscribers.get(sid, [])
                if stream in group:
                    group.remove(stream)
                last = not group
            if last and not self._broken:
                req = self._types["ProtoOAUnsubscribeSpotsReq"]()
                req.ctidTraderAccountId = self._account_id
                req.symbolId.append(sid)
                self._reactor.callFromThread(self._client.send, req)

    def latest_quote(self, symbol: str) -> QuoteTick:
        with self._lock:
            value = self._quotes.get(symbol)
            broken = self._broken
        if broken:
            raise broken
        if value is None:
            raise LookupError("Aucune quote live reçue : démarrer live_ticks")
        now = datetime.now(timezone.utc)
        if any(now - side_time > timedelta(seconds=self._max_age)
               for side_time in (_date(self._sides[self._symbol_id(symbol)]["bid"][0]),
                                 _date(self._sides[self._symbol_id(symbol)]["ask"][0]))):
            raise TimeoutError("Bid ou Ask périmé")
        return value
