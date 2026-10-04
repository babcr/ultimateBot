"""Consommateur live Dukascopy avec dispatch automatique par symbole.

Le module s'appuie sur DukascopyTick.live_ticks_multi pour conserver l'ordre
global d'arrivee des ticks, puis route chaque evenement vers les handlers du
symbole concerne.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from DukascopyTick import DukascopyStreamGapError, DukascopyTick
from ITick import QuoteTick, Tick, TradeTick

if TYPE_CHECKING:
    from Instrument import Instrument

QuoteHandler = Callable[[QuoteTick], None]
TradeHandler = Callable[[TradeTick], None]
TickHandler = Callable[[Tick], None]


def _normalize_symbol(symbol: str) -> str:
    result = symbol.replace('/', '').upper()
    if not re.fullmatch(r'[A-Z]{6}', result):
        raise ValueError(f'Symbole 6 lettres attendu: {symbol!r}')
    return result


class DukascopyLiveConsumer:
    """Route les ticks live vers des handlers et/ou des Instrument.

    Utilisation typique:

        consumer = DukascopyLiveConsumer(
            bridge_dir="D:\\jforex_live",
            symbols=["EURUSD", "GBPUSD", "BTCUSD"],
        )
        consumer.register_quote_handler("EURUSD", handle_eurusd_quote)
        consumer.run_forever()
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        bridge_dir: str | Path | None = None,
        source: DukascopyTick | None = None,
        cache_dir: str | Path | None = None,
        point_values: dict[str, int] | None = None,
        timeout: float = 15.0,
        poll_interval: float = 0.1,
        max_quote_age: float = 10.0,
        max_bridge_age: float = 20.0,
        history_request_interval: float = 0.0,
    ):
        normalized_symbols = []
        seen_symbols: set[str] = set()
        for raw_symbol in symbols:
            symbol = _normalize_symbol(raw_symbol)
            if symbol not in seen_symbols:
                normalized_symbols.append(symbol)
                seen_symbols.add(symbol)
        if not normalized_symbols:
            raise ValueError('Aucun symbole fourni')

        if source is None:
            if bridge_dir is None:
                raise ValueError('bridge_dir est requis quand source n\'est pas fourni')
            source = DukascopyTick(
                bridge_dir=bridge_dir,
                cache_dir=cache_dir,
                point_values=point_values,
                timeout=timeout,
                poll_interval=poll_interval,
                max_quote_age=max_quote_age,
                max_bridge_age=max_bridge_age,
                history_request_interval=history_request_interval,
            )

        self.source = source
        self.symbols = tuple(normalized_symbols)
        self._symbol_set = set(self.symbols)
        self._quote_handlers: dict[str, list[QuoteHandler]] = defaultdict(list)
        self._trade_handlers: dict[str, list[TradeHandler]] = defaultdict(list)
        self._symbol_handlers: dict[str, list[TickHandler]] = defaultdict(list)
        self._global_handlers: list[TickHandler] = []

    def _checked_symbol(self, symbol: str) -> str:
        normalized = _normalize_symbol(symbol)
        if normalized not in self._symbol_set:
            raise ValueError(f'Symbole non abonne dans ce consumer: {normalized}')
        return normalized

    def register_quote_handler(self, symbol: str, handler: QuoteHandler) -> None:
        self._quote_handlers[self._checked_symbol(symbol)].append(handler)

    def register_trade_handler(self, symbol: str, handler: TradeHandler) -> None:
        self._trade_handlers[self._checked_symbol(symbol)].append(handler)

    def register_symbol_handler(self, symbol: str, handler: TickHandler) -> None:
        self._symbol_handlers[self._checked_symbol(symbol)].append(handler)

    def register_global_handler(self, handler: TickHandler) -> None:
        self._global_handlers.append(handler)

    def attach_instrument(self, instrument: "Instrument") -> None:
        self.register_quote_handler(instrument.symbol, instrument.add_quote)

    def attach_instruments(self, instruments: Iterable["Instrument"]) -> None:
        for instrument in instruments:
            self.attach_instrument(instrument)

    def dispatch(self, tick: Tick) -> Tick:
        symbol = self._checked_symbol(tick.symbol)

        if isinstance(tick, QuoteTick):
            for handler in self._quote_handlers.get(symbol, ()):  # quotes en premier pour l'etat par paire
                handler(tick)
        elif isinstance(tick, TradeTick):
            for handler in self._trade_handlers.get(symbol, ()):  # trades ensuite si le moteur les exploite
                handler(tick)

        for handler in self._symbol_handlers.get(symbol, ()):  # hooks generiques par symbole
            handler(tick)
        for handler in self._global_handlers:  # hooks transverses: log, metriques, audit
            handler(tick)

        return tick

    def iter_dispatched(self) -> Iterator[Tick]:
        for tick in self.source.live_ticks_multi(self.symbols):
            yield self.dispatch(tick)

    def run_forever(self) -> None:
        for _tick in self.iter_dispatched():
            pass


def _parse_symbols(csv: str) -> tuple[str, ...]:
    return tuple(_normalize_symbol(part.strip()) for part in csv.split(',') if part.strip())


def _print_tick(tick: Tick) -> None:
    if isinstance(tick, QuoteTick):
        print(
            f'QUOTE\t{tick.symbol}\t{tick.timestamp.isoformat()}\t'
            f'{tick.bid}\t{tick.ask}\t{tick.bid_size}\t{tick.ask_size}\tseq={tick.sequence}'
        )
        return
    print(
        f'TRADE\t{tick.symbol}\t{tick.timestamp.isoformat()}\t'
        f'{tick.price}\t{tick.quantity}\tseq={tick.sequence}\t{tick.source_id}'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Consomme le flux live Dukascopy et le route par symbole.')
    parser.add_argument('--bridge-dir', required=True, help='Dossier contenant live_ticks.tsv')
    parser.add_argument('--symbols', required=True, help='CSV des symboles, ex: EURUSD,GBPUSD,BTCUSD')
    parser.add_argument('--quotes-only', action='store_true', help='N\'affiche que les QuoteTick')
    args = parser.parse_args(argv)

    consumer = DukascopyLiveConsumer(bridge_dir=args.bridge_dir, symbols=_parse_symbols(args.symbols))

    if args.quotes_only:
        def maybe_print_quote(tick: Tick) -> None:
            if isinstance(tick, QuoteTick):
                _print_tick(tick)

        consumer.register_global_handler(maybe_print_quote)
    else:
        consumer.register_global_handler(_print_tick)

    try:
        consumer.run_forever()
    except KeyboardInterrupt:
        return 130
    except DukascopyStreamGapError as error:
        print(f'Flux Dukascopy interrompu: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())