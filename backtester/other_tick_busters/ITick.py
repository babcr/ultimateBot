"""Contrat d'une source de transactions et de cotations historiques et live."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class TradeTick:
    symbol: str
    timestamp: datetime  # UTC, timezone-aware
    price: float
    quantity: float | None = None
    source_id: str | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class QuoteTick:
    symbol: str
    timestamp: datetime  # UTC, timezone-aware
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    source_id: str | None = None
    sequence: int | None = None


Tick: TypeAlias = TradeTick | QuoteTick


class ITick(ABC):
    """Une implémentation doit fournir les quatre flux pour chaque symbole annoncé.

    Aucun TradeTick ne doit être déduit d'un changement de bid/ask ou d'une
    bougie. Le moteur de backtest rejoue historical_ticks sans connaître la
    source ; la production consomme live_ticks puis consulte latest_quote.
    """

    @abstractmethod
    def historical_ticks(
        self, symbol: str, start: datetime, end: datetime
    ) -> Iterator[Tick]:
        """Trades ET quotes réels sur [start, end), en ordre de la source.

        Les bornes sont des datetime UTC conscients du fuseau horaire.
        Une source ne doit pas substituer des agrégats ou un autre flux.
        """
        ...

    @abstractmethod
    def live_ticks(self, symbol: str) -> Iterator[Tick]:
        """Trades ET quotes live en ordre de réception ; se termine à l'arrêt.

        L'implémentation doit gérer reconnexion et signaler les lacunes :
        un flux interrompu ne peut pas être traité comme continu.
        """
        ...

    @abstractmethod
    def latest_quote(self, symbol: str) -> QuoteTick:
        """Dernier bid/ask live connu, ou erreur si absent ou périmé.

        Vérifier timestamp et âge maximal avant toute préparation d'ordre.
        L'implémentation doit partager l'état avec live_ticks ou maintenir
        son propre abonnement aux quotes.
        """
        ...
