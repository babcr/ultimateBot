from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import ceil, floor, isfinite
from typing import Optional

from ultimateBot.backtester.config.backtest_config import (
    RiskConfig,
    SUPPORTED_CAPITAL_SOURCES,
    VolatilityConfig,
    normalize_candle_interval,
)
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.orders.order import OrderSide, OrderType
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide


DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()


@dataclass(frozen=True)
class RiskEvaluation:
    """Décision finale consommée par le builder d'ordres."""

    side: OrderSide
    order_type: OrderType
    entry_mode: SignalEntryMode
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    available_capital: float
    capital_at_risk: float
    risk_percent: float
    stop_distance: float
    take_profit_distance: float
    raw_spread: float
    spread_ratio: float
    entry_offset: float


class RiskManager:
    """Transforme un signal en plan d'ordre dimensionné et protégé."""

    _STRUCTURE_STOP_BUFFER_RATIO = 0.15
    _STRUCTURE_QUALITY_BUFFER_RATIO = 0.20
    _STRUCTURE_MIN_STOP_RATIO = 0.25

    def __init__(
        self,
        risk_per_trade: float = DEFAULT_RISK_CONFIG.risk_per_trade_percent,
        max_positions: int = DEFAULT_RISK_CONFIG.max_positions,
        available_money_ratio: float = DEFAULT_RISK_CONFIG.available_money_ratio,
        available_money_source: str = DEFAULT_RISK_CONFIG.available_money_source,
        accepted_risk_overrun: float = DEFAULT_RISK_CONFIG.accepted_risk_overrun,
        reward_risk_ratio: float = DEFAULT_RISK_CONFIG.reward_risk_ratio,
        limit_spread_ratio: float = DEFAULT_RISK_CONFIG.limit_spread_ratio,
        max_spread_ratio: float = DEFAULT_RISK_CONFIG.max_spread_ratio,
        fixed_quantity: Optional[float] = None,
        candle_interval: timedelta = DEFAULT_VOLATILITY_CONFIG.candle_interval,
        candle_lookback: int = DEFAULT_VOLATILITY_CONFIG.candle_lookback,
        minimum_candles: int = DEFAULT_VOLATILITY_CONFIG.minimum_candles,
        granularity_factor: float = DEFAULT_VOLATILITY_CONFIG.granularity_factor,
        loss_shrink_ratio: float = DEFAULT_VOLATILITY_CONFIG.loss_shrink_ratio,
        offset_shrink_ratio: float = DEFAULT_VOLATILITY_CONFIG.offset_shrink_ratio,
        volatility_mode: str = DEFAULT_VOLATILITY_CONFIG.mode,
    ):
        """Configure le sizing, les filtres de spread et la lecture de volatilité.

        Args:
            risk_per_trade: Pourcentage du capital risqué par trade.
            max_positions: Nombre max de positions simultanées.
            available_money_ratio: Part de la source de capital rendue risquable.
            available_money_source: Source de capital: equity, balance ou free_margin.
            accepted_risk_overrun: Dépassement toléré si la taille est forcée.
            reward_risk_ratio: Multiple du stop utilisé pour le take profit.
            limit_spread_ratio: Seuil où un market bascule en limit.
            max_spread_ratio: Seuil au-delà duquel le trade est rejeté.
            fixed_quantity: Taille imposée; sinon sizing par budget de risque.
            candle_interval: Période de bougie utilisée pour mesurer la volatilité.
            candle_lookback: Nombre de bougies conservées pour la vol.
            minimum_candles: Minimum de bougies avant toute décision.
            granularity_factor: Facteur de discrétisation du scale.
            loss_shrink_ratio: Réduit la distance de stop issue des bougies.
            offset_shrink_ratio: Réduit l'écart d'entrée des ordres pendants.
            volatility_mode: swing arrondit au-dessus, intraday au-dessous.
        """
        # Normalise la politique de risque et les garde-fous d'exposition.
        self.risk_config = RiskConfig(
            risk_per_trade_percent=self._normalize_percent(
                risk_per_trade,
                default=DEFAULT_RISK_CONFIG.risk_per_trade_percent,
            ),
            available_money_ratio=self._normalize_ratio(
                available_money_ratio,
                default=DEFAULT_RISK_CONFIG.available_money_ratio,
            ),
            available_money_source=self._normalize_capital_source(available_money_source),
            accepted_risk_overrun=self._normalize_non_negative(
                accepted_risk_overrun,
                default=DEFAULT_RISK_CONFIG.accepted_risk_overrun,
            ),
            reward_risk_ratio=max(
                0.1,
                self._normalize_non_negative(
                    reward_risk_ratio,
                    default=DEFAULT_RISK_CONFIG.reward_risk_ratio,
                ),
            ),
            limit_spread_ratio=max(
                0.0,
                self._normalize_non_negative(
                    limit_spread_ratio,
                    default=DEFAULT_RISK_CONFIG.limit_spread_ratio,
                ),
            ),
            max_spread_ratio=max(
                0.0,
                self._normalize_non_negative(
                    max_spread_ratio,
                    default=DEFAULT_RISK_CONFIG.max_spread_ratio,
                ),
            ),
            max_positions=max(1, int(max_positions)),
            fixed_quantity=fixed_quantity if fixed_quantity is not None and fixed_quantity > 0.0 else None,
        )
        # Normalise la lecture de volatilité dérivée des bougies reconstruites.
        self.volatility_config = VolatilityConfig(
            candle_interval=normalize_candle_interval(
                candle_interval,
                default=DEFAULT_VOLATILITY_CONFIG.candle_interval,
            ),
            candle_lookback=max(1, int(candle_lookback)),
            minimum_candles=max(1, int(minimum_candles)),
            granularity_factor=max(1.0, float(granularity_factor)),
            loss_shrink_ratio=max(1e-9, float(loss_shrink_ratio)),
            offset_shrink_ratio=max(0.0, float(offset_shrink_ratio)),
            mode=volatility_mode if volatility_mode in {"swing", "intraday"} else DEFAULT_VOLATILITY_CONFIG.mode,
        )
        self.candle_builder = CandleBuilder(
            candle_interval=self.volatility_config.candle_interval,
            lookback_candles=self.volatility_config.candle_lookback,
        )

    def on_tick(self, tick: Tick) -> None:
        """Alimente l'historique de volatilité avec chaque tick."""
        self.candle_builder.update(tick)

    def evaluate(
        self,
        signal: Signal,
        portfolio: Portfolio,
        tick: Tick,
        pending_orders_for_symbol: int = 0,
    ) -> Optional[RiskEvaluation]:
        """Refuse ou transforme un signal en plan d'ordre exécutable.

        Args:
            signal: Intention de trade et mode d'entrée demandé.
            portfolio: Etat courant du compte et des positions.
            tick: Bid/ask courant servant au placement.
            pending_orders_for_symbol: Empêche les doublons d'ordres en attente.
        """
        # Garde-fous structurels avant tout calcul de placement.
        if signal.side == SignalSide.HOLD:
            return None
        if len(portfolio.positions) >= self.risk_config.max_positions:
            return None
        if pending_orders_for_symbol > 0 or portfolio.has_open_position(signal.symbol):
            return None

        side = OrderSide.BUY if signal.side == SignalSide.BUY else OrderSide.SELL
        raw_spread = max(0.0, tick.ask - tick.bid)

        # Résume la vol récente en distances directement exploitables.
        volatility = self._get_volatility_snapshot(signal.symbol, raw_spread)
        if volatility is None:
            return None
        if volatility.spread_ratio > self.risk_config.max_spread_ratio:
            return None

        # Place l'entrée puis en déduit SL/TP selon le mode retenu.
        placement = self._build_order_plan(
            signal=signal,
            side=side,
            tick=tick,
            volatility=volatility,
        )
        if placement.entry_price <= 0.0:
            return None
        if placement.stop_loss <= 0.0 or placement.take_profit <= 0.0:
            return None

        # Extrait le capital risquable puis convertit le stop en taille.
        available_capital = self._get_available_capital(portfolio)
        if available_capital <= 0.0:
            return None

        quantity, risk_percent = self._resolve_quantity(
            available_capital=available_capital,
            stop_distance=placement.stop_distance,
        )
        if quantity <= 0.0:
            return None

        return RiskEvaluation(
            side=side,
            order_type=placement.order_type,
            entry_mode=placement.entry_mode,
            entry_price=placement.entry_price,
            stop_loss=placement.stop_loss,
            take_profit=placement.take_profit,
            quantity=quantity,
            available_capital=available_capital,
            capital_at_risk=available_capital * risk_percent / 100.0,
            risk_percent=risk_percent,
            stop_distance=placement.stop_distance,
            take_profit_distance=placement.take_profit_distance,
            raw_spread=raw_spread,
            spread_ratio=volatility.spread_ratio,
            entry_offset=volatility.entry_offset,
        )

    def _get_volatility_snapshot(self, symbol: str, raw_spread: float) -> Optional[_VolatilitySnapshot]:
        """Convertit les bougies récentes en stop, TP et offset d'entrée."""
        stats = self.candle_builder.get_stats(symbol)
        if stats is None or stats.sample_size < self.volatility_config.minimum_candles:
            return None
        if stats.average_size <= 0.0 or stats.max_size <= 0.0:
            return None

        granularity = self.volatility_config.granularity_factor
        scaled_size = granularity * stats.max_size / stats.average_size
        if self.volatility_config.mode == "intraday":
            scale = max(1, floor(scaled_size))
        else:
            scale = max(1, ceil(scaled_size))

        # Le scale renforce une bougie moyenne quand un extrême récent domine.
        loss_distance = (
            scale
            * stats.average_size
            * self.volatility_config.loss_shrink_ratio
            / granularity
        )
        take_profit_distance = self.risk_config.reward_risk_ratio * (loss_distance + raw_spread)
        entry_offset = (
            scale
            * stats.average_size
            * self.volatility_config.offset_shrink_ratio
            / granularity
        )
        if loss_distance <= 0.0:
            return None

        spread_ratio = raw_spread / loss_distance if loss_distance > 0.0 else float("inf")
        return _VolatilitySnapshot(
            stop_distance=loss_distance,
            take_profit_distance=take_profit_distance,
            entry_offset=entry_offset,
            spread_ratio=spread_ratio,
            raw_spread=raw_spread,
        )

    def _build_order_plan(
        self,
        signal: Signal,
        side: OrderSide,
        tick: Tick,
        volatility: _VolatilitySnapshot,
    ) -> _PlacementPlan:
        """Choisit le type d'ordre et construit l'entrée, le SL et le TP."""
        entry_mode = signal.entry_mode
        if (
            entry_mode == SignalEntryMode.MARKET
            and volatility.spread_ratio > self.risk_config.limit_spread_ratio
        ):
            # Spread trop large: on préfère attendre un meilleur prix.
            entry_mode = SignalEntryMode.LIMIT

        if entry_mode == SignalEntryMode.LIMIT:
            if side == OrderSide.BUY:
                entry_price = tick.bid - volatility.entry_offset
            else:
                entry_price = tick.ask + volatility.entry_offset
            stop_distance = self._resolve_stop_distance(
                signal=signal,
                side=side,
                entry_price=entry_price,
                base_stop_distance=volatility.stop_distance,
                raw_spread=volatility.raw_spread,
            )
            take_profit_distance = self.risk_config.reward_risk_ratio * (stop_distance + volatility.raw_spread)
            if side == OrderSide.BUY:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + take_profit_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - take_profit_distance
            return _PlacementPlan(
                order_type=OrderType.LIMIT,
                entry_mode=entry_mode,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_distance=stop_distance,
                take_profit_distance=take_profit_distance,
            )

        if entry_mode == SignalEntryMode.STOP:
            if side == OrderSide.BUY:
                entry_price = tick.ask + volatility.entry_offset
            else:
                entry_price = tick.bid - volatility.entry_offset
            stop_distance = self._resolve_stop_distance(
                signal=signal,
                side=side,
                entry_price=entry_price,
                base_stop_distance=volatility.stop_distance,
                raw_spread=volatility.raw_spread,
            )
            take_profit_distance = self.risk_config.reward_risk_ratio * (stop_distance + volatility.raw_spread)
            if side == OrderSide.BUY:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + take_profit_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - take_profit_distance
            return _PlacementPlan(
                order_type=OrderType.STOP,
                entry_mode=entry_mode,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_distance=stop_distance,
                take_profit_distance=take_profit_distance,
            )

        if side == OrderSide.BUY:
            entry_price = tick.ask
        else:
            entry_price = tick.bid
        stop_distance = self._resolve_stop_distance(
            signal=signal,
            side=side,
            entry_price=entry_price,
            base_stop_distance=volatility.stop_distance,
            raw_spread=volatility.raw_spread,
        )
        take_profit_distance = self.risk_config.reward_risk_ratio * (stop_distance + volatility.raw_spread)
        if side == OrderSide.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + take_profit_distance + volatility.raw_spread
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - take_profit_distance - volatility.raw_spread

        return _PlacementPlan(
            order_type=OrderType.MARKET,
            entry_mode=SignalEntryMode.MARKET,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            stop_distance=stop_distance,
            take_profit_distance=take_profit_distance,
        )

    def _resolve_stop_distance(
        self,
        signal: Signal,
        side: OrderSide,
        entry_price: float,
        base_stop_distance: float,
        raw_spread: float,
    ) -> float:
        structure_stop_distance = self._resolve_structure_stop_distance(
            signal=signal,
            side=side,
            entry_price=entry_price,
            base_stop_distance=base_stop_distance,
            raw_spread=raw_spread,
        )
        if structure_stop_distance is None:
            return base_stop_distance
        return structure_stop_distance

    def _resolve_structure_stop_distance(
        self,
        signal: Signal,
        side: OrderSide,
        entry_price: float,
        base_stop_distance: float,
        raw_spread: float,
    ) -> Optional[float]:
        protective_level = signal.protective_level_price
        if protective_level is None or protective_level <= 0.0:
            return None

        if side == OrderSide.BUY:
            protective_distance = entry_price - protective_level
        else:
            protective_distance = protective_level - entry_price
        if protective_distance <= 0.0:
            return None

        quality = self._normalize_quality(signal.protective_level_quality)
        buffer_ratio = (
            self._STRUCTURE_STOP_BUFFER_RATIO
            + self._STRUCTURE_QUALITY_BUFFER_RATIO * (1.0 - quality)
        )
        buffer = max(base_stop_distance * buffer_ratio, raw_spread)
        min_stop_distance = max(base_stop_distance * self._STRUCTURE_MIN_STOP_RATIO, raw_spread * 2.0)
        return max(protective_distance + buffer, min_stop_distance)

    def _get_available_capital(self, portfolio: Portfolio) -> float:
        """Retourne la fraction de capital autorisée pour le risque."""
        source_value = portfolio.get_capital_source(self.risk_config.available_money_source)
        return max(0.0, source_value * self.risk_config.available_money_ratio)

    def _resolve_quantity(self, available_capital: float, stop_distance: float) -> tuple[float, float]:
        """Convertit budget de risque et distance de stop en quantité."""
        if stop_distance <= 0.0 or available_capital <= 0.0:
            return 0.0, 0.0

        if self.risk_config.fixed_quantity is not None:
            quantity = self.risk_config.fixed_quantity
            risk_percent = 100.0 * stop_distance * quantity / available_capital
            if risk_percent - self.risk_config.risk_per_trade_percent > self.risk_config.accepted_risk_overrun:
                return 0.0, 0.0
            return quantity, risk_percent

        risk_budget = available_capital * self.risk_config.risk_per_trade_percent / 100.0
        quantity = risk_budget / stop_distance
        if quantity <= 0.0:
            return 0.0, 0.0
        risk_percent = 100.0 * stop_distance * quantity / available_capital
        return quantity, risk_percent

    def _normalize_capital_source(self, capital_source: str) -> str:
        normalized = str(capital_source or "equity").strip().lower()
        if normalized not in SUPPORTED_CAPITAL_SOURCES:
            return "equity"
        return normalized

    def _normalize_percent(self, value: float, default: float) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(numeric_value) or numeric_value <= 0.0:
            return default
        if numeric_value < 1.0:
            return numeric_value * 100.0
        return numeric_value

    def _normalize_ratio(self, value: float, default: float) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(numeric_value) or numeric_value <= 0.0:
            return default
        return min(numeric_value, 1.0)

    def _normalize_non_negative(self, value: float, default: float) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(numeric_value) or numeric_value < 0.0:
            return default
        return numeric_value

    def _normalize_quality(self, value: float) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not isfinite(numeric_value):
            return 0.0
        return max(0.0, min(1.0, numeric_value))


@dataclass(frozen=True)
class _PlacementPlan:
    """Placement final utilisé pour le sizing et la construction d'ordre."""

    order_type: OrderType
    entry_mode: SignalEntryMode
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_distance: float
    take_profit_distance: float


@dataclass(frozen=True)
class _VolatilitySnapshot:
    """Snapshot intermédiaire utilisé pour placer puis dimensionner l'ordre."""

    stop_distance: float
    take_profit_distance: float
    entry_offset: float
    spread_ratio: float
    raw_spread: float