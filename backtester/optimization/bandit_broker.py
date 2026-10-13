"""Broker event integration. This bridge has no order-sending interface.

The host owns execution/reconciliation and supplies authoritative net deal
amounts and initial monetary risk in the same named account currency.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite

from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.optimization.bandit_core import PositionLearningRecord, stable_id
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.reports.trade_report import TradeReport


@dataclass(frozen=True)
class BrokerDeal:
    position_id: str
    deal_id: str
    timestamp: datetime
    quantity: float
    exit_price: float
    pnl_net_account_currency: float
    account_currency: str
    complete: bool
    reason: CloseReason = CloseReason.STOP_LOSS
    pnl_gross_account_currency: float | None = None


@dataclass(frozen=True)
class BrokerCloseInstruction:
    position_id: str
    timestamp: datetime
    reason: CloseReason = CloseReason.END_OF_WEEK


class BrokerBanditBridge:
    def __init__(self, coordinator, *, account_scope):
        if coordinator.source not in ("live_real", "demo_executed"):
            raise ValueError("Broker events require live_real or demo_executed provenance")
        if not account_scope or coordinator.engine is None:
            raise ValueError("A bound coordinator and a stable broker/account scope are required")
        self.coordinator = coordinator
        self.account_scope = account_scope
        self.positions = {}
        self.reconciliation = {}
        rows = [*coordinator.store.pending_records("broker_position"), *coordinator.store.pending_records("broker_completed")]
        for row in rows:
            data = row["payload"]
            if data["account_scope"] == account_scope:
                self.positions[data["position_id"]] = data
                if row["kind"] == "broker_completed":
                    self.reconciliation[data["position_id"]] = "closed_deals_committed"

    def register_position(self, position_id, record, *, symbol, side, entry_price,
                          opened_at, initial_risk_account_currency, quantity, account_currency):
        if record.source != self.coordinator.source:
            raise ValueError("Position provenance does not match broker mode")
        if not account_currency or account_currency == "engine_price_quantity":
            raise ValueError("Broker positions need an explicit account-currency contract")
        if not isfinite(quantity) or quantity <= 0 or not isfinite(entry_price):
            raise ValueError("Invalid actual broker fill")
        bundle = self.coordinator.models[symbol]
        if bundle["identity"].model_id != record.decision.model_id:
            raise ValueError("Broker fill belongs to a different fixed model")
        if record.account_currency != account_currency:
            raise ValueError("Broker and recorded risk currencies differ")
        risk = initial_risk_account_currency
        if risk is not None and (not isfinite(risk) or risk <= 0):
            risk = None  # retained in audit, never fabricated or learned
        key = stable_id([record.decision.model_id, self.account_scope, str(position_id)])
        frozen = replace(record, trajectory_key=key, initial_risk=risk, initial_quantity=quantity,
                         provenance={**record.provenance, "broker_scope": self.account_scope,
                                     "broker_position_id": str(position_id), "costs": "authoritative broker net deals"})
        data = {"position_id": str(position_id), "account_scope": self.account_scope, "symbol": symbol,
                "side": side.name, "entry_price": entry_price, "opened_at": opened_at.isoformat(),
                "record": frozen.to_dict()}
        existing = self.positions.get(str(position_id))
        if existing is not None:
            # Reconciliation must reuse the entry snapshot, never today's policy.
            if (existing["record"]["decision"] != data["record"]["decision"]
                    or existing["record"]["initial_risk"] != risk):
                raise ValueError("Cannot replace an existing broker entry snapshot")
            return PositionLearningRecord.from_dict(existing["record"])
        self.positions[str(position_id)] = data
        self.coordinator.store.save_pending(frozen.decision.model_id, frozen.run_id, key, "broker_position", data)
        return frozen

    def entry_allowed(self, emitted_at, *, timestamp=None):
        now = timestamp or self.coordinator.market_clock or emitted_at
        return self.coordinator.engine.trading_schedule.pending_rejection(emitted_at, now) is None

    def weekly_close_requests(self, timestamp):
        """Instructions for the host's broker transport/timer; no virtual real fills."""
        policy = self.coordinator.engine.trading_schedule
        return [BrokerCloseInstruction(position_id, timestamp) for position_id, data in sorted(self.positions.items())
                if self.reconciliation.get(position_id) not in ("closed_deals_committed", "awaiting_authoritative_close_deals")
                and policy.position_due(datetime.fromisoformat(data["opened_at"]), timestamp)]

    def on_market_tick(self, tick):
        # Deliberately bypass native pending fills and native trade closure:
        # only broker events may close the principal execution trajectory.
        coordinator = self.coordinator
        coordinator.engine._market_clock = tick.timestamp
        coordinator.engine.market_state.update(tick)
        coordinator.on_market_tick(tick)
        coordinator.engine.risk_manager.on_tick(tick)
        coordinator.shadow.on_tick(tick)
        coordinator.flush_closures(tick.timestamp)
        coordinator.after_tick(tick)

    def process_deals(self, deals, *, timestamp):
        coordinator = self.coordinator
        # Partials precede the final flag within each position at equal time.
        ordered = sorted(deals, key=lambda deal: (deal.timestamp, str(deal.position_id), deal.complete, str(deal.deal_id)))
        completed = []
        for deal in ordered:
            data = self.positions.get(str(deal.position_id))
            if data is None:
                raise ValueError(f"Broker position {deal.position_id} has no persisted entry snapshot")
            record = PositionLearningRecord.from_dict(data["record"])
            if record.account_currency != deal.account_currency:
                raise ValueError("Net PnL and initial risk must have identical currency units")
            if not isfinite(deal.quantity) or deal.quantity <= 0 or not deal.deal_id:
                raise ValueError("Broker deals need stable IDs and valid quantities")
            report = TradeReport(
                0, data["symbol"], OrderSide[data["side"]], datetime.fromisoformat(data["opened_at"]),
                deal.timestamp, data["entry_price"], deal.exit_price, deal.quantity,
                deal.pnl_net_account_currency, 0.0, 0.0, deal.reason,
                trailing_stop_enabled=record.decision.action.trailing_enabled, learning_record=record,
                is_complete=deal.complete,
                event_id=stable_id([record.decision.model_id, self.account_scope, str(deal.deal_id)]),
                pnl_gross=deal.pnl_gross_account_currency,
            )
            coordinator.queue_trade(report)
            if deal.complete:
                completed.append((data, record))
        coordinator.flush_closures(timestamp)
        for data, record in completed:
            self.reconciliation[data["position_id"]] = "closed_deals_committed"
        # Keep completed snapshots so delayed broker replays remain idempotent.
        # In-memory entries survive this bridge's lifetime; the closure audit
        # persists their snapshots across later restarts.

    def reconcile_positions(self, open_position_ids):
        present = {str(value) for value in open_position_ids}
        for position_id, data in self.positions.items():
            if self.reconciliation.get(position_id) == "closed_deals_committed":
                continue
            state = "broker_open" if position_id in present else "awaiting_authoritative_close_deals"
            self.reconciliation[position_id] = state
            if state != "broker_open":
                record = PositionLearningRecord.from_dict(data["record"])
                self.coordinator.store.save_pending(record.decision.model_id, record.run_id,
                                                    stable_id([record.trajectory_key, "reconciliation"]),
                                                    "broker_incomplete", {"position_id": position_id, "state": state})
        return dict(self.reconciliation)
