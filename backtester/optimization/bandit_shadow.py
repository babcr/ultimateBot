"""Bounded virtual portfolios, each with its own strategy and immutable action."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.optimization.adaptive_strategy import RollingTradeProfitFactor
from ultimateBot.backtester.optimization.bandit_core import BanditAction, DecisionSnapshot, stable_id
from ultimateBot.backtester.optimization.bandit_checkpoint import engine_snapshot, restore_engine


class ShadowVariant:
    def __init__(self, runner, group, action, snapshot, symbol):
        # Import lazily: the real engine imports the optimizer only through its factory.
        from ultimateBot.backtester.engine.backtest_engine import BacktestEngine
        self.runner = runner
        self.group = group
        self.action = action
        self.symbol = symbol
        self.variant_id = action.action_id
        self.snapshot = replace(snapshot, action=action, reason="shadow_group_frozen")
        self.accept_entries = True
        main = runner.coordinator.engine
        strategy = runner.coordinator.strategy_factory()
        strategy.set_history_lookback_bars(action.lookback, symbol=symbol)
        strategy.set_protective_stop_buffer_atr(action.stop_extra_atr_multiplier, symbol=symbol)
        risk = deepcopy(main.risk_manager)
        risk.risk_percent_provider = None
        portfolio = Portfolio(main.portfolio.balance)
        builder = OrderBuilder(execution_delay=main.order_builder.execution_delay,
                               expiration_bars=main.order_builder.expiration_bars,
                               candle_interval=main.order_builder.candle_interval)
        self.engine = BacktestEngine(
            feed=[], batcher=TickBatcher(main.batcher.batch_size, analysis_interval=strategy.analysis_interval,
                                        keep_last_tick_only=True),
            strategy=strategy, risk_manager=risk, order_builder=builder,
            # A shadow engine constructs its virtual executor itself. It is
            # never passed, copied or given a reference to the real executor.
            execution_model=ExecutionModel(), portfolio=portfolio, position_manager=PositionManager(),
            trade_monitor=TradeMonitor(), trade_closer=TradeCloser(action.trailing_enabled),
            reports=ReportCollector(), market_state=main.market_state, learning_hooks=self,
        )
        self.engine.reports.add_trade_event_listener(self.on_trade_event)
        self.pf = RollingTradeProfitFactor(runner.coordinator.config.profit_factor_window)
        self.engine.reports.add_trade_listener(lambda report: self.pf.add(report.pnl))
        self.max_equity = portfolio.initial_balance
        self.drawdown = 0.0

    def prepare_order(self, order, evaluation, signal, tick):
        self.runner.coordinator.attach_order(order, evaluation, signal, tick, self.snapshot,
                                             source="shadow", group_id=self.group["group_id"], variant_id=self.variant_id)

    def on_order_queued(self, order):
        self.runner.coordinator.dirty = True

    def on_position_open(self, position, tick):
        self.runner.coordinator.position_opened(position, tick, persist=False)

    def on_trade_event(self, report):
        self.runner.coordinator.queue_trade(report)

    def retire(self):
        self.accept_entries = False
        # Rejected/expired unfilled entries have no fictitious outcome.
        self.engine.pending_orders_processor.queue.take_for_symbol(self.symbol)

    def on_tick(self, tick):
        if tick.symbol != self.symbol:
            return
        self.engine.on_tick(tick, update_market=False, allow_entries=self.accept_entries)
        equity = self.engine.portfolio.balance + self.unrealized(tick)
        self.max_equity = max(self.max_equity, equity)
        self.drawdown = max(self.drawdown, self.max_equity - equity)

    def unrealized(self, tick):
        return sum(((tick.bid - position.entry_price) if position.side.name == "BUY"
                    else (position.entry_price - tick.ask)) * position.quantity * position.profit_multiplier
                   for position in self.engine.portfolio.positions.values())

    def summary(self):
        tick = self.runner.coordinator.engine.market_state.get_tick(self.symbol)
        portfolio = self.engine.portfolio
        return {"group_id": self.group["group_id"], "variant_id": self.variant_id, "action": self.action.to_dict(),
                "new_entries_allowed": self.accept_entries, "closed_trades": len(self.engine.reports.trades),
                "pnl_net": portfolio.balance - portfolio.initial_balance, "rolling_pf": self.pf.value,
                "drawdown": self.drawdown, "open_positions": len(portfolio.positions),
                "pending_orders": len(self.engine.pending_orders_processor.queue),
                "unrealized_pnl": self.unrealized(tick) if tick else 0.0,
                "capital_at_risk": portfolio.capital_at_risk,
                "weight_budget": self.group["budget"], "simulation": "existing bid/ask virtual fill; no market impact"}

    def checkpoint(self):
        return {"group": self.group, "action": self.action.to_dict(), "snapshot": self.snapshot.to_dict(),
                "symbol": self.symbol, "accept_entries": self.accept_entries, "engine": engine_snapshot(self.engine),
                "pf": list(self.pf.pnl_samples), "max_equity": self.max_equity, "drawdown": self.drawdown}


class ShadowRunner:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.variants = []
        self.archived = []
        self.groups = {}
        self.current_groups = {}

    def choose_pool(self, bundle, context, capacity):
        state, space = bundle["state"], bundle["space"]
        selected = [bundle["decision"].action]
        baseline = BanditAction.from_dict(state["baseline"])
        if baseline.action_id != selected[0].action_id:
            selected.append(baseline)
        from ultimateBot.backtester.optimization.bandit_core import FrozenScaler, LinUCBPolicy
        scaler = FrozenScaler(self.coordinator.config.warmup, state["scaler"])
        valid = self.coordinator.valid_action_ids(bundle["identity"].payload["symbol"])
        candidates = [action for action in space.actions if action.lookback <= capacity and action.action_id in valid]
        if scaler.frozen and context.valid:
            policy = LinUCBPolicy(self.coordinator.config, [BanditAction.from_dict(a) for a in state["actions"]], state["policy"])
            scores = policy.scores(scaler.transform(context), [action.action_id for action in candidates])
            by_mean = sorted(scores, key=lambda item: (-item["mean"], item["action_id"]))
            by_bonus = sorted(scores, key=lambda item: (-item["bonus"], item["action_id"]))
            ranked = [space.by_id[row["action_id"]] for pair in zip(by_mean, by_bonus) for row in pair]
        else:
            # Evenly distributed initial samples, no policy/RNG mutation per tick.
            ranked = [candidates[index] for index in range(0, len(candidates), max(1, len(candidates) // max(1, self.coordinator.config.shadow_limit)))]
            ranked.extend(candidates)
        seen = {action.action_id for action in selected}
        for action in ranked:
            if action.action_id not in seen:
                selected.append(action)
                seen.add(action.action_id)
        return selected

    def start(self, symbol, timestamp, context):
        limit = self.coordinator.config.shadow_limit
        slots = limit - len(self.variants)
        if slots <= 0 or not context.valid or symbol in self.current_groups:
            return
        bundle = self.coordinator.models[symbol]
        capacity = self.coordinator.available_bars(symbol)
        actions = self.choose_pool(bundle, context, capacity)[:slots]
        if not actions:
            return
        group_id = stable_id([bundle["identity"].model_id, self.coordinator.trace_digest,
                              timestamp.isoformat(), sorted(action.action_id for action in actions)])
        excluded = bundle["decision"].action.action_id
        group = {"group_id": group_id, "started_at": timestamp.isoformat(), "symbol": symbol,
                 "start_bar": self.coordinator.closed_bars(symbol), "main_trades": 0,
                 "learners": [action.action_id for action in actions if action.action_id != excluded],
                 "budget": self.coordinator.config.shadow_budget, "counts": {}, "spent": 0.0,
                 "context": context.to_dict(), "model_id": bundle["identity"].model_id}
        self.coordinator.store.register_group(bundle["identity"].model_id, group)
        self.groups[group_id] = group
        self.current_groups[symbol] = group_id
        snapshot = replace(bundle["decision"], context=context)
        self.variants.extend(ShadowVariant(self, group, action, snapshot, symbol) for action in actions)
        self.coordinator.dirty = True

    def on_tick(self, tick):
        for variant in list(self.variants):
            group = variant.group
            expired = (self.coordinator.closed_bars(variant.symbol) - group["start_bar"] >= self.coordinator.config.effective_group_bars
                       or group["main_trades"] >= self.coordinator.config.effective_group_trades)
            if expired and variant.accept_entries:
                variant.retire()
                self.current_groups.pop(variant.symbol, None)
                self.coordinator.dirty = True
            variant.on_tick(tick)
            if not variant.accept_entries and not variant.engine.portfolio.positions:
                self.archived.append(variant.summary())
                self.variants.remove(variant)

    def main_closed(self, symbol):
        group_id = self.current_groups.get(symbol)
        if group_id:
            self.groups[group_id]["main_trades"] += 1

    def rotate(self, symbol, timestamp, context):
        # Rotation is called after a closure batch. Open retired trades drain
        # without liquidation, and still occupy one of the pool slots.
        self.start(symbol, timestamp, context)

    def checkpoint(self):
        return {"variants": [variant.checkpoint() for variant in self.variants], "archived": self.archived,
                "groups": self.groups, "current_groups": self.current_groups}

    def restore(self, state):
        self.archived = state["archived"]
        self.groups = state["groups"]
        self.current_groups = state["current_groups"]
        self.variants = []
        for saved in state["variants"]:
            group = self.groups[saved["group"]["group_id"]]
            variant = ShadowVariant(self, group, BanditAction.from_dict(saved["action"]),
                                    DecisionSnapshot.from_dict(saved["snapshot"]), saved["symbol"])
            restore_engine(variant.engine, saved["engine"])
            variant.accept_entries = saved["accept_entries"]
            for pnl in saved["pf"]:
                variant.pf.add(pnl)
            variant.max_equity, variant.drawdown = saved["max_equity"], saved["drawdown"]
            self.variants.append(variant)
