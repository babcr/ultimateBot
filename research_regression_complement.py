"""Paired causal replays on original quotes with shared multi-symbol capital.

The fixed-config replay jumps only over quotes that cannot fill/expire an order,
hit a stop/target, advance trailing, or close an analysis candle. It restores
actual candle history before every decision and marks skipped quote excursions.
No precomputed signals, sampled prices, or trade-subset projections are used.
"""
import argparse
from collections import defaultdict,deque
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date, timedelta
import io
import json
from pathlib import Path
import sys

import numpy as np

from ultimateBot.audit_breakout_frequency import aggregate_quotes, timestamp
from ultimateBot.audit_hybrid_trades import dates_between, read_quotes, source_path, write_json
from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.orders.order import OrderSide, OrderType
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.run_backtest import build_engine, parse_args
from ultimateBot.run_regression_hybrid import hybrid_config


def build_variant(name):
    config = hybrid_config()
    # Freeze the experiment recipes independently of future launcher defaults.
    config = replace(config,regression=replace(config.regression,max_spread_stop_ratio=.10))
    if name == "original":
        config = replace(config,regression=replace(config.regression,max_spread_stop_ratio=0.),
                         reentry=replace(config.reentry,quality_mode="legacy"))
    elif name in ("quality","quality-pullback","quality-bounce","quality-impulse"):
        config = replace(config,regression=replace(config.regression,max_spread_stop_ratio=0.))
    elif name == "spread":
        config = replace(config,reentry=replace(config.reentry,quality_mode="legacy"))
    elif name not in ("filtered","pullback","bounce"):
        raise ValueError("Unknown variant")
    if name in ("pullback","bounce","quality-pullback","quality-bounce"):
        config = replace(config,pullback=replace(config.pullback,enabled=True,regime="range" if name.endswith("bounce") else "trend"))
    if name == "quality-impulse":
        config = replace(config,impulse=replace(config.impulse,enabled=True))
    previous = sys.argv
    try:
        sys.argv = ["research"]
        engine = build_engine([],parse_args(config))
    finally:
        sys.argv = previous
    return engine


def load_history(history,symbol,start,end,minutes,cache):
    cache.mkdir(parents=True,exist_ok=True)
    bars = []
    paths = [source_path(history,symbol,day) for day in dates_between(start,end)]
    paths = [path for path in paths if path.exists()]
    for index,path in enumerate(paths):
        target = cache/f"{symbol}_{path.stem}_{minutes}m.json"
        contract = [path.stat().st_size,path.stat().st_mtime_ns,minutes,"round-trip-v2"]
        try:
            saved = json.loads(target.read_text()) if target.exists() else None
        except json.JSONDecodeError:
            saved = None
        if not saved or saved["contract"] != contract:
            quotes = read_quotes(path,symbol,float_precision="round_trip")
            saved = dict(contract=contract,bars=aggregate_quotes(quotes,minutes*60_000))
            target.write_text(json.dumps(saved),encoding="utf-8")
        bars.extend(saved["bars"])
        if index%100 == 0 or index == len(paths)-1:
            print(f"{symbol}: candle coverage {index+1}/{len(paths)}",flush=True)
    return bars


def first_true(mask):
    return int(mask.argmax()) if mask.any() else None


def next_trigger(engine,symbol,quotes,start):
    """First raw quote whose processing can change exposure or exit state."""
    if start >= len(quotes):
        return None
    times,bids,asks = quotes[start:,0],quotes[start:,1],quotes[start:,2]
    mask = np.zeros(len(times),dtype=bool)
    for position in engine.portfolio.positions.values():
        if position.symbol != symbol:
            continue
        buying = position.side == OrderSide.BUY
        price = bids if buying else asks
        direction = 1 if buying else -1
        if position.stop_loss is not None:
            mask |= direction*(price-position.stop_loss) <= 0
        if position.take_profit is not None:
            mask |= direction*(price-position.take_profit) >= 0
        state = position.trailing_stop_state
        if state is not None:
            threshold = state.activation_price+direction*(state.steps+1 if state.active else 0)*state.initial_stop_distance
            # Conservatively include a tiny ULP margin, matching the real
            # trailing manager's decimal-step tolerance. False events are safe.
            mask |= direction*(price-threshold) >= -4*np.spacing(np.abs(price))
    for order in engine.pending_orders_processor.queue.get_for_symbol(symbol):
        if order.expires_at is not None:
            mask |= times >= round(order.expires_at.timestamp()*1000)
        eligible = max(order.created_at,order.eligible_at or order.created_at)
        ready = times > round(eligible.timestamp()*1000)
        if order.expires_at is not None:
            ready &= times < round(order.expires_at.timestamp()*1000)
        if order.order_type == OrderType.LIMIT:
            ready &= asks <= order.limit_price if order.side == OrderSide.BUY else bids >= order.limit_price
        elif order.order_type == OrderType.STOP:
            ready &= asks >= order.stop_price if order.side == OrderSide.BUY else bids <= order.stop_price
        mask |= ready
    index = first_true(mask)
    return start+index if index is not None else None


def seed_decision(engine,symbol,bars,index,current,share_fit):
    market = engine.market_state
    capacity = engine.strategy.history_capacity_for_lookback(engine.strategy.history_lookback_bars)
    market_key = ('market',capacity)
    if market_key not in share_fit:
        # Closed records are read-only here; each variant keeps a separate
        # deque, swing cache and active candle.
        share_fit[market_key] = {source:[Candle(timestamp(row["start_ms"]),*row[source],row["count"],*row["volumes"])
            for row in bars[max(0,index-capacity+1):index+1]] for source in ('bid','ask','mid')}
    for source,builder in market._candle_builders.items():
        builder._active.pop(symbol,None)
        builder._swing_histories.pop(symbol,None)
        closed = share_fit[market_key][source]
        builder._closed[symbol] = deque(closed,maxlen=builder.lookback_candles)
        builder._candle_counts[symbol] = len(closed)
    market._point_sizes[symbol] = .00001
    original_fit = engine.strategy._fit_channel_before_event

    def cached_fit(*args):
        if "fit" not in share_fit:
            share_fit["fit"] = original_fit(*args)
        return share_fit["fit"]

    engine.strategy._fit_channel_before_event = cached_fit
    try:
        row = bars[index]
        close = Tick(timestamp(row["last_ms"]),symbol,row["bid"][3],row["ask"][3],0,0)
        context = StrategyContext(engine.portfolio,market,engine.pending_orders_processor.queue.count_for_symbol(symbol))
        signals = engine.strategy.on_market_data(TickBatch(symbol,[close]),context)
        if signals:
            # The strategy reads market history only. Restore risk history
            # only when it will actually evaluate a signal, rather than on
            # tens of thousands of empty/busy closed-candle decisions.
            builder = engine.risk_manager.candle_builder
            builder._active.pop(symbol,None)
            builder._swing_histories.pop(symbol,None)
            risk_key = ('risk',builder.lookback_candles)
            if risk_key not in share_fit:
                share_fit[risk_key] = [Candle(timestamp(row["start_ms"]),*row["mid"],row["count"],*row["volumes"])
                    for row in bars[max(0,index-builder.lookback_candles+1):index+1]]
            builder._closed[symbol] = deque(share_fit[risk_key],maxlen=builder.lookback_candles)
            builder._candle_counts[symbol] = len(share_fit[risk_key])
            builder.update(current)
        for signal in signals:
            engine._process_signal(signal,close)
    finally:
        engine.strategy._fit_channel_before_event = original_fit


def record_fills(engine):
    risks,contexts = {},{}
    original = engine.position_manager.open_position

    def opened(*args,**kwargs):
        position = original(*args,**kwargs)
        risks[position.trade_id] = abs(position.entry_price-position.stop_loss)*position.quantity*position.profit_multiplier
        contexts[position.trade_id] = dict(kwargs["order"].strategy_entry_context or {})
        return position

    engine.position_manager.open_position = opened
    return risks,contexts


def replay(engines,history,symbols,start,end,bars,log,lazy_quotes=False):
    if any(engine.trading_schedule.config.enabled or engine.adaptive_optimizer or engine.contextual_optimizer
           or engine.trade_validator or engine.trade_dataset or engine.risk_manager.risk_config.max_positions < len(symbols)
           or engine.strategy.regression_rules["regression_entry_mode"] != "hybrid" for engine in engines.values()):
        raise ValueError("Research replay requires fixed hybrid, no schedules/learning/dataset/validator, and enough shared position slots")
    recorded = {name:record_fills(engine) for name,engine in engines.items()}
    decisions = defaultdict(dict)
    for symbol in symbols:
        for index in range(len(bars[symbol])-1):
            decisions[bars[symbol][index+1]["first_ms"]][symbol] = index
    last_ticks = {}
    processed = 0
    with log.open("w",encoding="utf-8") as output,redirect_stdout(output):
        for day in dates_between(start,end):
            arrays = {}

            def load_symbol(symbol):
                path = source_path(history,symbol,day)
                arrays[symbol] = (read_quotes(path,symbol,float_precision="round_trip")[["timestamp","bid","ask","bid_volume","ask_volume"]].to_numpy()
                                  if path.exists() else np.empty((0,5)))

            for symbol in symbols:
                arrays[symbol] = None
                if not lazy_quotes or any(engine._hybrid_symbol_busy(symbol) for engine in engines.values()):
                    load_symbol(symbol)
            cursors = {symbol:0 for symbol in symbols}
            day_decisions = [(ms,symbols.index(symbol),symbol,index)
                for ms,entries in decisions.items() if timestamp(ms).date() == day for symbol,index in entries.items()]
            day_decisions.sort()
            decision_index = 0

            def advance(symbol,stop):
                values = arrays[symbol][cursors[symbol]:stop]
                if len(values):
                    for engine in engines.values():
                        for position in engine.portfolio.positions.values():
                            if position.symbol == symbol:
                                prices = values[:,1 if position.side == OrderSide.BUY else 2]
                                engine.trade_monitor._mark_position_excursion(position,float(prices.min()))
                                engine.trade_monitor._mark_position_excursion(position,float(prices.max()))
                    last_ticks[symbol] = Tick(timestamp(values[-1,0]),symbol,*map(float,values[-1,1:]))
                cursors[symbol] = stop

            while True:
                events = []
                for rank,symbol in enumerate(symbols):
                    if arrays[symbol] is None:
                        continue
                    for engine in engines.values():
                        index = next_trigger(engine,symbol,arrays[symbol],cursors[symbol])
                        if index is not None:
                            events.append((int(arrays[symbol][index,0]),rank,symbol,index))
                if decision_index < len(day_decisions):
                    ms,rank,symbol,bar_index = day_decisions[decision_index]
                    index = -1 if arrays[symbol] is None else int(np.searchsorted(arrays[symbol][:,0],ms))
                    if arrays[symbol] is not None and (index < cursors[symbol] or index >= len(arrays[symbol]) or arrays[symbol][index,0] != ms):
                        raise ValueError("Decision quote missing or already processed")
                    events.append((ms,rank,symbol,index))
                if not events:
                    for symbol in symbols:
                        if arrays[symbol] is not None:
                            advance(symbol,len(arrays[symbol]))
                    break
                ms,rank,symbol,index = min(events)
                for other_rank,other in enumerate(symbols):
                    if arrays[other] is None:
                        continue
                    stop = index if other == symbol else int(np.searchsorted(arrays[other][:,0],ms,
                                                         side="right" if other_rank < rank else "left"))
                    advance(other,stop)
                if index == -1:
                    # Flat, price-only models need only the actual first quote
                    # already retained in OHLC metadata. Quote sizes are not
                    # inputs to any allowed strategy/risk/execution component.
                    bar_index = day_decisions[decision_index][3]
                    opening = bars[symbol][bar_index+1]
                    tick = Tick(timestamp(ms),symbol,opening["bid"][0],opening["ask"][0],0.,0.)
                else:
                    values = arrays[symbol][index]
                    tick = Tick(timestamp(ms),symbol,*map(float,values[1:]))
                last_ticks[symbol] = tick
                for name,engine in engines.items():
                    print(f"[variant] {name}")
                    engine.market_state.last_ticks.update(last_ticks)
                    engine.on_tick(tick,update_market=False,allow_entries=False)
                if index != -1:
                    cursors[symbol] = index+1
                processed += 1
                if decision_index < len(day_decisions) and day_decisions[decision_index][:3] == (ms,rank,symbol):
                    bar_index = day_decisions[decision_index][3]
                    shared = {}
                    for name,engine in engines.items():
                        print(f"[decision-variant] {name}")
                        seed_decision(engine,symbol,bars[symbol],bar_index,tick,shared)
                    if arrays[symbol] is None and any(engine._hybrid_symbol_busy(symbol) for engine in engines.values()):
                        load_symbol(symbol)
                        actual = int(np.searchsorted(arrays[symbol][:,0],ms))
                        if (actual >= len(arrays[symbol]) or arrays[symbol][actual,0] != ms
                                or arrays[symbol][actual,1] != tick.bid or arrays[symbol][actual,2] != tick.ask):
                            raise ValueError("Cached first quote differs from raw execution history")
                        # This quote's monitoring phase ran before the order
                        # was queued. Only subsequent quotes may fill it.
                        cursors[symbol] = actual+1
                    decision_index += 1
            if day.day == 1:
                print(f"Progress through {day}",file=sys.__stdout__,flush=True)
    summaries = {}
    for name,engine in engines.items():
        risks,contexts = recorded[name]
        trades = []
        for trade in engine.reports.trades:
            context = contexts[trade.trade_id]
            trades.append(dict(trade_id=trade.trade_id,symbol=trade.symbol,side=trade.side.name,
                opened_at=trade.opened_at,closed_at=trade.closed_at,entry_price=trade.entry_price,exit_price=trade.exit_price,
                pnl=trade.pnl,r=trade.pnl/risks[trade.trade_id],branch=engine.trade_entry_modes[trade.trade_id],
                close_reason=trade.close_reason.value,mfe_r=trade.max_favorable_pnl/risks[trade.trade_id],
                mae_r=trade.max_adverse_pnl/risks[trade.trade_id],context=context))
        summaries[name] = dict(closed=len(trades),sum_r=sum(trade["r"] for trade in trades),
            profit_factor_r=sum(max(0,trade["r"]) for trade in trades)/max(1e-12,-sum(min(0,trade["r"]) for trade in trades)),
            balance=engine.portfolio.balance,open=len(engine.portfolio.positions),pending=len(engine.pending_orders_processor.queue),
            initial_balance=engine.portfolio.initial_balance,effective_rules=engine.strategy.regression_rules,
            processed_quote_events=processed,trades=trades,diagnostics=engine.strategy.diagnostics_summary(),
            rejection_counts={str(key):value for key,value in engine.entry_rejection_counts.items()})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,default=Path("C:/jforex_history"))
    parser.add_argument("--symbols",nargs="+",default=["EURUSD","GBPUSD","USDCAD","USDCHF"])
    parser.add_argument("--start-date",type=date.fromisoformat,default=date(2024,1,1))
    parser.add_argument("--end-date",type=date.fromisoformat,default=date(2025,12,31))
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"artifacts/regression_opportunities")
    parser.add_argument("--variants",nargs="+",choices=["original","quality","spread","filtered","pullback","bounce","quality-pullback","quality-bounce","quality-impulse"],
                        default=["original","quality","spread","filtered","quality-pullback","quality-bounce"])
    parser.add_argument("--lazy-quotes",action="store_true",help="While all variants are flat on a symbol, use cached first-quote metadata; load every executable raw quote as soon as any order is queued")
    args = parser.parse_args()
    if args.start_date > args.end_date or len(set(args.symbols)) != len(args.symbols):
        parser.error("Invalid bounds or repeated symbols")
    args.output.mkdir(parents=True,exist_ok=True)
    cache = Path(__file__).parent/"artifacts/hybrid_trade_audit/candle_cache"
    bars = {symbol:load_history(args.history,symbol,args.start_date,args.end_date,60,cache) for symbol in args.symbols}
    if any(not rows for rows in bars.values()):
        parser.error("Every selected symbol needs quote history in the selected period")
    engines = {name:build_variant(name) for name in args.variants}
    results = replay(engines,args.history,args.symbols,args.start_date,args.end_date,bars,args.output/"replay.log",lazy_quotes=args.lazy_quotes)
    write_json(args.output/"results.json",dict(start=args.start_date,end=args.end_date,symbols=args.symbols,
        method="quote-trigger-jumps-with-original-engine-and-shared-capital",lazy_quotes=args.lazy_quotes,terminal_unfillable_signals_omitted=True,results=results))
    print(json.dumps({name:{key:value for key,value in summary.items() if key not in ("trades","diagnostics")}
                      for name,summary in results.items()}),flush=True)


if __name__ == "__main__":
    main()
