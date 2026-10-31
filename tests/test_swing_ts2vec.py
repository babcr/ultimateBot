from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.research.swing_series import (
    FEATURE_SCHEMA, SWING_SCHEMA, SwingPoint, SwingSeries, SwingWindow, swing_features,
)
from ultimateBot.backtester.research.swing_opportunities import (
    SCHEMA, OPPORTUNITY_SCHEMA, OpportunityConfig, OpportunitySimulator, collect, connect, instant, milliseconds, read_day,
)
from ultimateBot.backtester.research.swing_encoder import SwingEncoder, SwingEncoderConfig
from ultimateBot.backtester.research.swing_resolver import (
    ResearchProtocol, ResolverConfig, SwingResolver, TrainingSelection, frozen_test, purged_split, select_collection, train,
)
from ultimateBot.run_swing_ts2vec import export, parse_args, training_selection, windows_for


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
PROTOCOL=ResearchProtocol('2024-01-01','2024-02-15','2024-02-16','2024-03-04')


def bar(hour, o=100., h=103., l=99., c=102.):
    return Candle(START + hour * HOUR, o, h, l, c)


def plus(c, delta=.1):
    return replace(c, open_price=c.open_price+delta, high_price=c.high_price+delta,
                   low_price=c.low_price+delta, close_price=c.close_price+delta)


def points(count=4, end=START + 12*HOUR):
    return tuple(SwingPoint("high" if i % 2 == 0 else "low", 101. if i % 2 == 0 else 100.,
                            end-(count-i)*HOUR, end-(count-i+1)*HOUR, end-(count-i-1)*HOUR)
                 for i in range(count))


def memory_db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('run',?,'running','{}')", (START.isoformat(),))
    return db


def add_event(db, swing, event_id="event", symbol="EURUSD"):
    db.execute("""INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,confirmed_at,
        stop_price,expires_at,status) VALUES(?,'run',?,?,?,?,?,?,?,?,'pending')""",
        (event_id, symbol, swing.kind, swing.side, swing.pivot_at.isoformat(), swing.pair_start.isoformat(),
         swing.confirmed_at.isoformat(), swing.price, (swing.confirmed_at+2*HOUR).isoformat()))


class SwingSeriesTests(unittest.TestCase):
    def test_high_uses_bid_low_uses_ask_and_confirmation_is_second_close(self):
        series = SwingSeries(HOUR, (SwingWindow("pairs", 1),))
        a, b, c = bar(0), bar(1, 102, 104, 100, 101), bar(2, 101, 103, 98, 102)
        self.assertIsNone(series.append(a, plus(a)))
        high = series.append(b, plus(b))
        low = series.append(c, plus(c))
        self.assertEqual((high.kind, high.price, high.side, high.confirmed_at), ("high", 104., -1, START+2*HOUR))
        self.assertEqual((low.kind, low.price, low.side, low.confirmed_at), ("low", 98.1, 1, START+3*HOUR))
        self.assertEqual(tuple(series.swings), (high, low))

    def test_latest_same_kind_replaces_even_if_less_extreme_and_snapshots_do_not_change(self):
        series = SwingSeries(HOUR, (SwingWindow("pairs", 2),))
        ask = lambda h: bar(h, 120, 122, 119, 121)
        series.append(bar(0, 100, 112, 99, 105), ask(0))
        first = series.append(bar(1, 105, 110, 102, 103), ask(1))
        frozen = tuple(series.swings)
        series.append(bar(2, 103, 108, 102, 106), ask(2))
        latest = series.append(bar(3, 106, 109, 103, 104), ask(3))
        self.assertLess(latest.price, first.price)
        self.assertEqual(tuple(series.swings), (latest,))
        self.assertEqual(frozen, (first,))
        self.assertEqual(series.counts["same_kind_latest_replacement"], 1)

    def test_exact_pair_count_and_duration_coverage_are_distinct(self):
        pair, duration = SwingWindow("pairs", 2), SwingWindow("duration", 3*3600)
        series = SwingSeries(HOUR, (pair, duration))
        seq = points(6)
        series.swings = deque(seq)
        series.observed_since = START
        at = seq[-1].confirmed_at
        self.assertEqual(series.context(pair, at), seq[-4:])
        self.assertEqual(series.context(duration, at), seq[-3:])
        series.observed_since = at - HOUR
        self.assertIsNone(series.context(duration, at))
        self.assertIsNone(series.context(SwingWindow("pairs", 4), at))

    def test_simultaneous_opposite_quote_reversals_are_not_given_an_invented_order(self):
        series = SwingSeries(HOUR, (SwingWindow("pairs", 1),))
        series.append(bar(0), bar(0, 102, 104, 99, 100))
        self.assertIsNone(series.append(bar(1, 102, 104, 99, 100), bar(1)))
        self.assertEqual(series.counts["ambiguous_simultaneous_reversals"], 1)
        self.assertFalse(series.swings)
        self.assertEqual([s.kind for s in series.last_candidates],['high','low'])

    def test_gap_is_not_a_two_candle_reversal_and_dojis_do_not_reverse(self):
        series = SwingSeries(HOUR, (SwingWindow("pairs", 1),))
        series.append(bar(0), plus(bar(0)))
        self.assertIsNone(series.append(bar(50, 102, 104, 99, 100), plus(bar(50, 102, 104, 99, 100))))
        self.assertIsNone(series.append(bar(51, 100, 104, 99, 100), plus(bar(51, 100, 104, 99, 100))))

    def test_features_capture_real_time_and_reject_future_non_alternating_inputs(self):
        seq = points()
        values = swing_features(seq, HOUR, seq[-1].confirmed_at)
        self.assertEqual(values.shape, (4, 6))
        self.assertEqual(values[0, 0], 0.)
        np.testing.assert_array_equal(values[:, 2], [1, -1, 1, -1])
        with self.assertRaises(ValueError):
            swing_features(seq, HOUR, seq[-1].confirmed_at-HOUR)
        with self.assertRaises(ValueError):
            swing_features((seq[0], replace(seq[1], kind="high")), HOUR, seq[-1].confirmed_at)
        delayed = (*seq[:-1], replace(seq[-1], pivot_at=seq[-1].pivot_at+48*HOUR,
                                    pair_start=seq[-1].pair_start+48*HOUR, confirmed_at=seq[-1].confirmed_at+48*HOUR))
        self.assertGreater(swing_features(delayed, HOUR, delayed[-1].confirmed_at)[-1, 4], values[-1, 4])


class SwingSimulationTests(unittest.TestCase):
    def simulate(self, quotes, swing=None, accelerated=True, config=None):
        db = memory_db()
        self.addCleanup(db.close)
        swing = swing or SwingPoint("low", 100., START, START, START+2*HOUR)
        add_event(db, swing)
        sim = OpportunitySimulator(db, config or OpportunityConfig(), accelerated)
        sim.add("event", "EURUSD", swing)
        sim.process(np.asarray(quotes, dtype=np.float64))
        return db, sim

    def test_fast_and_every_tick_paths_match_on_buy_sell_multistep_trailing_and_gap_stop(self):
        t = milliseconds(START+2*HOUR)
        quotes = np.asarray([[t+1,101,101.1], [t+2,101.5,101.6], [t+3,102.2,102.3],
                             [t+4,103.5,103.6], [t+5,105.7,105.8], [t+6,103.9,104.0]])
        for selling in (False, True):
            data = quotes if not selling else np.column_stack((quotes[:, 0], 300-quotes[:, 2], 300-quotes[:, 1]))
            swing = SwingPoint("high" if selling else "low", 200. if selling else 100., START, START, START+2*HOUR)
            fast, _ = self.simulate(data, swing)
            slow, _ = self.simulate(data, swing, accelerated=False)
            self.assertEqual(dict(fast.execute("SELECT * FROM swing_events").fetchone()), dict(slow.execute("SELECT * FROM swing_events").fetchone()))
            self.assertEqual(fast.execute("SELECT close_reason FROM swing_events").fetchone()[0], "TRAILING_STOP")

    def test_random_paths_match_reference_exactly_for_overlapping_opportunities(self):
        rng = np.random.default_rng(7)
        t = milliseconds(START+2*HOUR)
        for selling in (False, True):
            bids = 101 + np.r_[0., np.cumsum(rng.normal(0, .12, 3000))]
            quotes = np.column_stack((t+np.arange(len(bids))+1, bids, bids+.1))
            swing = SwingPoint("low", 100., START, START, START+2*HOUR)
            if selling:
                quotes = np.column_stack((quotes[:, 0],300-quotes[:, 2],300-quotes[:, 1]))
                swing = replace(swing, kind="high", price=200.)
            fast, _ = self.simulate(quotes, swing)
            slow, _ = self.simulate(quotes, swing, accelerated=False)
            self.assertEqual(dict(fast.execute("SELECT * FROM swing_events").fetchone()),dict(slow.execute("SELECT * FROM swing_events").fetchone()))

    def test_two_bars_expire_only_unfilled_order_and_never_a_filled_position(self):
        t = milliseconds(START+2*HOUR)
        db, sim = self.simulate([[t+1,101,101.1], [t+3*3600000,102.3,102.4], [t+5*3600000,101.0,101.1]])
        row = db.execute("SELECT * FROM swing_events").fetchone()
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["closed_at"], instant(t+5*3600000).isoformat())
        self.assertEqual(row["close_reason"], "TRAILING_STOP")
        expired, _ = self.simulate([[t+2*3600000,101,101.1]])
        self.assertEqual(expired.execute("SELECT status,pnl_r FROM swing_events").fetchone()[:], ("expired",None))

    def test_entry_uses_ask_for_buy_bid_for_sell_and_exact_swing_stop(self):
        t = milliseconds(START+2*HOUR)
        db, _ = self.simulate([[t+1,101,101.1]])
        row = db.execute("SELECT entry_price,stop_price,initial_risk FROM swing_events").fetchone()
        self.assertEqual(row[:2], (101.1,100.))
        self.assertAlmostEqual(row[2],1.1)
        sell, _ = self.simulate([[t+1,101,101.1]], SwingPoint("high",103.,START,START,START+2*HOUR))
        self.assertEqual(sell.execute("SELECT entry_price,stop_price FROM swing_events").fetchone()[:],(101.,103.))

    def test_open_and_pending_at_end_have_no_fabricated_label_and_fill_requires_newer_tick(self):
        t = milliseconds(START+2*HOUR)
        db, sim = self.simulate([[t,101,101.1]])
        sim.finish(START+3*HOUR)
        self.assertEqual(db.execute("SELECT status,pnl FROM swing_events").fetchone()[:],("pending_at_end",None))
        db, sim = self.simulate([[t+1,101,101.1]])
        sim.finish(START+30*HOUR)
        self.assertEqual(db.execute("SELECT status,pnl FROM swing_events").fetchone()[:],("open",None))

    def test_stop_already_crossed_executes_and_records_immediate_loss_without_risk_veto(self):
        t = milliseconds(START+2*HOUR)
        db, _ = self.simulate([[t+1,99.9,100.1]])
        row=db.execute("SELECT status,pnl_r,opened_at,closed_at FROM swing_events").fetchone()
        self.assertEqual(row['status'],'closed')
        self.assertAlmostEqual(row['pnl_r'],-2.)
        self.assertEqual(row['opened_at'],row['closed_at'])

    def test_zero_risk_distance_still_records_trade_and_pnl_without_inventing_r(self):
        t=milliseconds(START+2*HOUR)
        db,_=self.simulate([[t+1,99.9,100.]])
        row=db.execute('SELECT status,pnl,pnl_r FROM swing_events').fetchone()
        self.assertEqual(row['status'],'closed')
        self.assertLess(row['pnl'],0.)
        self.assertIsNone(row['pnl_r'])

    def test_buffer_moves_both_stops_outward_and_recomputes_initial_risk(self):
        t=milliseconds(START+2*HOUR)
        config=OpportunityConfig(stop_buffer_pips=2.)
        db,_=self.simulate([[t+1,101.,101.1]],config=config)
        row=db.execute('SELECT pivot_price,stop_price,initial_risk FROM swing_events').fetchone()
        self.assertEqual(row['pivot_price'],100.)
        self.assertAlmostEqual(row['stop_price'],99.9998)
        self.assertAlmostEqual(row['initial_risk'],1.1002)
        db,_=self.simulate([[t+1,101.,101.1]],SwingPoint('high',103.,START,START,START+2*HOUR),config=config)
        self.assertAlmostEqual(db.execute('SELECT stop_price FROM swing_events').fetchone()[0],103.0002)
        self.assertAlmostEqual(config.stop_for(SwingPoint('low',150.,START,START,START+HOUR),'USDJPY'),149.98)

    def test_overlapping_opportunities_cross_daily_batches_without_position_limits(self):
        t=milliseconds(START+2*HOUR)
        databases=[]
        for accelerated in (True,False):
            db=memory_db()
            self.addCleanup(db.close)
            sim=OpportunitySimulator(db,OpportunityConfig(),accelerated)
            for i in range(3):
                swing=SwingPoint('low',100.-i*.1,START,START,START+2*HOUR)
                add_event(db,swing,f'event{i}')
                sim.add(f'event{i}','EURUSD',swing)
            sim.process(np.asarray([[t+1,101.,101.1],[t+2,101.4,101.5]]))
            self.assertEqual(len(sim.positions),3)
            sim.process(np.asarray([[t+86400000,103.,103.1],[t+86400001,100.9,101.]]))
            self.assertFalse(sim.positions)
            databases.append(db)
        a,b=([dict(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')] for db in databases)
        self.assertEqual(a,b)

    def test_extreme_spread_and_virtual_quantity_do_not_trigger_any_risk_gate(self):
        t=milliseconds(START+2*HOUR)
        db,_=self.simulate([[t+1,101.,150.]],config=OpportunityConfig(risk_cash=1e12))
        row=db.execute('SELECT status,entry_price,quantity FROM swing_events').fetchone()
        self.assertEqual(row['status'],'open')
        self.assertEqual(row['entry_price'],150.)
        self.assertEqual(row['quantity'],1e12/50.)


def create_training_database(path, days=80, dates=None):
    db = connect(path)
    windows = (SwingWindow("pairs",2),SwingWindow("duration",43200))
    parameters = {"windows":[{"mode":w.mode,"value":w.value} for w in windows],"config":{"interval_seconds":3600},
                  "source_fingerprint":"synthetic", "feature_schema":FEATURE_SCHEMA,"swing_schema":SWING_SCHEMA,
                  "opportunity_schema":OPPORTUNITY_SCHEMA,"risk_management_filters":False}
    db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('run',?,'completed',?)",
               (START.isoformat(),json.dumps(parameters)))
    for day in range(days):
        end=(dates[day] if dates is not None else START+timedelta(days=day))+12*HOUR
        seq = points(4,end)
        swing = seq[-1]
        event = f"event{day}"
        add_event(db,swing,event)
        pnl_r = (-1.,0.,2.)[day%3]
        db.execute("UPDATE swing_events SET status='closed',opened_at=?,closed_at=?,pnl=?,pnl_r=? WHERE event_id=?",
                   ((end+timedelta(seconds=1)).isoformat(),(end+3*HOUR).isoformat(),100*pnl_r,pnl_r,event))
        for w in windows:
            selected = seq if w.mode=="pairs" else points(2+2*(day%3),end)
            values = swing_features(selected,HOUR,end)
            db.execute("""INSERT INTO swing_contexts(event_id,window_key,information_start,context_end,n_swings,
                          span_seconds,features,swings_json) VALUES(?,?,?,?,?,?,?,?)""",
                       (event,w.key,min(s.pair_start for s in selected).isoformat(),end.isoformat(),len(selected),
                        (selected[-1].confirmed_at-selected[0].confirmed_at).total_seconds(),values.astype('<f4').tobytes(),
                        json.dumps([s.to_dict() for s in selected])))
    db.commit()
    db.close()


class SwingLearningTests(unittest.TestCase):
    def test_fixed_window_keeps_pooled_symbols_and_ignores_other_windows(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            create_training_database(path)
            db = connect(path)
            db.execute("UPDATE swing_events SET symbol='GBPUSD' WHERE CAST(substr(event_id,6) AS INTEGER)%2=1")
            # An unselected window must neither reduce the cohort nor feed TS2Vec.
            db.execute("DELETE FROM swing_contexts WHERE window_key='duration_43200' AND context_id%4=0")
            db.execute("UPDATE swing_contexts SET features=X'00' WHERE window_key='duration_43200'")
            db.commit()
            db.close()
            config = ResolverConfig(min_encoder_samples=8,min_train_samples=15,min_validation_samples=8,
                                    min_accepted_validation=3,n_estimators=5,early_stopping_rounds=2,threads=1)
            encoder = SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            result = train(path,output_dir=Path(tmp)/'models',config=config,encoder_config=encoder,protocol=PROTOCOL,
                           selection=TrainingSelection(interval_seconds=3600,pairs=(2,)))
            self.assertEqual(list(result['windows']),['pairs_2'])
            self.assertEqual(result['scope']['symbols'],['EURUSD','GBPUSD'])
            self.assertEqual(result['split']['after_purge'],{'train':46,'validation':18})
            db = connect(path)
            rows = db.execute("""SELECT DISTINCT e.symbol,c.window_key FROM swing_assignments a
                JOIN swing_contexts c USING(context_id) JOIN swing_events e USING(event_id)""").fetchall()
            self.assertEqual({tuple(r) for r in rows},{('EURUSD','pairs_2'),('GBPUSD','pairs_2')})
            db.close()
            resolver = SwingResolver(result['selected_directory'],threads=1)
            seq = points(4,START+timedelta(days=70,hours=12))
            for symbol in ('EURUSD','GBPUSD'):
                self.assertTrue(np.isfinite(resolver.predict(seq,as_of=seq[-1].confirmed_at,symbol=symbol)['prediction']))
            with self.assertRaisesRegex(ValueError,'symbol'):
                resolver.predict(seq,as_of=seq[-1].confirmed_at,symbol='USDJPY')
            with self.assertRaisesRegex(ValueError,'couples'):
                resolver.predict(points(6,seq[-1].confirmed_at),as_of=seq[-1].confirmed_at,symbol='EURUSD')
            tested = frozen_test(path,directory=result['directory'],run_id='run',start='2024-03-05',end='2024-03-20',min_samples=8,threads=1)
            self.assertEqual(tested['labeled_contexts'],16)
            self.assertEqual(tested['scope']['window_keys'],['pairs_2'])

    def test_optional_symbol_subset_filters_encoder_labels_and_frozen_test(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            create_training_database(path)
            db = connect(path)
            db.execute("UPDATE swing_events SET symbol='GBPUSD' WHERE CAST(substr(event_id,6) AS INTEGER)%2=1")
            db.execute("UPDATE swing_contexts SET features=X'00' WHERE event_id IN (SELECT event_id FROM swing_events WHERE symbol='GBPUSD')")
            db.commit()
            db.close()
            config = ResolverConfig(min_encoder_samples=8,min_train_samples=15,min_validation_samples=5,
                                    min_accepted_validation=3,n_estimators=5,early_stopping_rounds=2,threads=1)
            encoder = SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            result = train(path,output_dir=Path(tmp)/'models',config=config,encoder_config=encoder,protocol=PROTOCOL,
                           selection=TrainingSelection(symbols=('eurusd',),durations=(43200,)))
            self.assertEqual(list(result['windows']),['duration_43200'])
            self.assertEqual(result['split']['after_purge'],{'train':23,'validation':9})
            self.assertEqual(result['windows']['duration_43200']['encoder_train_contexts'],23)
            tested = frozen_test(path,directory=result['directory'],run_id='run',start='2024-03-05',end='2024-03-20',min_samples=5,threads=1)
            self.assertEqual(tested['labeled_contexts'],8)
            db = connect(path)
            self.assertEqual(db.execute("""SELECT count(*) FROM swing_embeddings v JOIN swing_contexts c USING(context_id)
                JOIN swing_events e USING(event_id) WHERE e.symbol='GBPUSD'""").fetchone()[0],0)
            db.execute("UPDATE swing_events SET symbol='USDJPY' WHERE symbol='EURUSD'")
            db.commit()
            db.close()
            with self.assertRaisesRegex(ValueError,'symbols'):
                frozen_test(path,directory=result['directory'],run_id='run',start='2024-03-05',end='2024-03-20',min_samples=5,threads=1)

    def test_selection_rejects_uncollected_interval_or_window_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.sqlite3'
            create_training_database(path)
            for selection in (TrainingSelection(interval_seconds=14400),TrainingSelection(pairs=(8,)),
                              TrainingSelection(durations=(86400,)),TrainingSelection(symbols=('USDJPY',))):
                with self.subTest(selection=selection),self.assertRaises(ValueError):
                    train(path,run_id='run',output_dir=Path(tmp)/'models',selection=selection)
            self.assertFalse((Path(tmp)/'models').exists())

    def test_automatic_run_selection_matches_interval_and_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.sqlite3'
            create_training_database(path)
            db = connect(path)
            parameters = json.loads(db.execute("SELECT parameters_json FROM swing_runs").fetchone()[0])
            parameters['config']['interval_seconds'] = 14400
            parameters['symbols'] = ['EURUSD','GBPUSD']
            db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('h4','2025-01-01','completed',?)",(json.dumps(parameters),))
            db.commit()
            self.assertEqual(select_collection(db,None,TrainingSelection(interval_seconds=3600,pairs=(2,)))[0]['run_id'],'run')
            self.assertEqual(select_collection(db,None,TrainingSelection(interval_seconds=14400))[0]['run_id'],'h4')
            db.close()

    def test_default_year_protocol_never_reads_2026_features_or_outcomes_during_training(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            path=Path(tmp)/'data.sqlite3'
            dates=[datetime(year,1,1,tzinfo=timezone.utc)+timedelta(days=i) for year in range(2020,2026) for i in range(10)]
            dates.extend(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=i) for i in range(20))
            create_training_database(path,dates=dates)
            db=connect(path)
            # Deliberately unusable future inputs prove that train does not load them.
            db.execute("UPDATE swing_contexts SET features=X'00' WHERE context_end>='2026-01-01'")
            db.execute("UPDATE swing_events SET pnl_r=999999 WHERE confirmed_at>='2026-01-01'")
            db.commit()
            db.close()
            config=ResolverConfig(min_encoder_samples=8,min_train_samples=15,min_validation_samples=8,
                                  min_accepted_validation=3,n_estimators=5,early_stopping_rounds=2,threads=1)
            encoder=SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            result=train(path,output_dir=Path(tmp)/'models',config=config,encoder_config=encoder)
            self.assertFalse(result['test_evaluated'])
            db=connect(path)
            phases={}
            for phase,year in db.execute("SELECT DISTINCT a.phase,substr(c.context_end,1,4) FROM swing_assignments a JOIN swing_contexts c USING(context_id)"):
                phases.setdefault(phase,set()).add(year)
            self.assertEqual(phases,{'train':{'2020','2021','2022','2023'},'validation':{'2024','2025'}})
            self.assertEqual(db.execute("SELECT count(*) FROM swing_embeddings v JOIN swing_contexts c USING(context_id) WHERE c.context_end>='2026-01-01'").fetchone()[0],0)
            db.close()
            resolver=SwingResolver(result['selected_directory'],threads=1)
            self.assertEqual(resolver.encoder.metadata['available_after'],'2024-01-01T00:00:00+00:00')
            self.assertEqual(resolver.metadata['available_after'],'2026-01-01T00:00:00+00:00')

    def test_calendar_periods_exclude_labels_closing_in_the_following_period(self):
        cfg = ResolverConfig(min_train_samples=1,min_validation_samples=1)
        protocol=ResearchProtocol('2024-01-01','2024-01-02','2024-01-03','2024-01-04')
        def row(day,close):
            return {"confirmed_at":(START+timedelta(days=day,hours=12)).isoformat(),
                    "closed_at":(START+timedelta(days=close,hours=15)).isoformat()}
        rows = [row(0,0),row(1,2),row(2,2),row(3,4),row(4,4)]
        splits,info = purged_split(rows,protocol,cfg)
        self.assertEqual(splits['train'],rows[:1])
        self.assertEqual(splits['validation'],rows[2:3])
        self.assertEqual(info['after_purge'],{'train':1,'validation':1})

    def test_full_ts2vec_xgboost_two_window_training_reload_and_temporal_guard(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path=Path(tmp)/'data.sqlite3'
            create_training_database(path)
            config=ResolverConfig(min_encoder_samples=8,min_train_samples=15,min_validation_samples=8,
                                  min_accepted_validation=3,n_estimators=8,early_stopping_rounds=2,threads=1)
            encoder=SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            result=train(path,output_dir=Path(tmp)/'models',config=config,encoder_config=encoder,protocol=PROTOCOL)
            self.assertEqual(set(result['windows']),{'pairs_2','duration_43200'})
            selected=SwingResolver(result['selected_directory'],threads=1)
            self.assertEqual(selected.metadata['scope']['window_keys'],[selected.metadata['window_key']])
            self.assertEqual(selected.encoder.metadata['provenance']['scope']['window_keys'],[selected.metadata['window_key']])
            day=70
            count=4 if selected.metadata['window_key']=='pairs_2' else 2+2*(day%3)
            seq=points(count,START+timedelta(days=day,hours=12))
            scored=selected.predict(seq,as_of=seq[-1].confirmed_at)
            self.assertTrue(np.isfinite(scored['prediction']))
            self.assertIn('accepted',scored)
            with self.assertRaises(ValueError):
                selected.predict(points(4),as_of=START+12*HOUR)
            db=connect(path)
            models=db.execute('SELECT count(*) FROM swing_resolvers').fetchone()[0]
            self.assertEqual(models,2)
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(db.execute("SELECT count(*) FROM swing_predictions WHERE phase='test'").fetchone()[0],0)
            self.assertGreater(db.execute('SELECT count(*) FROM swing_predictions').fetchone()[0],0)
            self.assertGreater(db.execute('SELECT count(*) FROM swing_embeddings').fetchone()[0],0)
            model_id=db.execute('SELECT model_id FROM swing_embedding_models LIMIT 1').fetchone()[0]
            db.close()
            with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb',side_effect=AssertionError('No fit during test')):
                tested=frozen_test(path,directory=Path(result['directory']),run_id='run',start='2024-03-05',end='2024-03-20',min_samples=8,threads=1)
            self.assertFalse(tested['fit_called'])
            self.assertEqual(tested['labeled_contexts'],16)
            exported=Path(tmp)/'export.csv'
            export(path,'run',exported,model_id)
            self.assertIn('embedding_003',exported.read_text().splitlines()[0])

    def test_encoder_exact_length_inference_reload_reproducibility_and_no_future_fit(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path=Path(tmp)/'encoder.pt'
            config=SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            arrays=[swing_features(points(n),HOUR,START+12*HOUR) for n in (2,4,6)]
            service=SwingEncoder(config).fit(arrays,observed_at=[START+12*HOUR]*3,cutoff=START+13*HOUR,checkpoint=path,provenance={})
            vector=service.encode(arrays,observed_at=[START+14*HOUR]*3)
            loaded=SwingEncoder.load(path,threads=1)
            np.testing.assert_array_equal(vector,loaded.encode(arrays,observed_at=[START+14*HOUR]*3))
            np.testing.assert_allclose(vector[0],service.encode(arrays[:1],observed_at=[START+14*HOUR])[0])
            with self.assertRaises(ValueError):
                service.encode(arrays,observed_at=[START]*3)
            with self.assertRaises(ValueError):
                SwingEncoder(config).fit(arrays,observed_at=[START+14*HOUR]*3,cutoff=START,checkpoint=Path(tmp)/'bad.pt',provenance={})

    def test_binary_resolver_trains_and_reloads_without_future_outcomes_as_features(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            path=Path(tmp)/'data.sqlite3'
            create_training_database(path)
            config=ResolverConfig(target='win',min_encoder_samples=8,min_train_samples=15,min_validation_samples=8,
                                  min_accepted_validation=3,n_estimators=5,early_stopping_rounds=2,threads=1)
            encoder=SwingEncoderConfig(output_dims=4,hidden_dims=4,depth=1,train_iters=2,threads=1)
            result=train(path,output_dir=Path(tmp)/'models',config=config,encoder_config=encoder,protocol=PROTOCOL)
            resolver=SwingResolver(result['selected_directory'],threads=1)
            count=4 if resolver.metadata['window_key']=='pairs_2' else 2+2*(70%3)
            seq=points(count,START+timedelta(days=70,hours=12))
            scored=resolver.predict(seq,as_of=seq[-1].confirmed_at)
            self.assertTrue(0 <= scored['prediction'] <= 1)
            self.assertFalse(any('pnl' in name or 'exit' in name or 'closed' in name for name in resolver.metadata['feature_names']))


class SwingCliTests(unittest.TestCase):
    def test_train_selectors_choose_only_explicit_windows_and_allow_symbol_pooling(self):
        for options, expected in ((['--durations','12h'],['duration_43200']),
                                  (['--pairs','2'],['pairs_2']),
                                  (['--window-mode','duration'],['duration_43200']),
                                  ([],['duration_43200','pairs_2'])):
            selection = training_selection(parse_args(['train','--candle-interval','60m',*options]))
            self.assertEqual([w.key for w in selection.windows({'windows':[{'mode':'pairs','value':2},{'mode':'duration','value':43200}]})],expected)
        selection = training_selection(parse_args(['train','--symbols','EURUSD','GBPUSD','--pairs','2','4']))
        self.assertEqual(selection.symbols,('EURUSD','GBPUSD'))
        self.assertIsNone(training_selection(parse_args(['train'])).symbols)
        for mode in ('duration','pairs'):
            training_selection(parse_args(['run','--window-mode',mode,'--dry-run']))
        for options in (['--window-mode','duration','--pairs','8'],['--pairs','0'],['--candle-interval','7h']):
            with self.subTest(options=options),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(['train',*options])

    def test_both_modes_and_order_only_expiration_default(self):
        args=parse_args(['collect','--durations','6h','1d','--pairs','2','5'])
        self.assertEqual([w.key for w in windows_for(args)],['duration_21600','duration_86400','pairs_2','pairs_5'])
        self.assertEqual(args.order_expiration_bars,2)
        self.assertEqual(parse_args(['train']).target,'pnl-r')
        self.assertEqual(parse_args(['train','--target','win']).target,'win')
        defaults=parse_args(['train'])
        self.assertEqual((defaults.train_start,defaults.train_end,defaults.validation_start,defaults.validation_end),
                         ('2020-01-01','2023-12-31','2024-01-01','2025-12-31'))
        self.assertEqual(parse_args(['test','--resolver','frozen']).test_start,'2026-01-01')

    def test_bad_window_or_expiration_is_rejected(self):
        for options in (['--pairs','0'],['--order-expiration-bars','0'],['--trailing-start-r','nan'],['--candle-interval','7h']):
            with self.subTest(options=options),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(['collect',*options])


class SwingCollectionTests(unittest.TestCase):
    def test_raw_tick_collection_persists_frozen_contexts_and_independent_labels(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            folder=root/'history'/'EURUSD'/'2024'/'01'
            folder.mkdir(parents=True)
            path=folder/'2024-01-01.tsv'
            lines=[]
            for h in range(12):
                o,c=(101.,102.) if h%2==0 else (102.,101.)
                for offset,price in ((1,o),(1000,103.),(2000,100.),(3599,c)):
                    ms=milliseconds(START+h*HOUR+timedelta(seconds=offset))
                    lines.append(f'QUOTE\tEURUSD\t{ms}\t1\t1\t{price}\t{price+.1}\t1\t1')
            path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            database=root/'research.sqlite3'
            result=collect(database,history=root/'history',symbols=['EURUSD'],start=START.date(),end=START.date(),
                           windows=(SwingWindow('pairs',1),SwingWindow('duration',3*3600)), store_contexts=True)
            self.assertEqual(sum(result['statuses'].values()),11)
            self.assertEqual(result['contexts']['pairs_1'],10)
            db=connect(database)
            row=db.execute('SELECT swings_json FROM swing_contexts ORDER BY context_id LIMIT 1').fetchone()
            kinds=[s['kind'] for s in json.loads(row[0])]
            self.assertEqual(kinds,['high','low'])
            self.assertEqual(db.execute("SELECT count(*) FROM swing_events WHERE status IN ('open','pending_at_end') AND pnl_r IS NOT NULL").fetchone()[0],0)
            db.close()
            bad=folder/'2024-01-02.tsv'
            bad.write_text(f'QUOTE\tEURUSD\t{milliseconds(START+timedelta(days=1))}\t1\t1\t102\t101\t1\t1\n')
            with self.assertRaises(ValueError):
                read_day(bad,'EURUSD')


if __name__=='__main__':
    unittest.main()
