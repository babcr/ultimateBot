"""Direct swing grids in XGBoost, with the shared causal swing execution."""
from collections import defaultdict, deque
from datetime import timedelta

from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy


class SwingGridStrategy(SwingTS2VecStrategy):
    strategy_name = 'swing-grid'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalization_candles = defaultdict(deque)

    def _load_resolver(self, threads):
        from ultimateBot.backtester.research.swing_grid import SwingGridResolver
        return SwingGridResolver(self.directory, threads=threads)

    def _observe_closed_candle(self, symbol, bid, ask):
        if self.metadata.get('grid_normalization', 'swing-mean') != 'candle-close-mean':
            return
        candles = self.normalization_candles[symbol]
        candles.append((bid, ask))
        left = bid.start_time+self.analysis_interval-timedelta(seconds=self.window.value)
        while candles and candles[0][0].start_time < left:
            candles.popleft()

    def _predict_swing(self, points, *, as_of, symbol, stop_buffer_pips=None):
        extra = {} if stop_buffer_pips is None else dict(stop_buffer_pips=stop_buffer_pips)
        if self.metadata.get('grid_normalization', 'swing-mean') == 'candle-close-mean':
            return self.resolver.predict(points, as_of=as_of, symbol=symbol,
                candles=self.normalization_candles[symbol], **extra)
        return super()._predict_swing(points, as_of=as_of, symbol=symbol, **extra)
