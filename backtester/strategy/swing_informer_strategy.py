"""Use direct Informer gain predictions with the shared causal swing execution."""
from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy


class SwingInformerStrategy(SwingTS2VecStrategy):
    strategy_name = 'swing-informer'

    def _load_resolver(self, threads):
        from ultimateBot.backtester.research.swing_informer import SwingInformerResolver
        return SwingInformerResolver(self.directory, threads=threads)
