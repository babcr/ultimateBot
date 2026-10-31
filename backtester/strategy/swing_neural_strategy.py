"""Neural grid predictions with the existing causal swing execution and risk rules."""
from ultimateBot.backtester.strategy.swing_grid_strategy import SwingGridStrategy


class SwingNeuralStrategy(SwingGridStrategy):
    strategy_name = 'swing-nn'

    def __init__(self, directory, *, min_threshold=None, **kwargs):
        if min_threshold is None:
            from ultimateBot.backtester.strategy.swing_ts2vec_strategy import model_configuration
            min_threshold = model_configuration(directory)[1]['min_threshold']
        super().__init__(directory, min_threshold=min_threshold, **kwargs)

    def _load_resolver(self, threads):
        from ultimateBot.backtester.research.swing_neural import SwingNeuralResolver
        return SwingNeuralResolver(self.directory, threads=threads)
