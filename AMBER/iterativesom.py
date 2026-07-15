import logging

import numpy as np

from .map import Map, vesanto_size

logger = logging.getLogger(__name__)


class IterativeSOM:
    """Trains multiple SOMs across a range of map sizes and optionally returns the best one."""

    def __init__(self,
                 data,
                 period,
                 initial_lr,
                 size_range=None,
                 give_best=False,
                 random_seed=None,
                 validation_data=None):
        """
        :param data: (n_samples, n_features) training array
        :param period: number of training iterations per map
        :param initial_lr: initial learning rate
        :param size_range: map sizes to try; defaults to ±2 around Vesanto size
        :param give_best: if True, self.best_map holds the map with lowest QE
        :param random_seed: base seed; each map gets random_seed+i for independence
        :param validation_data: held-out data for model selection; None = use training data
        """
        if size_range is None:
            recommended = vesanto_size(data.shape[0])
            size_range = range(max(2, recommended - 2), recommended + 3)

        self.maps = {}
        best_qe = np.inf
        self.best_map = None

        for i, size in enumerate(size_range):
            seed = random_seed + i if random_seed is not None else None
            m = Map(data=data, size=size, period=period, initial_lr=initial_lr,
                    random_seed=seed)
            self.maps[size] = m

            if give_best:
                from .classification import Classification
                if validation_data is None:
                    logger.warning(
                        "IterativeSOM: model selection is evaluating QE on training data "
                        "(validation_data=None). Pass validation_data= to avoid "
                        "in-sample selection bias."
                    )
                eval_data = validation_data if validation_data is not None else data
                c = Classification(m, eval_data)
                if c.quantization_error < best_qe:
                    best_qe = c.quantization_error
                    self.best_map = m

    @staticmethod
    def calculate_range(data, min_size=2, max_size=None):
        """Returns a range of map sizes centred on the Vesanto recommendation."""
        recommended = vesanto_size(data.shape[0])
        lo = max(min_size, recommended - 2)
        hi = recommended + 2 if max_size is None else min(max_size, recommended + 2)
        return range(lo, hi + 1)
