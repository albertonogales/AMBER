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
                 metric='combined',
                 random_seed=None,
                 validation_data=None):
        """
        :param data: (n_samples, n_features) training array
        :param period: number of training iterations per map
        :param initial_lr: initial learning rate
        :param size_range: map sizes to try; defaults to ±2 around Vesanto size
        :param give_best: if True, self.best_map holds the map with the best score
        :param metric: model selection criterion — 'qe' (quantization error), 'te'
            (topological error), or 'combined' (normalised mean of both). 'qe' alone
            biases towards smaller maps; 'combined' is recommended.
        :param random_seed: base seed; each map gets random_seed+i for independence
        :param validation_data: held-out data for model selection; None = use training data
        """
        if metric not in ('qe', 'te', 'combined'):
            raise ValueError(f"metric must be 'qe', 'te', or 'combined', got '{metric}'.")

        if size_range is None:
            recommended = vesanto_size(data.shape[0])
            size_range = range(max(2, recommended - 2), recommended + 3)

        self.maps = {}
        self.best_map = None
        self._metric = metric

        if give_best:
            from .classification import Classification
            if validation_data is None:
                logger.warning(
                    "IterativeSOM: model selection is evaluating on training data "
                    "(validation_data=None). Pass validation_data= to avoid "
                    "in-sample selection bias."
                )
            eval_data = validation_data if validation_data is not None else data

            scores = {}
            for i, size in enumerate(size_range):
                seed = random_seed + i if random_seed is not None else None
                m = Map(data=data, size=size, period=period, initial_lr=initial_lr,
                        random_seed=seed)
                self.maps[size] = m
                c = Classification(m, eval_data)
                scores[size] = (c.quantization_error, c.topological_error)

            self.best_map = self._select_best(scores)
        else:
            for i, size in enumerate(size_range):
                seed = random_seed + i if random_seed is not None else None
                m = Map(data=data, size=size, period=period, initial_lr=initial_lr,
                        random_seed=seed)
                self.maps[size] = m

    def _select_best(self, scores: dict) -> 'Map':
        """Select the best map from a {size: (qe, te)} dict using self._metric."""
        qes = np.array([scores[s][0] for s in scores])
        tes = np.array([scores[s][1] for s in scores])

        if self._metric == 'qe':
            values = qes
        elif self._metric == 'te':
            values = tes
        else:  # combined: normalise each metric to [0,1] then average
            qe_norm = (qes - qes.min()) / (np.ptp(qes) + 1e-12)
            te_norm = (tes - tes.min()) / (np.ptp(tes) + 1e-12)
            values = 0.5 * qe_norm + 0.5 * te_norm

        best_size = list(scores.keys())[int(np.argmin(values))]
        logger.info(
            "IterativeSOM: best size=%d  QE=%.4f  TE=%.4f  (metric='%s')",
            best_size, scores[best_size][0], scores[best_size][1], self._metric,
        )
        return self.maps[best_size]

    @staticmethod
    def calculate_range(data, min_size=2, max_size=None):
        """Returns a range of map sizes centred on the Vesanto recommendation."""
        recommended = vesanto_size(data.shape[0])
        lo = max(min_size, recommended - 2)
        hi = recommended + 2 if max_size is None else max_size
        return range(lo, hi + 1)
