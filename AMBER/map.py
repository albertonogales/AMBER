from __future__ import annotations

import json
import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distances import (
    AVAILABLE_DISTANCES,
    GRID_DISTANCE,
    SIGNAL_DISTANCE_MATRIX,
    _DTW_WARN_THRESHOLD,
)

logger = logging.getLogger(__name__)


def vesanto_size(n_samples: int) -> int:
    """Map side recommended by Vesanto & Alhoniemi (2000): √(5·√N), min 2."""
    return max(2, round(np.sqrt(5.0 * np.sqrt(n_samples))))

class Map:
    """Kohonen SOM with configurable distance, neighbourhood, normalisation, and weight init.

    Training duration is controlled by ``period``, which counts **total weight update
    steps** (one pattern presented = one step), not epochs.  To run K epochs over a
    dataset of N samples pass ``period = K * N``.  The training log always prints the
    step count and its epoch equivalent so results are unambiguous.
    """

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 size: Optional[int] = None,
                 period: int = 10,
                 initial_lr: float = 0.1,
                 initial_neighbourhood: int = 0,
                 distance: str = 'euclidean',
                 dtw_band: Optional[int] = None,
                 use_decay: bool = False,
                 lr_decay: str = 'linear',
                 normalization: str = 'none',
                 presentation: str = 'random',
                 weights: str = 'random',
                 random_seed: Optional[int] = None) -> None:

        """
        :param data: (n_samples, n_features) array; if given, train() is called immediately.
        :param size: map side (size×size neurons); None auto-selects via Vesanto heuristic.
        :param period: **Total weight update steps**, not epochs.
            One epoch over a dataset of N samples equals N update steps.
            To train for K epochs use ``period = K * N``.
            Example: 200 samples × 100 epochs → ``period=20000``.
            The default of 10 is intentionally tiny; always set this explicitly.
            A warning is logged when period < N (less than one full pass over the data).
        :param initial_lr: initial learning rate, must be in (0, 1).
        :param initial_neighbourhood: initial neighbourhood radius; defaults to size.
        :param distance: BMU search metric — 'euclidean', 'manhattan', 'chebyshev',
            'cosine', 'correlation', 'dtw', or 'cross_correlation'.
        :param dtw_band: Sakoe-Chiba half-width for DTW (None = unconstrained).
        :param use_decay: True = Gaussian neighbourhood; False = bubble (hard boundary).
        :param lr_decay: 'linear' (default) or 'asymptotic' decay schedule.
        :param normalization: 'none', 'zscore'/'fwn', 'robust', '01scale',
            'zscore_sample', 'robust_sample', 'minmax_sample', or 'l2'/'euclidean'.
        :param presentation: 'random' (default) or 'sequential'.
        :param weights: weight init — 'random', 'random_negative', 'sample', or 'PCA'.
        :param random_seed: integer for reproducible results; None = non-deterministic.
        """

        self.__trained = False
        if period <= 1:
            raise ValueError(f"'period' must be > 1, got {period}.")
        if not (0 < initial_lr < 1):
            raise ValueError(f"'initial_lr' must be in (0, 1), got {initial_lr}.")

        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        if size is None:
            if data is None:
                raise ValueError(
                    "Provide either 'size' or 'data' so the map size can be determined."
                )
            size = vesanto_size(data.shape[0])
            logger.info(
                f"Map size set automatically to {size}×{size} "
                f"({size**2} neurons) using Vesanto's heuristic "
                f"(N={data.shape[0]})."
            )

        if size < 2:
            raise ValueError(f"'size' must be >= 2, got {size}.")
        if distance not in AVAILABLE_DISTANCES:
            raise ValueError(
                f"Unknown distance '{distance}'. "
                f"Available options: {AVAILABLE_DISTANCES}"
            )

        self.map_size = size
        self.presentation = presentation
        self.initial_lr = initial_lr
        self.distance = distance
        self.dtw_band = dtw_band
        self.use_decay = use_decay
        self.lr_decay  = lr_decay
        self.num_data = 0
        self.input_data_dimension = 0
        self.period = period
        self.neighbourhood = initial_neighbourhood if initial_neighbourhood != 0 \
            else size
        self.normalization = normalization
        self.weights_init = weights

        self.weights: np.ndarray = np.ones(1)
        self._norm_params: dict = {}

        self.__ids_matrix: np.ndarray = self._build_ids_matrix()

        if data is not None:
            self.train(data)

    def _build_ids_matrix(self) -> np.ndarray:
        """Build the (rows, cols, 2) grid-coordinate index matrix."""
        ids: list[list[list[int]]] = []
        for y in range(self.map_size):
            row: list[list[int]] = []
            for x in range(self.map_size):
                row.append([y, x])
            ids.append(row)
        return np.array(ids)

    def train(self,
              data: np.ndarray) -> None:
        """Train the SOM on the provided data.

        :param data: 2-D numpy array — rows are samples, columns are features
        """
        self.num_data = data.shape[0]
        self.input_data_dimension = data.shape[1]
        training_data = self.__normalize(data, method=self.normalization)
        if not np.all(np.isfinite(training_data)):
            raise ValueError(
                "Training data contains non-finite values (inf or nan) after "
                "normalization. Check input data or feature extraction — e.g. "
                "sample_entropy returns inf for signals with no template matches."
            )
        self.weights = self.__init_weights(data=training_data, method=self.weights_init)

        if self.distance == 'dtw' and self.dtw_band is None:
            win = self.input_data_dimension
            if win > _DTW_WARN_THRESHOLD:
                total_pairs = self.period * self.map_size ** 2
                warnings.warn(
                    f"Training with distance='dtw', band=None, and window length "
                    f"{win}. Pure-Python DTW is O(N²) per neuron-pattern pair; this "
                    f"run will perform {total_pairs:,} DTW calls (~{total_pairs * win**2 / 1e9:.1f}B "
                    f"inner-loop iterations). Consider setting dtw_band (e.g. "
                    f"dtw_band={win // 10}) to limit cost to O(N·band).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        epochs_equiv = self.period / self.num_data
        logger.info(
            f"TRAINING: {self.period} steps over {self.num_data} samples "
            f"({epochs_equiv:.1f} equivalent epoch(s))."
        )
        if self.period < self.num_data:
            logger.warning(
                f"period={self.period} is less than the number of training samples "
                f"({self.num_data}), so the SOM will see fewer than one full pass "
                f"over the data. To train for K epochs set period = K * {self.num_data}."
            )
        for numPresentation in tqdm(range(1, self.period + 1)):
            if self.presentation == 'sequential':
                new_pattern = training_data[(numPresentation - 1) % self.num_data]
            else:
                new_pattern = training_data[self._rng.integers(0, self.num_data)]

            bmu = self.calculate_bmu(new_pattern)

            eta = self.variation_learning_rate(self.initial_lr, numPresentation,
                                               self.period,
                                               mode=self.lr_decay)
            v_final = 1 if self.use_decay else 0
            v = self.variation_neighbourhood(self.neighbourhood, numPresentation,
                                             self.period, v_final,
                                             mode=self.lr_decay)
            self.__adjust_weights(v, eta, bmu[1], new_pattern)

        self.__trained = True
        logger.info("FINISHED.")

    def reinforce(self, training_data: np.ndarray,
                  reinforcement: int = 0,
                  extension: int = 1,
                  compression: float = 0.5) -> None:
        """Fine-tuning phase: additional rounds with multiplied period and compressed lr.

        :param training_data: 2-D numpy array — rows are samples, columns are features
        :param reinforcement: number of additional rounds (0 = no-op)
        :param extension: period multiplier per round
        :param compression: learning-rate scale factor per round
        """
        norm_data = self.transform(training_data)
        n_reinforce = len(norm_data)
        origin_initial_lr = self.initial_lr
        # Fine-tuning neighbourhood decays to 1, then stays there across rounds.
        round_neighbourhood = self.neighbourhood
        for _round in range(reinforcement):

            self.period = int(self.period * extension)
            reinforcement_lr = origin_initial_lr * compression
            origin_initial_lr = reinforcement_lr

            for numPresentation in tqdm(range(1, self.period + 1)):
                if self.presentation == 'sequential':
                    new_pattern = norm_data[(numPresentation - 1) % n_reinforce]
                else:
                    new_pattern = norm_data[self._rng.integers(0, n_reinforce)]

                bmu = self.calculate_bmu(new_pattern)

                eta = self.variation_learning_rate(reinforcement_lr, numPresentation,
                                                   self.period, mode=self.lr_decay)
                v = self.variation_neighbourhood(round_neighbourhood, numPresentation,
                                                 self.period, final=1,
                                                 mode=self.lr_decay)
                self.__adjust_weights(v, eta, bmu[1], new_pattern)

            round_neighbourhood = 1

        self.__trained = True

    def calculate_bmu(self, pattern: np.ndarray) -> Tuple:
        """Return (bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos) for pattern."""
        dist_fn = SIGNAL_DISTANCE_MATRIX[self.distance]
        kwargs = {'band': self.dtw_band} if self.distance == 'dtw' else {}
        distances = dist_fn(self.weights, pattern, **kwargs)

        bmu_dist = np.min(distances)
        bmu_pos  = np.unravel_index(np.argmin(distances), distances.shape)

        distances[bmu_pos] = np.inf

        second_bmu_dist = np.min(distances)
        second_bmu_pos  = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos

    def _grid_distance(self, ids_matrix: np.ndarray, bmu_pos: Tuple) -> np.ndarray:
        """(rows, cols) grid-space distances to bmu_pos; always Euclidean or Chebyshev."""
        grid_dist_fn = GRID_DISTANCE.get(self.distance, GRID_DISTANCE['euclidean'])
        return grid_dist_fn(ids_matrix, bmu_pos)

    @staticmethod
    def variation_learning_rate(initial_lr: float, i: int, iterations_number: int,
                                mode: str = 'linear') -> float:
        """Learning rate at iteration i.

        - 'linear':     η(t) = η₀·(1 − t/T)
        - 'asymptotic': η(t) = η₀/(1 + t/(T/2)) — better Robbins-Monro conditions.
        """
        if mode == 'asymptotic':
            return initial_lr / (1.0 + i / (iterations_number / 2.0))
        # Divide by (T+1) so the final iteration gets a small but non-zero rate.
        return initial_lr * (1.0 - i / (iterations_number + 1))

    @staticmethod
    def variation_neighbourhood(initial_neighbourhood: float, i: int,
                                iterations_number: int, final: float = 0,
                                mode: str = 'linear') -> float:
        """Neighbourhood radius at iteration i; same schedule as variation_learning_rate.

        :param final: minimum radius at end of training (default 0).
        """
        if mode == 'asymptotic':
            return final + initial_neighbourhood / (
                1.0 + i / (iterations_number / 2.0))
        return final + initial_neighbourhood * (1.0 - i / iterations_number)

    @staticmethod
    def decay(distance_BMU: np.ndarray, current_neighbourhood: float) -> np.ndarray:
        """Gaussian neighbourhood h(r,t) = exp(−‖r−r_BMU‖²/(2σ²)) (Kohonen 1982)."""
        return np.exp(-(distance_BMU ** 2) / (2 * (current_neighbourhood ** 2)))

    def __adjust_weights(self, v: float, eta: float, bmu: Tuple, pattern: np.ndarray) -> None:
        """Kohonen update: w += η·h·(x−w), with h Gaussian or bubble per use_decay."""
        distances = self._grid_distance(self.__ids_matrix, bmu)

        if self.use_decay:
            h = self.decay(distances, v)
        else:
            h = (distances <= v).astype(float)

        self.weights += eta * np.expand_dims(h, axis=2) * (pattern - self.weights)

    def __normalize(self, data: np.ndarray, method: str) -> np.ndarray:
        """Normalise training data; store params for global methods so transform() is consistent."""
        if method in ('none',):
            return data

        data = data.astype(float, copy=True)

        if method in ('zscore', 'fwn'):
            mean = data.mean(axis=0)
            std  = data.std(axis=0)
            std[std == 0] = 1.0
            self._norm_params = {'mean': mean, 'std': std}
            return (data - mean) / std

        if method == 'robust':
            median = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            self._norm_params = {'median': median, 'iqr': iqr}
            return (data - median) / iqr

        if method == '01scale':
            lo  = data.min(axis=0)
            hi  = data.max(axis=0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            self._norm_params = {'lo': lo, 'hi': hi, 'rng': rng}
            return (data - lo) / rng

        if method == 'zscore_sample':
            mean = data.mean(axis=1, keepdims=True)
            std  = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            return (data - mean) / std

        if method == 'robust_sample':
            median = np.median(data, axis=1, keepdims=True)
            q75 = np.percentile(data, 75, axis=1, keepdims=True)
            q25 = np.percentile(data, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            return (data - median) / iqr

        if method == 'minmax_sample':
            lo = data.min(axis=1, keepdims=True)
            hi = data.max(axis=1, keepdims=True)
            rng = hi - lo
            rng[rng == 0] = 1.0
            return (data - lo) / rng

        if method in ('l2', 'euclidean'):
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return data / norms

        raise ValueError(
            f"Unknown normalization '{method}'. Available: "
            "'none', 'zscore'/'fwn', 'robust', '01scale', "
            "'zscore_sample', 'robust_sample', 'minmax_sample', 'l2'/'euclidean'"
        )

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the normalisation fitted during training to new data.

        Always pass raw (un-normalised) data at the same scale as train().

        :param data: (n, d) array
        :return: normalised array, same shape
        """
        method = self.normalization
        if method == 'none':
            return data

        data = data.astype(float, copy=True)
        p = self._norm_params

        if method in ('zscore', 'fwn'):
            return (data - p['mean']) / p['std']

        if method == 'robust':
            return (data - p['median']) / p['iqr']

        if method == '01scale':
            return (data - p['lo']) / p['rng']

        if method == 'zscore_sample':
            mean = data.mean(axis=1, keepdims=True)
            std  = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            return (data - mean) / std

        if method == 'robust_sample':
            median = np.median(data, axis=1, keepdims=True)
            q75 = np.percentile(data, 75, axis=1, keepdims=True)
            q25 = np.percentile(data, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            return (data - median) / iqr

        if method == 'minmax_sample':
            lo = data.min(axis=1, keepdims=True)
            hi = data.max(axis=1, keepdims=True)
            rng = hi - lo
            rng[rng == 0] = 1.0
            return (data - lo) / rng

        if method in ('l2', 'euclidean'):
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return data / norms

        return data

    def __init_weights(self, data: np.ndarray, method: str) -> np.ndarray:
        """Initialise the weight grid.

        :param method: 'random' [0,1], 'random_negative' [-1,1], 'sample' (real rows),
            or 'PCA' (plane of top-2 principal components; SVD for numerical stability).
        """
        if method == 'random':
            return self._rng.random(self.input_data_dimension *
                                    (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'random_negative':
            return self._rng.uniform(-1, 1, self.input_data_dimension *
                                     (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'sample':
            # Each neuron gets a real data row; preserves feature correlations.
            idx = self._rng.choice(self.num_data,
                                   size=self.map_size ** 2,
                                   replace=True)
            return data[idx].reshape(
                self.map_size, self.map_size, self.input_data_dimension)

        elif method == 'PCA':
            # SVD avoids squaring the condition number vs. explicit covariance eigen-decomp.
            data_c = data.astype(float) - data.mean(axis=0)
            _, _, Vt = np.linalg.svd(data_c, full_matrices=False)
            pc1, pc2 = Vt[0], Vt[1]
            pca_weights = np.zeros(
                (self.map_size, self.map_size, self.input_data_dimension))
            for i, c1 in enumerate(np.linspace(-1, 1, self.map_size)):
                for j, c2 in enumerate(np.linspace(-1, 1, self.map_size)):
                    pca_weights[i, j] = c1 * pc1 + c2 * pc2
            return pca_weights

        raise ValueError(f"Unknown weight initialisation method: '{method}'")

    @classmethod
    def load_classifier(cls, filename: str = 'Model') -> 'Map':
        """Load a trained Map from ``<filename>.json``."""
        with open(filename + '.json') as json_file:
            data = json.load(json_file)

            for model in data['model']:
                map_size = model['map_size']
                input_data_dimension = model['input_data_dimension']
                presentation = model['presentation']
                initial_lr = model['initial_lr']
                distance = model['distance']
                dtw_band  = model.get('dtw_band', None)
                use_decay = model['use_decay']
                num_data = model['num_data']
                period = model['period']
                neighbourhood = model['neighbourhood']
                normalization = model.get('normalization', 'none')
                weights_init  = model.get('weights_init', 'random')
                lr_decay      = model.get('lr_decay', 'linear')
                random_seed   = model.get('random_seed', None)
                weights = np.array(model['weights'])

        new_map = Map(data=None,
                      size=map_size,
                      period=period,
                      initial_lr=initial_lr,
                      initial_neighbourhood=neighbourhood,
                      distance=distance,
                      dtw_band=dtw_band,
                      use_decay=use_decay,
                      lr_decay=lr_decay,
                      normalization=normalization,
                      weights=weights_init,
                      random_seed=random_seed,
                      )

        new_map.weights = np.array(weights, dtype=float)
        new_map.input_data_dimension = input_data_dimension
        new_map.presentation = presentation
        new_map.num_data = num_data
        raw_params = model.get('norm_params', {})
        new_map._norm_params = {k: np.array(v) if isinstance(v, list) else v
                                for k, v in raw_params.items()}
        new_map.__trained = True
        logger.info('Imported successfully')

        return new_map

    def save_classifier(self, filename: str = 'Model') -> None:
        """Serialise the trained Map to ``<filename>.json``."""
        data: dict = {'model': []}

        data['model'].append({
            'map_size': self.map_size,
            'input_data_dimension': self.input_data_dimension,
            'presentation': self.presentation,
            'initial_lr': self.initial_lr,
            'distance': self.distance,
            'dtw_band': self.dtw_band,
            'use_decay': self.use_decay,
            'num_data': self.num_data,
            'period': self.period,
            'neighbourhood': self.neighbourhood,
            'normalization': self.normalization,
            'weights_init':  self.weights_init,
            'lr_decay':      self.lr_decay,
            'random_seed':   self.random_seed,
            'weights': self.weights.tolist(),
            'norm_params': {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in self._norm_params.items()},
        })

        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)

        logger.info('Saved successfully')
