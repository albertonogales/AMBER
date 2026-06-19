from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distances import AVAILABLE_DISTANCES, GRID_DISTANCE, SIGNAL_DISTANCE_MATRIX

logger = logging.getLogger(__name__)


def vesanto_size(n_samples: int) -> int:
    """Return the map side length recommended by Vesanto & Alhoniemi (2000).

    The rule of thumb is that the total number of neurons should be
    approximately 5·√N, giving a square map of side √(5·√N).

    Reference:
        Vesanto, J. & Alhoniemi, E. (2000). Clustering of the
        self-organizing map. IEEE Transactions on Neural Networks, 11(3).

    :param n_samples: number of training samples N
    :return: integer map side length (minimum 2)
    """
    return max(2, round(np.sqrt(5.0 * np.sqrt(n_samples))))

class Map:
    """
    Map class is the main component of AMBER. It contains the classifying map that allows for classification and
    is subject of analysis in search of data information
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

        """Initializing the map requires some information provided

        :param data: numpy array of 2 dimensions. First dimension corresponds to data samples, while the second
        represents an specific sample's data
        :param size: side length of the square map (map will be size×size neurons).
            If None (default), the size is chosen automatically using the Vesanto &
            Alhoniemi (2000) heuristic: side = √(5·√N), where N is the number of
            training samples.  'data' must be provided when size is None.
        :param period: Total number of individual pattern presentations during training.
            Each step selects **one** pattern (randomly or sequentially) and updates the map.
            This is *not* the number of full passes over the dataset — ``period=T`` means
            T weight updates in total.  To approximate K full epochs over N samples, set
            ``period = K * N``.  A small value produces an undertrained map; too large a
            value compresses activations towards the map borders.
        :param initial_lr: Learning rate determines how much neurons will move on the map
        :param initial_neighbourhood: Initial neighbourhood determines how many neurons will learn. If none is
        provided, it will default to size
        :param distance: Distance used to find the BMU (signal space).
            Options:
            The first group correspond to classical ones and has been provided to be used with signal feature extraction
            - 'euclidean'        L2; general purpose (default)
            - 'manhattan'        L1; robust to spike artefacts
            - 'chebyshev'        L∞; sensitive to the single largest deviation
            The second group correspond to those focused on signal similarity
            - 'cosine'           amplitude-invariant; suited to spectral feature vectors
            - 'correlation'      shape-only; ignores mean and scale
            - 'dtw'              handles temporal misalignment; best for raw biosignals/audio
            - 'cross_correlation' shift-invariant; suited to periodic signals (ECG, EEG rhythms)
        :param dtw_band: Sakoe-Chiba half-width in samples for DTW (None = unconstrained).
            Ignored for other distances. Constraining the band reduces O(n²) cost.
        :param use_decay: If True, use a Gaussian neighbourhood function (smooth influence taper).
            If False, use a bubble function (uniform influence within radius, zero outside).
        :param lr_decay: Learning-rate (and neighbourhood-radius) decay schedule.
            - ``'linear'``     : η(t) = η₀·(1 − t/T)  — simple, widely used (default)
            - ``'asymptotic'`` : η(t) = η₀/(1 + t/(T/2)) — faster early decay, slower
              fine-tuning phase; better satisfies Robbins-Monro convergence conditions
        :param normalization: Normalization applied to training data before each train() call.
            Global strategies operate across all samples (per feature column):
            - 'none'           : no normalization (default)
            - 'zscore' / 'fwn' : per-feature z-score  (mean=0, std=1 across samples)
            - 'robust'         : per-feature median/IQR scaling; robust to outliers and artefacts
            - '01scale'        : global min-max scaling to [0, 1]
            Per-sample strategies normalize each window independently (row-wise):
            - 'zscore_sample'  : z-score within each window; removes baseline and amplitude
            - 'robust_sample'  : median/IQR within each window
            - 'minmax_sample'  : min-max [0,1] within each window
            - 'l2' / 'euclidean': L2-normalise each window to unit norm
        :param presentation: If set to 'sequential' data will be presented sequentially, otherwise it will be presented
        randomly
        :param weights: Technique used to initialize the weights. Current options include:
            - 'random': From 0 to 1
            - 'random_negative': From -1 to 1
            - 'sample': Takes samples from data. This is useful if data is not normalized
            - 'PCA': sequence of vectors taken along a hyperplane spanned by the two largest principal components of the dataset.
        :param random_seed: Seed for the random number generator. Pass an integer for
            reproducible results (same seed → same weights and same training trajectory).
            None (default) uses a non-deterministic seed.
        """

        # Validate input parameters
        self.__trained = False
        if period <= 1:
            raise ValueError(f"'period' must be > 1, got {period}.")
        if not (0 < initial_lr < 1):
            raise ValueError(f"'initial_lr' must be in (0, 1), got {initial_lr}.")

        # Store seed and create seeded RNG
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # Resolve map size
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

        # Initialize weights and normalisation state
        self.weights = np.ones(1)
        self._norm_params: dict = {}

        # Create index matrix
        ids: list[list[list[int]]] = []
        for y in range(self.map_size):
            row: list[list[int]] = []
            for x in range(self.map_size):
                row.append([y, x])
            ids.append(row)
        self.__ids_matrix: np.ndarray = np.array(ids)

        if data is not None:
            self.train(data)

    def train(self,
              data: np.ndarray) -> None:
        """Train the SOM on the provided data.

        :param data: 2-D numpy array — rows are samples, columns are features
        """
        self.num_data = data.shape[0]
        self.input_data_dimension = data.shape[1]
        # Normalizating input data
        training_data = self.__normalize(data, method=self.normalization)
        if not np.all(np.isfinite(training_data)):
            raise ValueError(
                "Training data contains non-finite values (inf or nan) after "
                "normalization. Check input data or feature extraction — e.g. "
                "sample_entropy returns inf for signals with no template matches."
            )
        self.weights = self.__init_weights(data=training_data, method=self.weights_init)

        logger.info("TRAINING...")
        # Input patterns
        for numPresentation in tqdm(range(1, self.period + 1)):
            if self.presentation == 'sequential':
                # Select patterns sequentialy
                new_pattern = training_data[(numPresentation - 1) % self.num_data]
            else:
                # Select patterns randomly
                new_pattern = training_data[self._rng.integers(0, self.num_data)]

            # Getting the winner neuron
            bmu = self.calculate_bmu(new_pattern)

            # Getting learning rate value and current neighbourhood
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
        """Continue training with a fine-tuning (reinforcement) phase.

        Each reinforcement round multiplies ``period`` by ``extension`` and
        compresses the learning rate by ``compression``.  The neighbourhood
        radius decays from its current value to 1 over the extended period,
        honouring the ``use_decay`` and ``lr_decay`` settings configured at
        construction time.

        .. note::
            ``reinforcement=0`` (default) is a no-op — the map is unchanged.
            Pass ``reinforcement >= 1`` to activate the phase.

        :param training_data: 2-D numpy array — rows are samples, columns are features
        :param reinforcement: number of additional reinforcement rounds (0 = no-op)
        :param extension: period multiplier applied each round (e.g. 2 doubles iterations)
        :param compression: learning-rate scale factor per round (e.g. 0.5 halves lr)
        """
        norm_data = self.transform(training_data)
        n_reinforce = len(norm_data)
        origin_initial_lr = self.initial_lr
        # Fine-tuning neighbourhood starts from the trained initial value and decays to 1.
        # Across rounds it continues from 1 (fine-tuning phase, not ordering phase).
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

            # After each round neighbourhood has converged to 1; keep it there for next round
            round_neighbourhood = 1

        self.__trained = True

    # GETTING BMU AND SECOND BMU
    def calculate_bmu(self, pattern: np.ndarray) -> Tuple:
        """Calculates the Best Matching Unit (BMU) for a pattern using the
        configured signal-space distance.

        :param pattern: 1-D array of the input pattern
        :return:
            - bmu_dist:        distance from pattern to BMU weight vector
            - bmu_pos:         (row, col) grid coordinates of the BMU
            - second_bmu_dist: distance from pattern to second-best neuron
            - second_bmu_pos:  (row, col) grid coordinates of the second-best neuron
        """
        dist_fn = SIGNAL_DISTANCE_MATRIX[self.distance]
        kwargs = {'band': self.dtw_band} if self.distance == 'dtw' else {}
        distances = dist_fn(self.weights, pattern, **kwargs)  # type: ignore[operator]

        bmu_dist = np.min(distances)
        bmu_pos  = np.unravel_index(np.argmin(distances), distances.shape)

        distances[bmu_pos] = np.inf

        second_bmu_dist = np.min(distances)
        second_bmu_pos  = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos

    def _grid_distance(self, ids_matrix: np.ndarray, bmu_pos: Tuple) -> np.ndarray:
        """Distance between neuron grid positions; used for neighbourhood update.
        Always operates in 2-D grid space, independent of the signal distance.

        :param ids_matrix: (rows, cols, 2) array of grid coordinates
        :param bmu_pos:    (row, col) position of the BMU
        :return:           (rows, cols) distance array
        """
        grid_dist_fn = GRID_DISTANCE.get(self.distance, GRID_DISTANCE['euclidean'])
        return grid_dist_fn(ids_matrix, bmu_pos)

    # VARIATION OF LEARNING RATE
    @staticmethod
    def variation_learning_rate(initial_lr: float, i: int, iterations_number: int,
                                mode: str = 'linear') -> float:
        """Calculate the learning rate for iteration *i*.

        Two decay schedules are supported:

        * ``'linear'``     : η(t) = η₀ · (1 − t/T)
          Simple linear decay to zero.  Widely used and easy to reason about.

        * ``'asymptotic'`` : η(t) = η₀ / (1 + t / (T/2))
          Decays quickly at first (coarse ordering) then slows down
          (fine-tuning).  Better satisfies the Robbins–Monro stochastic
          approximation convergence conditions (Ση = ∞, Ση² < ∞)
          (Robbins & Monro, 1951, Ann. Math. Stat. 22(3):400-407).

        :param initial_lr: initial learning rate η₀
        :param i: current iteration index (1-based)
        :param iterations_number: total number of iterations T
        :param mode: ``'linear'`` (default) or ``'asymptotic'``
        :return: learning rate for iteration i
        """
        if mode == 'asymptotic':
            return initial_lr / (1.0 + i / (iterations_number / 2.0))
        # default: linear — divides by (T+1) so the final iteration receives a small
        # but non-zero learning rate instead of exactly 0.
        return initial_lr * (1.0 - i / (iterations_number + 1))

    # VARIATION NEIGHBOURHOOD
    @staticmethod
    def variation_neighbourhood(initial_neighbourhood: float, i: int,
                                iterations_number: int, final: float = 0,
                                mode: str = 'linear') -> float:
        """Calculate the neighbourhood radius for iteration *i*.

        Uses the same decay schedule as the learning rate.

        * ``'linear'``     : σ(t) = σ_final + σ₀ · (1 − t/T)
        * ``'asymptotic'`` : σ(t) = σ_final + σ₀ / (1 + t / (T/2))

        :param initial_neighbourhood: initial neighbourhood radius σ₀
        :param i: current iteration index (1-based)
        :param iterations_number: total number of iterations T
        :param final: minimum radius retained at the end (default 0)
        :param mode: ``'linear'`` (default) or ``'asymptotic'``
        :return: neighbourhood radius for iteration i
        """
        if mode == 'asymptotic':
            return final + initial_neighbourhood / (
                1.0 + i / (iterations_number / 2.0))
        # default: linear
        return final + initial_neighbourhood * (1.0 - i / iterations_number)

    # NEIGHBOURHOOD FUNCTION
    @staticmethod
    def decay(distance_BMU: np.ndarray, current_neighbourhood: float) -> np.ndarray:
        """Gaussian neighbourhood function h(r, t).

        Returns the influence weight for every neuron given its grid distance
        to the BMU and the current neighbourhood radius σ(t):

            h(r, t) = exp(−‖r − r_BMU‖² / (2σ(t)²))

        This is Kohonen's original formulation.  Influence tapers smoothly
        toward zero as distance grows — no hard boundary is applied.

        :param distance_BMU: (rows, cols) array of grid distances to the BMU
        :param current_neighbourhood: current neighbourhood radius σ(t)
        :return: (rows, cols) array of influence weights in (0, 1]
        """
        return np.exp(-(distance_BMU ** 2) / (2 * (current_neighbourhood ** 2)))

    # Function to update weights
    def __adjust_weights(self, v: float, eta: float, bmu: Tuple, pattern: np.ndarray) -> None:
        """Update all neuron weights for one training step.

        Uses the Kohonen weight-update rule:

            w(t+1) = w(t) + η(t) · h(r, t) · (x − w(t))

        where h(r, t) is:

        * **Gaussian** (``use_decay=True``) — smooth influence taper following
          Kohonen (1982).  No hard boundary: every neuron receives a small
          update, but neurons far from the BMU are affected negligibly.

        * **Bubble** (``use_decay=False``) — uniform influence 1 inside the
          neighbourhood radius, 0 outside.  Simpler but introduces a
          discontinuity at the boundary.

        :param v: current neighbourhood radius σ(t)
        :param eta: current learning rate η(t)
        :param bmu: (row, col) BMU grid position
        :param pattern: current input pattern x
        """
        distances = self._grid_distance(self.__ids_matrix, bmu)

        if self.use_decay:
            # Pure Gaussian — smooth taper, no hard cut (Kohonen 1982)
            h = self.decay(distances, v)
        else:
            # Bubble — binary mask inside radius
            h = (distances <= v).astype(float)

        # Vectorised update over the full weight grid
        self.weights += eta * np.expand_dims(h, axis=2) * (pattern - self.weights)

    def __normalize(self, data: np.ndarray, method: str) -> np.ndarray:
        """Normalize training data and store parameters for global methods.

        Global methods operate per feature column and store the fitted
        parameters in ``self._norm_params`` so that :meth:`transform` can
        apply the same transformation consistently to new data.
        Per-sample methods normalize each row independently — no parameters
        are stored.
        """
        if method in ('none',):
            return data

        data = data.astype(float, copy=True)

        # --- global, per-feature ---
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
            # Per-feature min-max scaling to [0, 1]
            lo  = data.min(axis=0)
            hi  = data.max(axis=0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            self._norm_params = {'lo': lo, 'hi': hi, 'rng': rng}
            return (data - lo) / rng

        # --- per-sample (row-wise): no global parameters stored ---
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
        """Apply the normalization fitted during training to new data.

        For global methods (``zscore``, ``robust``, ``01scale``), uses the
        parameters stored during :meth:`train`.  For per-sample methods the
        transformation is reapplied independently to each row.  For
        ``'none'`` returns ``data`` unchanged.

        Always pass **raw** (un-normalised) data — the same scale as what
        was passed to :meth:`train`.

        :param data: 2-D array of samples, shape ``(n, d)``
        :return: normalised array with the same shape
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

        # Per-sample methods — reapply independently (no stored params needed)
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

        return data  # 'none' or unrecognised method — return unchanged

    def __init_weights(self, data: np.ndarray, method: str) -> np.ndarray:
        """ Function to initialize the weights matrix

        :param data: Data used to train the map
        :param method: Method to initialize the map. Available options include:
            - 'random': From 0 to 1
            - 'random_negative': From -1 to 1
            - 'sample': Takes samples from data. This is useful if data is not normalized
        :return:
        """
        if method == 'random':
            # Getting the weights from random values between 0 and 1
            return self._rng.random(self.input_data_dimension *
                                    (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'random_negative':
            # Getting the weights from random values between -1 and 1
            return self._rng.uniform(-1, 1, self.input_data_dimension *
                                     (self.map_size ** 2)).reshape(
                (self.map_size, self.map_size, self.input_data_dimension))

        elif method == 'sample':
            # Draw map_size² random whole samples from the training data.
            # Each neuron is initialised to a real data point, preserving
            # feature correlations.  (Previous implementation incorrectly
            # sampled individual scalars, destroying all correlations.)
            idx = self._rng.choice(self.num_data,
                                   size=self.map_size ** 2,
                                   replace=True)
            return data[idx].reshape(
                self.map_size, self.map_size, self.input_data_dimension)

        elif method == 'PCA':
            # Initialise weights along the plane spanned by the two principal
            # components of the training data (Kohonen 2001, p. 154).
            #
            # We use SVD on the mean-centred data matrix rather than
            # eigendecomposition of the explicit covariance matrix.  SVD is
            # numerically more stable for high-dimensional data because it
            # avoids squaring the condition number of the data matrix.
            data_c = data.astype(float) - data.mean(axis=0)
            _, _, Vt = np.linalg.svd(data_c, full_matrices=False)
            pc1, pc2 = Vt[0], Vt[1]          # top-2 right singular vectors
            pca_weights = np.zeros(
                (self.map_size, self.map_size, self.input_data_dimension))
            for i, c1 in enumerate(np.linspace(-1, 1, self.map_size)):
                for j, c2 in enumerate(np.linspace(-1, 1, self.map_size)):
                    pca_weights[i, j] = c1 * pc1 + c2 * pc2
            return pca_weights

        # Fallback — should never be reached given input validation in train()
        raise ValueError(f"Unknown weight initialisation method: '{method}'")

    ######################################################
    #                    JSON METHODS                    #
    ######################################################

    # LOAD CLASSIFIER FROM THE FILE
    @classmethod
    def load_classifier(cls, filename: str = 'Model') -> 'Map':
        """Load a previously saved Map from a JSON file.

        :param filename: path without the ``.json`` extension (default ``'Model'``)
        :return: a fully restored :class:`Map` instance ready for classification
        :raises FileNotFoundError: if ``<filename>.json`` does not exist
        """
        # Opening the JSON file and getting all the models
        with open(filename + '.json') as json_file:
            data = json.load(json_file)

            # Reading and setting all the attributes
            for model in data['model']:
                map_size = model['map_size']
                input_data_dimension = model['input_data_dimension']
                presentation = model['presentation']
                initial_lr = model['initial_lr']
                distance = model['distance']
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
                      use_decay=use_decay,
                      lr_decay=lr_decay,
                      normalization=normalization,
                      weights=weights_init,
                      random_seed=random_seed,
                      )

        new_map.weights = weights
        new_map.input_data_dimension = input_data_dimension
        new_map.presentation = presentation
        new_map.num_data = num_data
        raw_params = model.get('norm_params', {})
        new_map._norm_params = {k: np.array(v) if isinstance(v, list) else v
                                for k, v in raw_params.items()}
        new_map.__trained = True
        # Showing a message to the user
        logger.info('Imported successfully')

        return new_map

    # SAVE CLASSIFIER IN THE FILE
    def save_classifier(self, filename: str = 'Model') -> None:
        """Serialise the trained Map to a JSON file.

        Saves weights, hyperparameters, and training metadata so the map can
        be fully restored with :meth:`load_classifier`.

        :param filename: path without the ``.json`` extension (default ``'Model'``)
        """
        # Creating the JSON object
        data: dict = {'model': []}

        # Setting array

        # Appending the model
        data['model'].append({
            'map_size': self.map_size,
            'input_data_dimension': self.input_data_dimension,
            'presentation': self.presentation,
            'initial_lr': self.initial_lr,
            'distance': self.distance,
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

        # Writing in the file
        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)

        # Showing a message to the user
        logger.info('Saved successfully')
