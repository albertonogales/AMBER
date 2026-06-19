"""
Recurrent Self-Organising Map (RSOM) for temporal / sequential data.

Standard SOM treats every input independently, which discards temporal
structure.  TemporalMap extends Map with a context vector that accumulates
a decaying memory of recently activated neurons, making BMU search sensitive
to the history of the input sequence.

Update rule (Voegtlin, 2002):
    context_t  =  α · context_{t-1}  +  (1 - α) · w_{BMU_{t-1}}
    d_eff(x_t, w_j) = (1 - β) · d(x_t, w_j)  +  β · ||context_t - w_j||

Parameters
----------
context_weight  (α)  : retention of previous context; 0 = no memory, 1 = pure memory
context_influence (β): weight of context distance vs. signal distance;
                       0 = plain SOM, 1 = context-only

Notes
-----
- Data must be presented in temporal order; TemporalMap forces
  presentation='sequential' and warns if changed.
- Call reset_context() between independent sequences (e.g. between
  different EEG recordings or audio files).
- The context distance is always Euclidean in weight space, regardless of
  the signal-space distance metric chosen for BMU search.  When
  ``distance`` is not ``'euclidean'``, the two components of the combined
  metric operate on different scales; the effective balance set by
  ``context_influence`` therefore depends on the magnitude of both terms.
  For reliable scale parity, use ``distance='euclidean'`` or normalise
  data before training so all distances are O(1).
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np

from .distances import SIGNAL_DISTANCE_MATRIX, euclidean_distance_matrix
from .map import Map, vesanto_size

logger = logging.getLogger(__name__)


class TemporalMap(Map):
    """Recurrent SOM that incorporates a temporal context vector."""

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 size: Optional[int] = None,
                 period: int = 10,
                 initial_lr: float = 0.1,
                 initial_neighbourhood: int = 0,
                 distance: str = 'euclidean',
                 dtw_band: Optional[int] = None,
                 use_decay: bool = False,
                 normalization: str = 'none',
                 weights: str = 'random',
                 context_weight: float = 0.5,
                 context_influence: float = 0.5,
                 random_seed: Optional[int] = None) -> None:
        """
        :param context_weight: α — controls how much of the previous context is
            retained each step (0 = forget immediately, 1 = never update).
        :param context_influence: β — how strongly context distance contributes
            to BMU selection relative to signal distance (0 = plain SOM,
            0.5 = equal weight, 1 = context only).
        :param random_seed: Seed for the random number generator. Pass an integer
            for reproducible results. None (default) uses a non-deterministic seed.

        All other parameters are identical to Map.__init__.
        """
        assert 0.0 <= context_weight <= 1.0, 'context_weight must be in [0, 1]'
        assert 0.0 <= context_influence <= 1.0, 'context_influence must be in [0, 1]'

        self.context_weight = context_weight
        self.context_influence = context_influence
        self._context = None

        # Resolve size here because super().__init__ receives data=None,
        # so it cannot apply the Vesanto heuristic on its own.
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

        # Temporal SOM requires sequential presentation to preserve order.
        super().__init__(
            data=None,          # delay training until context is ready
            size=size,
            period=period,
            initial_lr=initial_lr,
            initial_neighbourhood=initial_neighbourhood,
            distance=distance,
            dtw_band=dtw_band,
            use_decay=use_decay,
            normalization=normalization,
            presentation='sequential',
            weights=weights,
            random_seed=random_seed,
        )

        if data is not None:
            self.train(data)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def reset_context(self) -> None:
        """Reset the context vector to zero.

        Call this between independent sequences (e.g. different subjects,
        different recordings) so that history from one sequence does not
        bleed into the next.
        """
        self._context = None

    # ------------------------------------------------------------------
    # Overridden BMU (context-aware)
    # ------------------------------------------------------------------

    def calculate_bmu(self, pattern: np.ndarray) -> Tuple:
        """BMU search incorporating the temporal context vector.

        Combines signal-space distance with context distance.  After the
        BMU is found, updates the context vector using the winner's weights.

        :param pattern: 1-D input array
        :return: (bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos)
        """
        dist_fn = SIGNAL_DISTANCE_MATRIX[self.distance]
        kwargs = {'band': self.dtw_band} if self.distance == 'dtw' else {}
        signal_dist = dist_fn(self.weights, pattern, **kwargs)

        if self._context is not None and self.context_influence > 0:
            context_dist = euclidean_distance_matrix(self.weights, self._context)
            combined = ((1.0 - self.context_influence) * signal_dist
                        + self.context_influence * context_dist)
        else:
            combined = signal_dist

        bmu_dist = float(np.min(combined))
        bmu_pos  = np.unravel_index(np.argmin(combined), combined.shape)

        combined[bmu_pos] = np.inf
        second_bmu_dist = float(np.min(combined))
        second_bmu_pos  = np.unravel_index(np.argmin(combined), combined.shape)

        # Update context with the winning neuron's weight vector
        winner_weights = self.weights[bmu_pos]
        if self._context is None:
            self._context = winner_weights.copy()
        else:
            self._context = (self.context_weight * self._context
                             + (1.0 - self.context_weight) * winner_weights)

        return bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos

    # ------------------------------------------------------------------
    # Overridden train / reinforce (reset context before each pass)
    # ------------------------------------------------------------------

    def train(self, data: np.ndarray) -> None:
        """Train the map on a temporally ordered dataset.

        Context is reset at the start of each training call so that
        separate calls to train() are independent.
        """
        self.reset_context()
        super().train(data)

    def reinforce(self, training_data: np.ndarray, reinforcement: int = 0,
                  extension: int = 1, compression: float = 0.5) -> None:
        """Reinforcement training; context is reset before each pass."""
        self.reset_context()
        super().reinforce(training_data, reinforcement, extension, compression)

    # ------------------------------------------------------------------
    # Serialisation (extends parent JSON with temporal parameters)
    # ------------------------------------------------------------------

    def save_classifier(self, filename: str = 'Model') -> None:
        """Save map to JSON, including temporal parameters."""
        data: dict = {'model': []}
        data['model'].append({
            'map_size':             self.map_size,
            'input_data_dimension': self.input_data_dimension,
            'presentation':         self.presentation,
            'initial_lr':           self.initial_lr,
            'distance':             self.distance,
            'dtw_band':             self.dtw_band,
            'use_decay':            self.use_decay,
            'num_data':             self.num_data,
            'period':               self.period,
            'neighbourhood':        self.neighbourhood,
            'normalization':        self.normalization,
            'weights_init':         self.weights_init,
            'context_weight':       self.context_weight,
            'context_influence':    self.context_influence,
            'random_seed':          self.random_seed,
            'weights':              self.weights.tolist(),
            'norm_params':          {k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in self._norm_params.items()},
        })
        with open(filename + '.json', 'w') as f:
            json.dump(data, f)
        logger.info('Saved successfully')

    @classmethod
    def load_classifier(cls, filename: str = 'Model') -> 'TemporalMap':
        """Load a TemporalMap from a JSON file saved by save_classifier."""
        with open(filename + '.json') as f:
            raw = json.load(f)
        model = raw['model'][0]

        tm = cls(
            data=None,
            size=model['map_size'],
            period=model['period'],
            initial_lr=model['initial_lr'],
            initial_neighbourhood=model['neighbourhood'],
            distance=model['distance'],
            dtw_band=model.get('dtw_band'),
            use_decay=model['use_decay'],
            normalization=model.get('normalization', 'none'),
            weights=model.get('weights_init', 'random'),
            context_weight=model.get('context_weight', 0.5),
            context_influence=model.get('context_influence', 0.5),
            random_seed=model.get('random_seed', None),
        )
        tm.weights              = np.array(model['weights'])
        tm.input_data_dimension = model['input_data_dimension']
        tm.num_data             = model['num_data']
        raw_params = model.get('norm_params', {})
        tm._norm_params = {k: np.array(v) if isinstance(v, list) else v
                           for k, v in raw_params.items()}
        tm._Map__trained        = True   # type: ignore[attr-defined]  # name-mangled parent attr
        logger.info('Imported successfully')
        return tm
