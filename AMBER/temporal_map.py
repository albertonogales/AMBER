"""
Recurrent SOM (RSOM) for temporal/sequential data (Voegtlin, 2002).

Extends Map with a context vector that decays past activations into BMU search:
    context_t  = α·context_{t-1} + (1-α)·w_{BMU_{t-1}}
    d_eff(x,w) = (1-β)·d(x,w) + β·d(context,w)

Call reset_context() between independent sequences.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np

from .distances import SIGNAL_DISTANCE_MATRIX
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
                 lr_decay: str = 'linear',
                 normalization: str = 'none',
                 weights: str = 'random',
                 context_weight: float = 0.5,
                 context_influence: float = 0.5,
                 random_seed: Optional[int] = None,
                 confirm: bool = False) -> None:
        """
        :param context_weight: α — context retention per step (0 = forget, 1 = freeze).
        :param context_influence: β — context vs. signal weight in BMU search (0 = plain SOM).
        :param random_seed: integer for reproducible results; None = non-deterministic.
        :param confirm: must be True to proceed with training; forces the user to acknowledge
            that the data meets TemporalMap requirements (single session, temporal order, ≥10 samples).

        All other parameters are identical to Map.__init__.
        """
        if not 0.0 <= context_weight <= 1.0:
            raise ValueError(f"context_weight must be in [0, 1], got {context_weight}.")
        if not 0.0 <= context_influence <= 1.0:
            raise ValueError(f"context_influence must be in [0, 1], got {context_influence}.")

        self.context_weight = context_weight
        self.context_influence = context_influence
        self._context = None
        self._confirm = confirm

        # Resolve size before super().__init__ since we pass data=None to defer training.
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

        super().__init__(
            data=None,
            size=size,
            period=period,
            initial_lr=initial_lr,
            initial_neighbourhood=initial_neighbourhood,
            distance=distance,
            dtw_band=dtw_band,
            use_decay=use_decay,
            lr_decay=lr_decay,
            normalization=normalization,
            presentation='sequential',  # order must be preserved for temporal context
            weights=weights,
            random_seed=random_seed,
        )

        if data is not None:
            self.train(data)

    def reset_context(self) -> None:
        """Zero the context vector; call between independent sequences."""
        self._context = None

    def calculate_bmu(self, pattern: np.ndarray) -> Tuple:
        """BMU search combining signal and context distances; updates context after."""
        dist_fn = SIGNAL_DISTANCE_MATRIX[self.distance]
        kwargs = {'band': self.dtw_band} if self.distance == 'dtw' else {}
        signal_dist = dist_fn(self.weights, pattern, **kwargs)

        if self._context is not None and self.context_influence > 0:
            context_dist = dist_fn(self.weights, self._context, **kwargs)
            combined = ((1.0 - self.context_influence) * signal_dist
                        + self.context_influence * context_dist)
        else:
            combined = signal_dist

        bmu_dist = float(np.min(combined))
        bmu_pos  = np.unravel_index(np.argmin(combined), combined.shape)

        combined[bmu_pos] = np.inf
        second_bmu_dist = float(np.min(combined))
        second_bmu_pos  = np.unravel_index(np.argmin(combined), combined.shape)

        winner_weights = self.weights[bmu_pos]
        if self._context is None:
            self._context = winner_weights.copy()
        else:
            self._context = (self.context_weight * self._context
                             + (1.0 - self.context_weight) * winner_weights)

        return bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos

    def train(self, data: np.ndarray) -> None:
        """Train on a single temporally ordered sequence; resets context before training."""
        self._check_temporal_assumptions(data, self._confirm)
        self.reset_context()
        super().train(data)

    def train_sequential(self,
                         recordings: list,
                         n_passes: int = 10,
                         shuffle_recordings: bool = True) -> None:
        """
        Train RSOM on multiple independent sequences with context reset between them.

        This is the correct training procedure for multi-session temporal data
        (e.g., multiple nights of EEG, multiple sensor runs).  The context vector
        is zeroed at the start of every recording so that temporal memory does not
        bleed across sessions.  Within each recording, samples are presented in
        their original (temporal) order.

        :param recordings: list of 2-D arrays of shape (n_epochs, n_features).
                           All arrays must share the same feature dimension.
        :param n_passes: number of full passes over the complete set of recordings.
        :param shuffle_recordings: if True (default), shuffle the order in which
                                   recordings are visited each pass, while preserving
                                   within-recording temporal order.
        """
        if not recordings:
            raise ValueError("recordings must be a non-empty list of arrays.")
        n_feat = recordings[0].shape[1]
        for i, rec in enumerate(recordings):
            if rec.ndim != 2 or rec.shape[1] != n_feat:
                raise ValueError(
                    f"recordings[{i}] has shape {rec.shape}; "
                    f"expected (n_epochs, {n_feat})."
                )
            self._check_temporal_assumptions(rec, self._confirm)

        if not self._confirm:
            raise ValueError(
                "Set confirm=True on the TemporalMap constructor to acknowledge "
                "that each recording is a single continuous temporally ordered sequence."
            )

        all_data = np.concatenate(recordings, axis=0)
        self.num_data = all_data.shape[0]
        self.input_data_dimension = n_feat

        # Normalise using statistics from all recordings combined.
        all_norm = self._Map__normalize(all_data, method=self.normalization)  # type: ignore[attr-defined]
        self.weights = self._Map__init_weights(data=all_norm, method=self.weights_init)  # type: ignore[attr-defined]

        # Split back into per-recording normalised arrays.
        recordings_norm, offset = [], 0
        for rec in recordings:
            n = len(rec)
            recordings_norm.append(all_norm[offset:offset + n])
            offset += n

        total_steps = self.num_data * n_passes
        self.period  = total_steps
        step = 0

        logger.info(
            f"SEQUENTIAL TRAINING: {n_passes} passes × {len(recordings)} recordings "
            f"= {total_steps} steps."
        )

        for _pass in range(n_passes):
            order = (self._rng.permutation(len(recordings))
                     if shuffle_recordings
                     else range(len(recordings)))
            for idx in order:
                self.reset_context()
                for pattern in recordings_norm[idx]:
                    bmu = self.calculate_bmu(pattern)
                    eta = self.variation_learning_rate(
                        self.initial_lr, step + 1, total_steps, mode=self.lr_decay)
                    v_final = 1 if self.use_decay else 0
                    v = self.variation_neighbourhood(
                        self.neighbourhood, step + 1, total_steps, v_final,
                        mode=self.lr_decay)
                    self._Map__adjust_weights(v, eta, bmu[1], pattern)  # type: ignore[attr-defined]
                    step += 1

        self._Map__trained = True  # type: ignore[attr-defined]
        logger.info("FINISHED (sequential multi-recording training).")

    @staticmethod
    def _check_temporal_assumptions(data: np.ndarray, confirmed: bool) -> None:
        """Enforce that the user has acknowledged TemporalMap data requirements."""
        issues = []
        if data.shape[0] < 10:
            issues.append(
                f"  - Too few samples ({data.shape[0]}): the context vector needs at least "
                f"10 consecutive samples to stabilise."
            )

        requirement_msg = (
            "\n"
            "TemporalMap requires data from a SINGLE continuous sequence (one individual,\n"
            "one session) in TEMPORAL ORDER. Each row must be one epoch of that stream.\n"
            "\n"
            "Before training, verify ALL of the following:\n"
            "  [1] All samples belong to the same individual / recording session.\n"
            "  [2] Samples are in chronological order (not shuffled).\n"
            "  [3] Each row is one epoch from a continuous stream (e.g. a 30-s EEG window).\n"
            "  [4] You have at least 10 samples (ideally 50+) per sequence.\n"
            "\n"
            "For multi-patient data: train a plain Map on all patients, then classify\n"
            "each patient separately calling tm.reset_context() between sessions.\n"
            "\n"
            "Once verified, set confirm=True to proceed with training."
        )

        if issues:
            raise ValueError(
                "\nData does not meet TemporalMap requirements:\n"
                + "\n".join(issues)
                + "\n\nTraining aborted."
            )

        if not confirmed:
            raise ValueError(requirement_msg)

    def reinforce(self, training_data: np.ndarray, reinforcement: int = 0,
                  extension: int = 1, compression: float = 0.5) -> None:
        """Reinforcement training; validates data requirements and resets context before each pass."""
        self._check_temporal_assumptions(training_data, self._confirm)
        self.reset_context()
        super().reinforce(training_data, reinforcement, extension, compression)

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
            'lr_decay':             self.lr_decay,
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
            lr_decay=model.get('lr_decay', 'linear'),
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
        tm._confirm             = True   # loaded map was already confirmed at training time
        logger.info('Imported successfully')
        return tm
