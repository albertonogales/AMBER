from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mode as _scipy_mode
from tqdm.auto import tqdm

from .distances import SIGNAL_DISTANCE_SCALAR, euclidean_distance
from .map import Map

logger = logging.getLogger(__name__)


class Classification:
    """Classifies data on a trained Map and computes quality metrics."""

    def __init__(self, som: Map, classification_data: np.ndarray,
                 other: Optional[pd.DataFrame] = None,
                 tagged: bool = False,
                 verbose: bool = False) -> None:
        """
        :param som: trained Map instance
        :param classification_data: (n_samples, n_features) array to classify
        :param other: optional DataFrame concatenated with classification_map
        :param tagged: if True, first column of classification_data holds labels
        :param verbose: log debug info about labels and data
        """
        pd.options.mode.chained_assignment = None

        if classification_data.shape[0] == 0:
            raise ValueError("classification_data must contain at least one sample.")

        if tagged:
            if classification_data.shape[1] < 2:
                raise ValueError(
                    "tagged=True requires at least 2 columns (label + 1 feature), "
                    f"got {classification_data.shape[1]}."
                )
            self.classification_labels = classification_data[:, 0]
            self.classification_data = classification_data[:, 1:]
        else:
            self.classification_data = classification_data
            self.classification_labels = np.arange(classification_data.shape[0])

        if self.classification_data.shape[1] != som.input_data_dimension:
            raise ValueError(
                f"Feature dimension mismatch: map was trained on "
                f"{som.input_data_dimension} features, got {self.classification_data.shape[1]}."
            )

        if not np.all(np.isfinite(self.classification_data)):
            raise ValueError(
                "classification_data contains NaN or inf values. "
                "Check your data or feature extraction pipeline."
            )

        if verbose:
            logger.debug("\n\nTags: \n" + str(self.classification_labels))
            logger.debug("\n\nClassification data: \n" + str(self.classification_data))

        self._k = som.map_size
        self.activations_map = np.zeros((som.map_size, som.map_size), dtype=int)
        self.distances_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.topological_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.umatriz = np.zeros((som.map_size * 2 - 1, som.map_size * 2 - 1), dtype=float)
        self.topological_error: float = 0.0
        self.quantization_error: float = 0.0           # configured distance (primary)
        self.quantization_error_euclidean: float = 0.0 # always Euclidean (for cross-library comparison)
        self.distortion: float = 0.0
        self.topological_error_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.quantization_error_map = np.zeros((som.map_size, som.map_size), dtype=float)

        bmu_positions = np.zeros((self.classification_data.shape[0], 2), dtype=int)

        structure = {
            'labels': self.classification_labels.tolist(),
            'data': self.classification_data.tolist(),
            'x': np.zeros(self.classification_data.shape[0], dtype=int).tolist(),
            'y': np.zeros(self.classification_data.shape[0], dtype=int).tolist(),
            'dist': np.zeros(self.classification_data.shape[0], dtype=float).tolist()
        }

        self.classification_map = pd.DataFrame(structure)

        if other is not None:
            self.classification_map = pd.concat([self.classification_map, other], axis=1)

        scalar_dist_fn = SIGNAL_DISTANCE_SCALAR[som.distance]
        dtw_kwargs = {'band': som.dtw_band} if som.distance == 'dtw' else {}

        n_samples = self.classification_data.shape[0]

        # Apply training normalisation so BMU search uses the same feature space as the weights.
        norm_data = som.transform(self.classification_data)

        for pattern in tqdm(range(0, n_samples)):
            bmu, bmu_pos, second_bmu, second_bmu_pos = som.calculate_bmu(norm_data[pattern])

            if np.max(np.abs(np.array(bmu_pos) - np.array(second_bmu_pos))) > 1:  # Chebyshev > 1 → non-adjacent
                self.topological_map[bmu_pos] += 1

            distance = scalar_dist_fn(som.weights[bmu_pos], norm_data[pattern],
                                      **dtw_kwargs)
            self.activations_map[bmu_pos] += 1
            self.distances_map[bmu_pos] += distance
            bmu_positions[pattern] = bmu_pos

            self.classification_map.loc[pattern, 'x']    = bmu_pos[0]
            self.classification_map.loc[pattern, 'y']    = bmu_pos[1]
            self.classification_map.loc[pattern, 'dist'] = distance

        self.num_activations = np.count_nonzero(self.activations_map != 0)

        self.mean_distance_map = np.sum(self.distances_map) / n_samples

        self.distances_map = np.around(self.distances_map, decimals=5)

        self.topological_error = np.sum(self.topological_map) / n_samples
        self.topological_error_map = np.divide(self.topological_map, self.activations_map,
                                               out=np.zeros_like(self.topological_map),
                                               where=self.activations_map != 0)

        self.quantization_error = np.sum(self.distances_map) / n_samples
        self.quantization_error_map = np.divide(self.distances_map, self.activations_map,
                                                out=np.zeros_like(self.distances_map),
                                                where=self.activations_map != 0)

        # Secondary QE always Euclidean; vectorised over all samples.
        bmu_weights = som.weights[bmu_positions[:, 0], bmu_positions[:, 1], :]
        eucl_total = float(np.sum(np.sqrt(np.sum((bmu_weights - norm_data)**2, axis=-1))))
        self.quantization_error_euclidean = eucl_total / n_samples

        # Distortion (Graepel et al. 1997): D = (1/N) Σᵢ Σⱼ h_σ(BMU(xᵢ),j)·‖xᵢ−wⱼ‖²
        sigma = max(float(som.neighbourhood), 1.0)
        k = som.map_size
        grid_idx = np.array([[[i, j] for j in range(k)] for i in range(k)],
                            dtype=float)

        bmu_r = bmu_positions.astype(float)
        sq_dist_grid = np.sum(
            (grid_idx[np.newaxis] - bmu_r[:, np.newaxis, np.newaxis, :]) ** 2,
            axis=-1
        )
        h = np.exp(-sq_dist_grid / (2.0 * sigma ** 2))
        diff = som.weights[np.newaxis] - norm_data[:, np.newaxis, np.newaxis, :]
        sq_dist_w = np.sum(diff ** 2, axis=-1)
        self.distortion = float(np.sum(h * sq_dist_w)) / n_samples

        # U-Matrix (Ultsch & Siemon 1990): (2k-1)×(2k-1) grid of inter-neuron distances.
        # Odd indices = edge distances; even indices = neuron cells (mean of adjacent edges).
        size = 2 * k - 1
        u = np.zeros((size, size), dtype=float)

        for i in range(k):
            for j in range(k):
                if i < k - 1:
                    u[2*i+1, 2*j] = euclidean_distance(som.weights[i, j], som.weights[i+1, j])
                if j < k - 1:
                    u[2*i, 2*j+1] = euclidean_distance(som.weights[i, j], som.weights[i, j+1])
                if i < k - 1 and j < k - 1:
                    u[2*i+1, 2*j+1] = (
                        euclidean_distance(som.weights[i,   j],   som.weights[i+1, j+1]) +
                        euclidean_distance(som.weights[i+1, j],   som.weights[i,   j+1])
                    ) * 0.5

        for i in range(k):
            for j in range(k):
                neighbours = []
                if i > 0:
                    neighbours.append(u[2*i-1, 2*j])
                if i < k-1:
                    neighbours.append(u[2*i+1, 2*j])
                if j > 0:
                    neighbours.append(u[2*i,   2*j-1])
                if j < k-1:
                    neighbours.append(u[2*i,   2*j+1])
                u[2*i, 2*j] = np.mean(neighbours) if neighbours else 0.0

        self.umatriz = u

    def cluster_purity(self, true_labels: np.ndarray) -> float:
        """Cluster purity: fraction of samples assigned to their neuron's majority class.

        Each neuron is labelled with the most frequent true class among its assigned
        samples (majority vote).  Purity is the proportion of samples whose true label
        matches their neuron's majority label.

        :param true_labels: 1-D array of ground-truth class labels, one per sample,
                            in the same order as the data passed to Classification.
        :return: purity in [0, 1]; higher is better.
        """
        true_labels = np.asarray(true_labels)
        if true_labels.shape[0] != self.classification_data.shape[0]:
            raise ValueError(
                f"true_labels length ({true_labels.shape[0]}) must match the number "
                f"of classified samples ({self.classification_data.shape[0]})."
            )

        bmu_flat = (self.classification_map['x'].to_numpy(dtype=int) * self._k
                    + self.classification_map['y'].to_numpy(dtype=int))

        neuron_labels: dict = {}
        for b, lbl in zip(bmu_flat, true_labels):
            neuron_labels.setdefault(b, []).append(lbl)

        majority = {b: int(_scipy_mode(v, keepdims=False).mode)
                    for b, v in neuron_labels.items()}

        correct = sum(majority.get(b, -1) == lbl
                      for b, lbl in zip(bmu_flat, true_labels))
        return correct / len(true_labels)
