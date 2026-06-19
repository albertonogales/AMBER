from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .distances import SIGNAL_DISTANCE_SCALAR, euclidean_distance
from .map import Map

logger = logging.getLogger(__name__)


class Classification:
    """
    Clasification class. Holds information about what has been classified.
    """
    def __init__(self, som: Map, classification_data: np.ndarray,
                 other: Optional[pd.DataFrame] = None,
                 tagged: bool = False,
                 verbose: bool = False) -> None:
        """Creates and classifies some data on top of the som map

        :param som: Map instance which will be responsible of the classification
        :param classification_data: 2-D numpy array of samples to classify
        :param other: optional extra DataFrame to concatenate with classification_map
        :param tagged: if True, first column of classification_data is treated as labels
        :param verbose: if True, log debug info about labels and data
        """
        pd.options.mode.chained_assignment = None  # default='warn'

        # If the input data is tagged, keep all the tags; if not, create them
        if tagged:
            self.classification_labels = classification_data[:, 0]
            self.classification_data = classification_data[:, 1:]
        else:
            self.classification_data = classification_data
            self.classification_labels = np.arange(classification_data.shape[0])

        if verbose:
            logger.debug("\n\nTags: \n" + str(self.classification_labels))
            logger.debug("\n\nClassification data: \n" + str(self.classification_data))

        # Declaration and initialization
        self.activations_map = np.zeros((som.map_size, som.map_size), dtype=int)
        self.distances_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.topological_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.umatriz = np.zeros((som.map_size * 2 - 1, som.map_size * 2 - 1), dtype=float)
        self.topological_error = 0
        self.quantization_error = 0           # configured distance (primary)
        self.quantization_error_euclidean = 0 # always Euclidean (for cross-library comparison)
        self.distortion = 0
        self.topological_error_map = np.zeros((som.map_size, som.map_size), dtype=float)
        self.quantization_error_map = np.zeros((som.map_size, som.map_size), dtype=float)

        # Store bmu positions for distortion computation (filled in the loop below)
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

        # Scalar distance function matching the map's training metric
        scalar_dist_fn = SIGNAL_DISTANCE_SCALAR[som.distance]
        dtw_kwargs = {'band': som.dtw_band} if som.distance == 'dtw' else {}

        n_samples = self.classification_data.shape[0]

        # Apply the same normalisation used during training so that BMU search
        # operates in the same feature space as the trained weights.
        norm_data = som.transform(self.classification_data)

        # Input all the patterns
        for pattern in tqdm(range(0, n_samples)):
            # Getting the BMU neuron
            bmu, bmu_pos, second_bmu, second_bmu_pos = som.calculate_bmu(norm_data[pattern])

            # Topological error: BMU and 2nd-BMU are not adjacent (Chebyshev distance > 1 covers all 8 neighbours)
            if np.max(np.abs(np.array(bmu_pos) - np.array(second_bmu_pos))) > 1:
                self.topological_map[bmu_pos] += 1

            # Distance measured with the map's own configured metric (primary)
            distance = scalar_dist_fn(som.weights[bmu_pos], norm_data[pattern],
                                      **dtw_kwargs)  # type: ignore[operator]
            self.activations_map[bmu_pos] += 1
            self.distances_map[bmu_pos] += distance
            bmu_positions[pattern] = bmu_pos

            self.classification_map.loc[pattern, 'x']    = bmu_pos[0]
            self.classification_map.loc[pattern, 'y']    = bmu_pos[1]
            self.classification_map.loc[pattern, 'dist'] = distance

        # Number of neurons that have identified pattern
        self.num_activations = np.count_nonzero(self.activations_map != 0)

        # Mean distance per sample (consistent with quantization_error denominator)
        self.mean_distance_map = np.sum(self.distances_map) / n_samples

        # Decreasing the number of decimal places to 5
        self.distances_map = np.around(self.distances_map, decimals=5)

        # Calculating topological error and map
        # TE = fraction of samples whose BMU and 2nd-BMU are non-adjacent (Villmann 1997)
        self.topological_error = np.sum(self.topological_map) / n_samples
        self.topological_error_map = np.divide(self.topological_map, self.activations_map,
                                               where=self.activations_map != 0)

        # Primary QE: uses the map's configured distance — semantically correct.
        # distances_map was accumulated with scalar_dist_fn, so this is consistent
        # with the metric used during both training and BMU search.
        self.quantization_error = np.sum(self.distances_map) / n_samples
        self.quantization_error_map = np.divide(self.distances_map, self.activations_map,
                                                where=self.activations_map != 0)

        # Secondary QE: always Euclidean, vectorised. (N, d) - (N, d) → (N,)
        bmu_weights = som.weights[bmu_positions[:, 0], bmu_positions[:, 1], :]  # (N, d)
        eucl_total = float(np.sum(np.sqrt(np.sum((bmu_weights - norm_data)**2, axis=-1))))
        self.quantization_error_euclidean = eucl_total / n_samples

        # Distortion measure (Graepel et al. 1997 / Heskes 1999)
        # D = (1/N) Σᵢ Σⱼ h_σ(BMU(xᵢ), j) · ||xᵢ − wⱼ||²
        # where h_σ is a Gaussian neighbourhood with σ = initial neighbourhood radius.
        sigma = max(float(som.neighbourhood), 1.0)
        k = som.map_size
        # Build grid-position index array once: (k, k, 2)
        grid_idx = np.array([[[i, j] for j in range(k)] for i in range(k)],
                            dtype=float)

        # Vectorised distortion: bmu_positions (N,2), grid_idx (k,k,2)
        bmu_r = bmu_positions.astype(float)  # (N, 2)
        # sq_dist_grid: (N, k, k)
        sq_dist_grid = np.sum(
            (grid_idx[np.newaxis] - bmu_r[:, np.newaxis, np.newaxis, :]) ** 2,
            axis=-1
        )
        h = np.exp(-sq_dist_grid / (2.0 * sigma ** 2))  # (N, k, k)
        # sq_dist_w: (N, k, k)
        diff = som.weights[np.newaxis] - norm_data[:, np.newaxis, np.newaxis, :]
        sq_dist_w = np.sum(diff ** 2, axis=-1)
        self.distortion = float(np.sum(h * sq_dist_w)) / n_samples

        # U-Matrix (Ultsch & Siemon 1990) — full (2k-1)×(2k-1) representation
        # Even indices (2i, 2j): neuron cells — filled with mean of adjacent edge distances.
        # Odd indices: inter-neuron edge distances.
        #   (2i+1, 2j)   : horizontal edge between neuron (i,j) and (i+1,j)
        #   (2i,   2j+1) : vertical   edge between neuron (i,j) and (i,j+1)
        #   (2i+1, 2j+1) : diagonal   average of the two crossing diagonals
        size = 2 * k - 1
        u = np.zeros((size, size), dtype=float)

        for i in range(k):
            for j in range(k):
                # Horizontal edge →
                if i < k - 1:
                    u[2*i+1, 2*j] = euclidean_distance(som.weights[i, j], som.weights[i+1, j])
                # Vertical edge ↓
                if j < k - 1:
                    u[2*i, 2*j+1] = euclidean_distance(som.weights[i, j], som.weights[i, j+1])
                # Diagonal cell (mean of two crossing diagonal distances)
                if i < k - 1 and j < k - 1:
                    u[2*i+1, 2*j+1] = (
                        euclidean_distance(som.weights[i,   j],   som.weights[i+1, j+1]) +
                        euclidean_distance(som.weights[i+1, j],   som.weights[i,   j+1])
                    ) * 0.5

        # Fill neuron cells with the mean of their adjacent edge distances
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
