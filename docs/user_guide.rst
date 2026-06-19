User guide
==========

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

AMBER covers the full SOM workflow in six steps:

.. code-block:: text

   Raw data
      │
      ▼
   Normalisation  ──  Map.train()
      │
      ▼
   Classification  ──  BMU assignment, activation map, U-matrix,
      │                quantisation error, topological error
      ▼
   Temporal analysis  ──  TemporalMap + TemporalAnalysis
      │                   transition matrix, dwell times, trajectory
      ▼
   Visualisation  ──  heat map, U-matrix, trajectory, transition heatmap
      │
      ▼
   Save / load  ──  Map.save_classifier() / Map.load_classifier()


Training a SOM
--------------

Basic usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import AMBER

   data = np.random.rand(300, 10)   # 300 samples, 10 features each

   som = AMBER.Map(data=data, period=200)
   # Map size is chosen automatically by the Vesanto & Alhoniemi (2000) heuristic:
   # size = max(2, round(sqrt(5 * sqrt(N))))
   print(f"Map size: {som.map_size}×{som.map_size}")

Choosing the map size manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   som = AMBER.Map(data=data, size=10, period=200)

The ``vesanto_size`` helper is also available directly:

.. code-block:: python

   print(AMBER.vesanto_size(300))   # recommended side length for 300 samples

Distance metrics
~~~~~~~~~~~~~~~~

Seven signal distance metrics are available for BMU selection:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - When to use
   * - ``euclidean`` *(default)*
     - General-purpose; fast; works well on normalised data
   * - ``manhattan``
     - Robust to outliers in individual dimensions
   * - ``chebyshev``
     - Sensitive to the single largest dimension difference
   * - ``cosine``
     - Pattern shape matters more than magnitude (e.g. text, gene expression)
   * - ``correlation``
     - Amplitude-invariant; standard in climatology and EEG
   * - ``dtw``
     - Time-series with phase/speed variation (speech, vibration, ECG)
   * - ``cross_correlation``
     - Shift-invariant similarity between periodic signals

.. code-block:: python

   som = AMBER.Map(data=data, period=200, distance='dtw')
   print(AMBER.AVAILABLE_DISTANCES)   # full list

Normalisation strategies
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Effect
   * - ``none``
     - No normalisation
   * - ``zscore`` / ``fwn``
     - Per-feature zero mean, unit variance
   * - ``robust``
     - Per-feature median/IQR scaling — robust to outliers
   * - ``01scale``
     - Per-feature min–max scaling to [0, 1]
   * - ``l2`` / ``euclidean``
     - Per-sample unit L2 norm
   * - ``zscore_sample``
     - Per-sample zero mean, unit variance
   * - ``robust_sample``
     - Per-sample median/IQR scaling
   * - ``minmax_sample``
     - Per-sample min–max scaling to [0, 1]

.. code-block:: python

   som = AMBER.Map(data=data, period=200,
                   distance='correlation',
                   normalization='robust')

Weight initialisation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   som = AMBER.Map(data=data, period=200, weights='PCA')
   # options: 'random', 'random_negative', 'sample', 'PCA'

Other training options
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   som = AMBER.Map(
       data=data,
       period=500,
       initial_lr=0.5,            # starting learning rate (default 0.5)
       initial_neighbourhood=0,   # 0 = auto-set to map_size
       use_decay=True,            # cosine decay of lr and neighbourhood
       distance='euclidean',
       normalization='zscore',
       weights='PCA',
   )

Classification
--------------

.. code-block:: python

   cls = AMBER.Classification(som, data)

   # Quality metrics
   print(cls.quantization_error)   # mean distance between samples and BMUs
   print(cls.topological_error)    # fraction of samples whose 2nd-BMU is not adjacent

   # Per-sample BMU positions: DataFrame with columns x, y, bmu_distance
   print(cls.classification_map.head())

   # Activation map: (map_size, map_size) count of BMU hits
   print(cls.activations_map)

   # U-matrix: (2k-1, 2k-1) inter-neuron distance grid
   print(cls.umatriz.shape)

Labelled classification
~~~~~~~~~~~~~~~~~~~~~~~

If your data has labels, pass them to get per-neuron majority-vote assignment:

.. code-block:: python

   labels = np.array(['A', 'B', 'A', ...])   # one per sample
   cls    = AMBER.Classification(som, data, labels)


Temporal / Recurrent SOM
-------------------------

``TemporalMap`` extends ``Map`` with a context vector (Voegtlin 2002 RSOM):

.. math::

   c_t = \alpha \cdot c_{t-1} + (1 - \alpha) \cdot w_{\text{BMU}}

   d_{\text{eff}} = (1 - \beta) \cdot d_{\text{signal}} + \beta \cdot d_{\text{context}}

where α is ``context_weight`` (memory decay) and β is ``context_influence``
(balance between current signal and accumulated context).

.. code-block:: python

   tsom = AMBER.TemporalMap(
       data=data,
       period=200,
       context_weight=0.6,      # α — how much history is retained
       context_influence=0.3,   # β — how much context affects BMU selection
       distance='euclidean',
       normalization='zscore',
   )

   tcls = AMBER.Classification(tsom, data)
   ta   = AMBER.TemporalAnalysis(tcls)

   print(ta.summary())
   # stability          — fraction of consecutive identical BMUs
   # mean_path_length   — mean Euclidean grid distance per step
   # transition_matrix  — (k², k²) raw transition counts
   # dwell_times()      — dict of {(row, col): mean_dwell}

   # Most frequent transitions
   for (src, dst), count in ta.most_frequent_transitions(top_k=5):
       print(f"neuron {src} → {dst}  ({count} times)")

   # Reset context between independent sequences
   tsom.reset_context()


Feature extraction
------------------

.. code-block:: python

   from AMBER import FeatureExtractor

   fe = FeatureExtractor(fs=256)   # 256 Hz sampling rate

   # Single window → 1-D feature vector
   x = fe.extract(signal_window, features=[
       'rms', 'zero_crossing_rate',
       'alpha_power', 'beta_power', 'spectral_entropy',
       'hjorth_mobility', 'hjorth_complexity',
   ])

   # Batch → (n_windows, n_features) ready for Map.train
   X = fe.extract_batch(windows, features=['rms', 'kurtosis', 'spectral_entropy'])

   # Feature names in the same order as extract()
   names = fe.feature_names(['rms', 'kurtosis', 'spectral_entropy'])

Available features
~~~~~~~~~~~~~~~~~~

**Statistical** (no sampling frequency needed):
``mean``, ``std``, ``var``, ``skewness``, ``kurtosis``, ``rms``,
``peak_to_peak``, ``zero_crossing_rate``, ``line_length``

**Spectral** (``fs`` required):
``spectral_power``, ``dominant_frequency``, ``spectral_entropy``,
``spectral_centroid``, ``spectral_rolloff``,
``delta_power``, ``theta_power``, ``alpha_power``, ``beta_power``, ``gamma_power``

**Complexity** (no ``fs`` needed):
``hjorth_activity``, ``hjorth_mobility``, ``hjorth_complexity``, ``sample_entropy``

**Librosa** (``fs`` + librosa required):
``mfcc`` → returns ``n_mfcc`` values (mean of each coefficient over time)


Iterative map-size selection
----------------------------

``IterativeSOM`` trains maps across a range of sizes and returns the one with
the lowest topological error:

.. code-block:: python

   isom = AMBER.IterativeSOM(data=data, period=100)
   best = isom.best_map()
   print(f"Best size: {best.map_size}")

   # Inspect all sizes tested
   print(isom.calculate_range())


Visualisation
-------------

.. code-block:: python

   # Activation heat map
   AMBER.Visualization.heat_map(cls, colorscale='Reds')

   # U-matrix (cluster boundaries)
   AMBER.Visualization.umatrix(cls, colorscale='binary')

   # Elevation map (3-D surface)
   AMBER.Visualization.elevation_map(cls)

   # Weight profile for one neuron
   AMBER.Visualization.characteristics_graph(som, row=2, column=3)
   AMBER.Visualization.characteristics_bargraph(som, row=2, column=3,
                                                 labels=feature_names)

   # Codebook vectors (all dimensions)
   AMBER.Visualization.codebook_vectors(som, headers=feature_names)

   # Temporal plots
   AMBER.Visualization.trajectory(tcls, ta, background='activations')
   AMBER.Visualization.transition_matrix_plot(ta, normalised=True)
   AMBER.Visualization.dwell_time_map(ta, tcls)


Save and load
-------------

.. code-block:: python

   # Save to JSON
   som.save_classifier('my_model')        # writes my_model.json

   # Load back
   loaded = AMBER.Map.load_classifier('my_model')
   _, bmu_pos, _, _ = loaded.calculate_bmu(data[0])

   # TemporalMap save/load works the same way
   tsom.save_classifier('temporal_model')
   loaded_t = AMBER.TemporalMap.load_classifier('temporal_model')


Application domains and example notebooks
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Domain
     - Notebook
     - Key features used
   * - EEG / sleep staging
     - ``examples/eeg_som.ipynb``
     - Band powers, Hjorth, TemporalMap, forbidden-transition check
   * - Audio / instruments
     - ``examples/audio_som.ipynb``
     - MFCCs, intra vs. inter-class BMU distance, TemporalMap trajectory
   * - Climate regimes
     - ``examples/climate_som.ipynb``
     - Raw gridded fields, correlation distance, seasonal dwell time
   * - Predictive maintenance
     - ``examples/predictive_maintenance_som.ipynb``
     - Raw DTW windows, health index, feature comparison
   * - Ablation study
     - ``examples/ablation_study.ipynb``
     - All distances, normalisations, context sweep, Vesanto heuristic
