Quick start
===========

Example notebooks
-----------------

Ready-to-run Jupyter notebooks are provided in the ``examples/`` folder:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - What it demonstrates
   * - ``colors_example.ipynb``
     - Basic SOM on synthetic RGB colour data
   * - ``eeg_som.ipynb``
     - EEG sleep-stage classification with band-power features and TemporalMap (Sleep-EDF / synthetic)
   * - ``audio_som.ipynb``
     - Instrument and speech clustering with MFCCs and raw-waveform DTW (NSynth / FSDD / synthetic)
   * - ``climate_som.ipynb``
     - Weather-regime discovery from raw gridded climate fields (ERA5 / synthetic)
   * - ``predictive_maintenance_som.ipynb``
     - Bearing degradation tracking from raw vibration windows and hand-crafted features (CWRU / synthetic)
   * - ``ablation_study.ipynb``
     - Systematic comparison of all 7 distances, 9 normalisations, context parameters, map-size heuristic, and weight initialisations

---

Basic SOM
---------

.. code-block:: python

   import numpy as np
   import AMBER

   data = np.random.rand(200, 8)

   # Map size is chosen automatically via Vesanto & Alhoniemi (2000)
   som = AMBER.Map(data=data, period=100)
   print(f"Map: {som.map_size}×{som.map_size}")

   cls = AMBER.Classification(som, data)
   print(f"Quantisation error : {cls.quantization_error:.4f}")
   print(f"Topological error  : {cls.topological_error:.4f}")

   AMBER.Visualization.heat_map(cls)
   AMBER.Visualization.umatrix(cls)

Custom distance and normalisation
----------------------------------

.. code-block:: python

   som = AMBER.Map(
       data=data,
       size=12,
       period=200,
       distance='dtw',           # Dynamic Time Warping
       normalization='zscore',
       weights='PCA',
       use_decay=True,
   )

Available distances: ``euclidean``, ``manhattan``, ``chebyshev``, ``cosine``,
``correlation``, ``dtw``, ``cross_correlation``.

Save and load
-------------

.. code-block:: python

   som.save_classifier('my_model')          # writes my_model.json
   loaded = AMBER.Map.load_classifier('my_model')

Temporal / Recurrent SOM
-------------------------

.. code-block:: python

   tsom = AMBER.TemporalMap(
       data=eeg_windows,          # (n_samples, n_features)
       period=100,
       context_weight=0.5,        # α — context memory decay
       context_influence=0.3,     # β — context vs. signal balance
   )

   cls = AMBER.Classification(tsom, eeg_windows)
   ta  = AMBER.TemporalAnalysis(cls)
   print(ta.summary())

   AMBER.Visualization.trajectory(cls, ta)
   AMBER.Visualization.transition_matrix_plot(ta)
   AMBER.Visualization.dwell_time_map(ta, cls)

Feature extraction for biosignals
----------------------------------

.. code-block:: python

   from AMBER import FeatureExtractor

   fe = FeatureExtractor(fs=256)       # 256 Hz sampling rate

   # Single window → 1-D feature vector
   feats = fe.extract(window, features=[
       'rms', 'alpha_power', 'beta_power',
       'spectral_entropy', 'hjorth_mobility',
   ])

   # Batch → (n_windows, n_features) array ready for Map.train
   X = fe.extract_batch(windows, features=['rms', 'alpha_power'])

   # Feature names in the same order as extract()
   names = fe.feature_names(['rms', 'alpha_power'])

Iterative map-size selection
-----------------------------

.. code-block:: python

   isom = AMBER.IterativeSOM(data=data, period=50)
   best = isom.best_map()
   print(f"Best size: {best.map_size}")
