Getting started
===============

Install AMBER
-------------

**Requirements:** Python 3.9 or later.

.. code-block:: bash

   pip install amber-som

This installs AMBER together with its required dependencies:
NumPy, pandas, matplotlib, Plotly, tqdm, and scikit-learn.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Package
     - Install
     - Required for
   * - scipy
     - ``pip install scipy``
     - Welch PSD for spectral features; improved skewness/kurtosis
   * - librosa
     - ``pip install librosa``
     - MFCC feature extraction

Install with extras in one command:

.. code-block:: bash

   pip install "amber-som[spectral]"      # adds scipy
   pip install "amber-som[mfcc]"          # adds librosa
   pip install "amber-som[spectral,mfcc]" # both

Install from source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/albertonogales/AMBER.git
   cd AMBER
   pip install -e ".[dev]"   # editable install with test dependencies

Verify the installation
-----------------------

.. code-block:: python

   import AMBER
   print(AMBER.__version__)   # e.g. 2.0.0
   print(AMBER.__all__)

Five-line quick start
---------------------

.. code-block:: python

   import numpy as np
   import AMBER

   data = np.random.rand(200, 8)
   som  = AMBER.Map(data=data, period=100)          # map size chosen automatically
   cls  = AMBER.Classification(som, data)
   print(f"Quantisation error : {cls.quantization_error:.4f}")
   print(f"Topological error  : {cls.topological_error:.4f}")
   AMBER.Visualization.heat_map(cls)

Running the tests
-----------------

.. code-block:: bash

   pip install -r requirements-dev.txt
   pytest        # 372 tests, ~99 % coverage

A coverage HTML report is written to ``coverage_html/``.
