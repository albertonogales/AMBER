Installation
============

Requirements
------------

* Python 3.8 or later
* numpy, pandas, matplotlib, plotly, tqdm, scikit-learn (installed automatically)

Optional:

* **scipy** — improves accuracy of spectral features (Welch PSD) and skewness/kurtosis
* **librosa** — required only for MFCC feature extraction

From PyPI
---------

.. code-block:: bash

   pip install amber-som

   # with optional spectral / MFCC support
   pip install "amber-som[spectral]"
   pip install "amber-som[mfcc]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/ufvceiec/AMBER.git
   cd AMBER
   pip install -e ".[dev]"

Running the tests
-----------------

.. code-block:: bash

   pytest          # 265 tests, ~95 % coverage
