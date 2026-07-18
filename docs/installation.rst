Installation
============

Requirements
------------

* Python 3.9 or later
* numpy, pandas, matplotlib, plotly, tqdm, scikit-learn, scipy (installed automatically)

Optional:

* **librosa** — required only for MFCC feature extraction

From PyPI
---------

.. code-block:: bash

   pip install amber-som

   # with optional MFCC support
   pip install "amber-som[mfcc]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/albertonogales/AMBER.git
   cd AMBER
   pip install -e ".[dev]"

Running the tests
-----------------

.. code-block:: bash

   pytest          # 374 tests, ~99 % coverage
