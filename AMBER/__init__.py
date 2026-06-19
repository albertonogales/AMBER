"""AMBER — Self-Organizing Maps for Python.

Core classes
------------
Map               : Standard SOM
Classification    : BMU assignment, activation map, U-matrix, error metrics
TemporalMap       : Recurrent SOM (RSOM, Voegtlin 2002)
TemporalAnalysis  : Temporal metrics (transition matrix, stability, dwell times)
FeatureExtractor  : Feature extraction for time series / biosignals
IterativeSOM      : Grid-search over map sizes
Visualization     : Heat maps, U-matrix, trajectory plots, transition heatmaps

Convenience exports
-------------------
vesanto_size       : Vesanto & Alhoniemi (2000) map-size heuristic
AVAILABLE_DISTANCES: list of supported BMU distance names
"""

from .map import Map, vesanto_size
from .classification import Classification
from .visualization import Visualization
from .temporal_map import TemporalMap
from .temporal_analysis import TemporalAnalysis
from .features import FeatureExtractor
from .distances import AVAILABLE_DISTANCES
from .iterativesom import IterativeSOM

__version__ = "2.1.0"
__author__ = "Alberto Nogales, Álvaro José García-Tejedor"
__email__ = "alberto.nogales@uah.es"
__license__ = "GPL-3.0"

__all__ = [
    "Map",
    "Classification",
    "Visualization",
    "TemporalMap",
    "TemporalAnalysis",
    "FeatureExtractor",
    "IterativeSOM",
    "vesanto_size",
    "AVAILABLE_DISTANCES",
]
