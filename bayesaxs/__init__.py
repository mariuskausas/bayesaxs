"""
bayesaxs
Python implementation of BSS-SAXS
"""

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__all__ = ["bayesScatter", "bayesChi", "bayesCluster", "bayesModel", "bayesPlots"]

from .bayesScatter import Curve, Scatter
from .bayesCluster import HDBSCAN
from .bayesModel import BayesModel
from .bayesPlots import plot_heatmap, plot_multi_scatters, plot_single_scatter
