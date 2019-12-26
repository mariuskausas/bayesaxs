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

__all__ = ["basis", "cluster", "refinement", "viz"]
