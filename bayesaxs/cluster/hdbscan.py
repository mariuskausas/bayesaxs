import hdbscan

from bayesaxs.cluster.base import BaseCluster
from bayesaxs.cluster.base import _get_cluster_metric


class HDBSCAN(BaseCluster):

	def __init__(self, min_cluster_size=5, metric="euclidean", core_dist_n_jobs=-1, **kwargs):
		BaseCluster.__init__(self)
		self._clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
										metric=metric,
										core_dist_n_jobs=core_dist_n_jobs, **kwargs)

	def fit_predict(self, metric="xyz", **kwargs):
		"""Perform HDBSCAN clustering."""

		# Transform trajectory based on a clustering metric
		cluster_input = _get_cluster_metric(metric)(traj=self._traj, **kwargs)
		self._cluster_labels = self._clusterer.fit_predict(cluster_input)
