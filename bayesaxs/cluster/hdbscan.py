import hdbscan

from bayesaxs.cluster.base import BaseCluster
from bayesaxs.cluster.base import _get_cluster_metric


class HDBSCAN(BaseCluster):
	"""
	HDBSCAN clustering object.

	The HDBSCAN object allows to perform a clustering on a given trajectory.

	Parameters
	----------
	min_cluster_size : int
		The minimum size of clusters.
	metric : str
		The metric to use when calculating distance between instances in a feature array.
	core_dist_n_jobs : int
		Number of parallel jobs to run in core distance computations.
		For core_dist_n_jobs below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

	Attributes
	----------
	clusterer : hdbscan.hdbscan_.HDBSCAN object
		Clustering object initialized using HDBSCAN.
	"""

	def __init__(self, min_cluster_size=5, metric="euclidean", core_dist_n_jobs=-1, **kwargs):
		""" Create a new HDBSCAN object."""

		BaseCluster.__init__(self)
		self._clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
										metric=metric,
										core_dist_n_jobs=core_dist_n_jobs, **kwargs)

	def fit_predict(self, metric="xyz", **kwargs):
		"""
		Predict cluster labels using HDBSCAN.

		You can select from the following metrics to cluster on:
			1) xyz - cluster using xyz coordinates.
			2) distances - cluster on pairwise inter-atom distances.
			3) DRID - cluster using DRID distances.

		The distances and DRID are calculated using mdtraj.

		For effective clustering using xyz coordinates,
		a rotational and translational movements of a trajectory
		should be removed prior.

		Parameters
		----------
		metric : str
			Available options ("xyz", "distances", "DRID").
			The "xyz" option is set by default.
		kwargs : str
			Additional parameters could be passed to mdtraj.compute_distances()
			or mdtraj.compute_drid() functions.
		"""

		# Transform trajectory based on a clustering metric
		cluster_input = _get_cluster_metric(metric)(traj=self._traj, **kwargs)
		self._cluster_labels = self._clusterer.fit_predict(cluster_input)

		return
