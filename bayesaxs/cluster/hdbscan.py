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
	kwargs :
		Rest of other parameters for setting dbscan.hdbscan_.HDBSCAN object.

	Attributes
	----------
	clusterer : hdbscan.hdbscan_.HDBSCAN object
		Clustering object initialized using HDBSCAN.
	"""

	def __init__(self, min_cluster_size=5, metric="euclidean", core_dist_n_jobs=-1, **kwargs):
		""" Instantiate a new HDBSCAN object."""

		BaseCluster.__init__(self)
		self._clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
										metric=metric,
										core_dist_n_jobs=core_dist_n_jobs,
										**kwargs)

	def get_clusterer(self):
		"""
		Get initialized HDBSCAN object.

		Returns
		-------
		out : hdbscan.hdbscan_.HDBSCAN object
			HDBSCAN clustering object.
		"""

		return self._clusterer

	def fit_predict(self, metric, atom_selection):
		"""
		Predict cluster labels using HDBSCAN.

		You can select from the following metrics to cluster on:
			1) xyz - cluster using xyz coordinates.
			2) distances - cluster on pairwise inter-atom distances.
			3) DRID - cluster using DRID distances.

		The distances and DRID are calculated using mdtraj.

		For effective clustering using XYZ coordinates,
		rotational and translational movements of a trajectory
		should be removed prior.

		Parameters
		----------
		metric : str
			Available options ("xyz", "distances", "DRID").
        atom_selection : ndarray
            Numpy array (N, ) containing indices of atoms, which will be used for clustering.
		"""

		# Transform trajectory based on a clustering metric
		cluster_input = _get_cluster_metric(metric)(traj=self._traj, atom_selection=atom_selection)
		# Predict cluster labels
		self._cluster_labels = self._clusterer.fit_predict(cluster_input)

		return
