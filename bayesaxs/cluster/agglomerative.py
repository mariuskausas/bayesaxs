import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from bayesaxs.cluster.base import BaseCluster
from bayesaxs.cluster.base import _compute_pairwise_rmsd


class Agglomerative(BaseCluster):
    """
    Agglomerative clustering object.

    The Agglomerative object allows to perform a clustering on a given trajectory.
    """

    def __init__(self):
        """ Instantiate a new Agglomerative object."""

        BaseCluster.__init__(self)
        self._rmsd_distances = None
        self._linkage_matrix = None
        self._linkage_cutoff = None

    def get_rmsd_distances(self):
        """
        Get pairwise RMSD between conformations.

        Returns
        -------
        out : ndarray
            Pairwise RMSD between conformations.
        """

        return self._rmsd_distances

    def get_linkage_matrix(self):
        """
        Get clustering linkage matrix.

        Returns
        -------
        out : ndarray
            The hierarchical clustering encoded as a linkage matrix.
        """

        return self._linkage_matrix

    def get_linkage_cutoff(self):
        """
        Get agglomerative clustering linkage cutoff.

        Returns
        -------
        out : float
            Cutoff value for selecting a number of clusters.
        """

        return self._linkage_cutoff

    def fit_predict(self, atom_selection, method="ward", metric="euclidean", cutoff_value=0.25):
        """
        Predict cluster labels using Agglomerative clustering.

        Parameters
        ----------
        atom_selection : ndarray
            Numpy array (N, ) containing indices of atoms.
        method : str
            Linkage algorithm to use.
        metric : str
            The metric to use when calculating distance between instances in a feature array.
        cutoff_value : float
            Cutoff value for selecting a number of clusters.
        """

        # Compute pairwise RMSD values between conformations
        self._rmsd_distances = _compute_pairwise_rmsd(traj=self.get_traj(),
                                                      atom_selection=atom_selection)
        # Convert square-form distance matrix into a vector one
        reduced_distances = squareform(self._rmsd_distances, checks=False)
        # Perform hierarchical clustering
        self._linkage_matrix = sch.linkage(reduced_distances,
                                           method=method,
                                           metric=metric)
        # Compute cutoff value for hierarchical clustering
        self._linkage_cutoff = cutoff_value * max(self._linkage_matrix[:, 2])
        # Form flat clusters from the hierarchical clustering
        self._cluster_labels = sch.fcluster(self._linkage_matrix,
                                            t=self._linkage_cutoff,
                                            criterion='distance')

        return
