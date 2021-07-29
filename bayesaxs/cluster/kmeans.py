from sklearn.cluster import KMeans as skl_kmeans

from bayesaxs.cluster.base import BaseCluster
from bayesaxs.cluster.base import _get_cluster_metric


class KMeans(BaseCluster):
    """
    KMeans clustering object.

    The KMeans object allows to perform a clustering on a given trajectory.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form
    init : str
        Method for initialization, e.g. k-means++ (default) or random.
    n_init : int
        NNumber of time the k-means algorithm will be run with different centroid seeds.
         The final results will be the best output of n_init consecutive runs in terms of inertia.
    max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
    kwargs :
        All other parameters for setting KMeans object.

    Attributes
    ----------
    clusterer : sklearn.cluster.KMeans object
        Clustering object initialized using sklearn KMeans.
    """

    def __init__(self, n_clusters=5, init="k-means++", n_init=10, max_iter=300, **kwargs):
        """ Instantiate a new KMeans object."""

        BaseCluster.__init__(self)
        self._clusterer = skl_kmeans(n_clusters=n_clusters,
                                    init=init,
                                    n_init=n_init,
                                    max_iter=max_iter,
                                    **kwargs)

    def get_clusterer(self):
        """
        Get initialized KMeans object.

        Returns
        -------
        out : sklearn.cluster.KMeans object
            KMeans clustering object.
        """

        return self._clusterer

    def fit_predict(self, metric, atom_selection):
        """
        Predict cluster labels using KMeans.

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
