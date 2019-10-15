import os
import glob
from itertools import combinations
import numpy as np
import mdtraj as mdt
import hdbscan
from bayesaxs.bayesScatter import Base


def _get_cluster_metric(metric):
	""" Get metrics for clustering."""
	cluster_metrics = {"xyz": _cluster_XYZ,
				"distances": _cluster_distances,
				"DRID": _cluster_drid}
	return cluster_metrics[metric]


def _cluster_XYZ(traj, atom_selection):
	""" Prepare XYZ coordinates of a trajectory for clustering."""
	temp = traj.xyz[:, atom_selection]
	frames = temp.shape[0]
	atoms = temp.shape[1]
	reshaped_XYZ = temp.reshape((frames, atoms * 3))
	reshaped_XYZ = reshaped_XYZ.astype("float64")
	temp = []
	return reshaped_XYZ


def _cluster_distances(traj, atom_selection):
	""" Calculate pair-wise atom distances of a trajectory for clustering."""
	atom_pairs = list(combinations(atom_selection, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)
	return pairwise_distances


def _cluster_drid(traj, atom_selection):
	""" Calulate DRID representation of a trajectory for clustering."""
	drid_distances = mdt.compute_drid(traj=traj, atom_indices=atom_selection)
	return drid_distances


class Trajectory(Base):

	def __init__(self):
		Base.__init__(self)
		self._pdb = None
		self._traj = None

	def __repr__(self):
		return "Trajectory: {}".format(self._traj)

	def load_traj(self, pdb_path, traj_path, stride=1):
		""" Load a trajectory."""
		self._pdb = mdt.load_pdb(pdb_path)
		self._traj = mdt.load(traj_path, top=pdb_path, stride=stride)

	def get_pdb(self):
		""" Return loaded .pdb topology."""
		return self._pdb

	def get_traj(self):
		"""Return loaded trajectory."""
		return self._traj


class BaseCluster(Trajectory):

	def __init__(self):
		Trajectory.__init__(self)
		self._cwdir = os.path.join(os.getcwd(), '')
		self._cluster_labels = None
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._leader_set = None

	def get_cluster_labels(self):
		""" Return cluster labels for each frame in a trajectory."""
		return self._cluster_labels

	def save_traj_clusters(self):
		"""Save each cluster as .xtc trajectory."""

		# Create a directory where to put cluster trajectories
		self._traj_cluster_dir = os.path.join(self._cwdir, self._title + "_traj_clusters", '')
		Base._mkdir(self._traj_cluster_dir)

		#  Extract clusters into .xtc trajectories
		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			self._traj[self._cluster_labels == cluster].save_xtc(filename=self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")

	def get_path_to_traj_clusters(self):
		""" Get path to trajectory clusters."""
		return self._traj_cluster_dir

	@staticmethod
	def _extract_leader(top, traj, trajnum, output_dir):
		"""
		Extract a representative conformer from a given single cluster trajectory. """

		# Load the trajectory
		traj = mdt.load(traj, top=top, stride=1)
		nframes = traj.n_frames

		# Calculate pairwise RMSD between frames
		rmsd_matrix = np.zeros((nframes, nframes))
		for i in range(nframes):
			rmsd_matrix[i:i + 1, :] = mdt.rmsd(target=traj, reference=traj, frame=i, parallel=True)

		# Calculate the sum along each and get leader index on the smallest number
		rmsd_sum = np.sum(rmsd_matrix, axis=1)
		leader_index = np.where(rmsd_sum == rmsd_sum.min())[0][0]

		# Save the leader as a PDB
		traj[leader_index].save_pdb(filename=output_dir + "cluster_leader_" + str(trajnum) + ".pdb")

	def save_cluster_leaders(self):
		"""Save each cluster leader as .pdb from cluster trajectories."""

		# Create a directory where to put cluster leaders extracted from cluster trajectories
		self._cluster_leader_dir = os.path.join(self._cwdir, self._title + "_cluster_leaders", '')
		Base._mkdir(self._cluster_leader_dir)

		# Extract a representative conformer from a given cluster trajectory. Skip HDBSCAN noise assignment (cluster -1)
		for cluster in range(self._cluster_labels.min() + 1, self._cluster_labels.max() + 1):
			BaseCluster._extract_leader(top=self._pdb,
							traj=(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
							trajnum=cluster,
							output_dir=self._cluster_leader_dir)

		# Initialize cluster leaders PDBs
		self._leader_set = glob.glob((self._cluster_leader_dir + "*"))

	def get_path_to_cluster_leaders(self):
		""" Get path to cluster leader directory."""
		return self._cluster_leader_dir

	def get_cluster_leaders(self):
		""" Get cluster leaders."""
		return self._leader_set


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
