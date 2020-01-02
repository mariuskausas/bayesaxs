import os
import glob
from itertools import combinations

import numpy as np
import mdtraj as mdt

from bayesaxs.base import Base


def _get_cluster_metric(metric):
	""" Get a clustering metric.

	Following metrics are available:
		1) xyz - cluster using xyz coordinates.
		2) distances - cluster on pairwise inter-atom distances.
		3) DRID - cluster using DRID distances.

	For effective clustering using xyz coordinates,
	a rotational and translational movements of a trajectory
	should be removed prior.

	Returns
	-------
	out : function
		Clustering metric function.
	"""

	cluster_metrics = {"xyz": _cluster_xyz,
				"distances": _cluster_distances,
				"DRID": _cluster_drid}

	return cluster_metrics[metric]


def _cluster_xyz(traj, atom_selection):
	""" Prepare input of XYZ coordinates for clustering.

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : array
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	reshaped_xyz : array
		Numpy array (frames, atoms * 3) of reshaped XYZ coordinates.
	"""

	temp = traj.xyz[:, atom_selection]
	frames = temp.shape[0]
	atoms = temp.shape[1]
	reshaped_xyz = temp.reshape((frames, atoms * 3))
	reshaped_xyz = reshaped_xyz.astype("float64")

	return reshaped_xyz


def _cluster_distances(traj, atom_selection):
	""" Calculate pair-wise atom distances of a trajectory for clustering.

	Pair-wise distances are calculated using mdtraj.compute_distances().

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : array
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	pairwise_distances : array
		Numpy array (M, N). M equals number of frames.
		N equals 2 chooses k, where k is the number of atoms.
	"""

	atom_pairs = list(combinations(atom_selection, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)

	return pairwise_distances


def _cluster_drid(traj, atom_selection):
	""" Calulate DRID representation of a trajectory for clustering.

	DRID distances are calculated using mdtraj.compute_drid().

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : array
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	drid_distances : array
		Numpy array (M, N). M equals number of frames.
		N equals number of computed DRID distances.

	"""

	drid_distances = mdt.compute_drid(traj=traj, atom_indices=atom_selection)

	return drid_distances


class Trajectory(Base):
	""" Basic container for molecular dynamics trajectory.

	The Trajectory object allows loading trajectory and inspecting
	loaded topology and trajectory file.

	Attributes
	----------
	pdb : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj .pdb topology object.
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	"""

	def __init__(self):
		""" Create a new Trajectory object."""

		Base.__init__(self)
		self._pdb = None
		self._traj = None

	def __repr__(self):
		return "Trajectory: {}".format(self._traj)

	def load_traj(self, pdb_path, traj_path, stride=1):
		""" Load a molecular trajectory.

		Parameters
		----------
		pdb_path : str
			Path to topology .pdb file.
		traj_path : str
			Path to trajectory (mdtraj supported extensions).
		stride : int
			Skip through a trajectory. Default set to 1.
		"""

		self._pdb = mdt.load_pdb(pdb_path)
		self._traj = mdt.load(traj_path, top=pdb_path, stride=stride)

		return

	def get_pdb(self):
		""" Return loaded .pdb topology.

		Returns
		-------
		pdb : mdtraj.core.trajectory.Trajectory object
			Loaded mdtraj .pdb topology object.
		"""

		return self._pdb

	def get_traj(self):
		"""Return loaded trajectory.

		Returns
		-------
		traj : mdtraj.core.trajectory.Trajectory object
			Loaded mdtraj trajectory.
		"""

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
