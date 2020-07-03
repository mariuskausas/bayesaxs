import os
import glob
from itertools import combinations

import numpy as np
import mdtraj as mdt
from scipy.spatial.distance import pdist, squareform

from bayesaxs.base import Base


def _cluster_xyz(traj, atom_selection):
	"""
	Prepare input of XYZ coordinates for clustering.

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : ndarray
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	reshaped_xyz : ndarray
		Numpy array (frames, atoms * 3) of reshaped XYZ coordinates.
	"""

	temp = traj.xyz[:, atom_selection]
	frames = temp.shape[0]
	atoms = temp.shape[1]
	reshaped_xyz = temp.reshape((frames, atoms * 3))
	reshaped_xyz = reshaped_xyz.astype("float64")

	return reshaped_xyz


def _cluster_distances(traj, atom_selection):
	"""
	Calculate pairwise atom distances of a trajectory for clustering.

	Pairwise distances are calculated using mdtraj.compute_distances().

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : ndarray
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	pairwise_distances : ndarray
		Numpy array (M, N). M equals number of frames.
		N equals 2 chooses k, where k is the number of atoms.
	"""

	atom_pairs = list(combinations(atom_selection, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)

	return pairwise_distances


def _cluster_drid(traj, atom_selection):
	"""
	Calulate DRID representation of a trajectory for clustering.

	DRID distances are calculated using mdtraj.compute_drid().

	Parameters
	----------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	atom_selection : ndarray
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	drid_distances : ndarray
		Numpy array (M, N). M equals number of frames.
		N equals number of computed DRID distances.
	"""

	drid_distances = mdt.compute_drid(traj=traj, atom_indices=atom_selection)

	return drid_distances


def _get_cluster_metric(metric):
	"""
	Get a clustering metric.

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


def _extract_xyz(path_to_top, path_to_traj, atom_selection):
	"""
	Calculate pairwise frame RMSD using XYZ coordinates.

	Parameters
	----------
	path_to_top : str
		Path to the topology .pdb file.
	path_to_traj : str
		Path to the trajectory file (mdtraj supported extensions).
	atom_selection : ndarray
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	rmsd_matrix : ndarray
		Numpy matrix (N, N) with pairwise frame RMSD values,
		where N equals the number of frames.
	"""

	# Load the trajectory
	traj = mdt.load(path_to_traj, top=path_to_top)
	nframes = traj.n_frames

	# Calculate XYZ RMSD between frames
	rmsd_matrix = np.zeros((nframes, nframes))
	for i in range(nframes):
		rmsd_matrix[i:i + 1, :] = mdt.rmsd(target=traj,
							reference=traj,
							frame=i,
							atom_indices=atom_selection,
							parallel=True)

	return traj, rmsd_matrix


def _extract_distances(path_to_top, path_to_traj, atom_selection):
	"""
	Calculate pairwise atom distance RMSD between frames.

	Parameters
	----------
	path_to_top : str
		Path to the topology .pdb file.
	path_to_traj : str
		Path to the trajectory file (mdtraj supported extensions).
	atom_selection : ndarray
		Numpy array (N, ) containing indices of atoms.

	Returns
	-------
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	rmsd_matrix : ndarray
		Numpy array (N, N) with pairwise atom distance RMSD between frames,
		where N equals the number of frames.
	"""

	# Load the trajectory
	traj = mdt.load(path_to_traj, top=path_to_top)

	# Calculate pairwise atom distance RMSD between frames
	atom_pairs = list(combinations(atom_selection, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)
	rmsd_mat = squareform(pdist(pairwise_distances, metric="euclidean"))

	return traj, rmsd_mat


def _get_extract_metric(metric):
	"""
	Get metric for extracting representative trajectory cluster leader.

	Following metrics are used to calculate RMSD scores:
		1) xyz - xyz coordinates.
		2) distances - cluster on pairwise inter-atom distances.

	For effective extraction of a representative trajectory cluster leader,
	while using xyz, a rotational and translational movements
	of a trajectory should be removed prior.

	Returns
	-------
	out : function
		Extraction metric function.
	"""

	extract_metrics = {"xyz": _extract_xyz,
				"distances": _extract_distances}

	return extract_metrics[metric]


class Trajectory(Base):
	"""
	Basic container for molecular dynamics trajectories.

	The Trajectory object allows loading trajectory and inspecting
	loaded topology and trajectory file.

	Attributes
	----------
	top : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj .pdb topology object.
	traj : mdtraj.core.trajectory.Trajectory object
		Loaded mdtraj trajectory.
	"""

	def __init__(self):
		""" Create a new Trajectory object."""

		Base.__init__(self)
		self._top = None
		self._traj = None

	def __repr__(self):
		return "Trajectory: {}".format(self._traj)

	def load_traj(self, top_path, traj_path, stride=1):
		"""
		Load a molecular trajectory.

		Parameters
		----------
		top_path : str
			Path to topology .pdb file.
		traj_path : str
			Path to trajectory file (mdtraj supported extensions).
		stride : int
			Skip through a trajectory. Default set to 1.
		"""

		self._top = mdt.load_pdb(top_path)
		self._traj = mdt.load(traj_path, top=top_path, stride=stride)

		return

	def get_top(self):
		"""
		Return loaded .pdb topology.

		Returns
		-------
		top : mdtraj.core.trajectory.Trajectory object
			Loaded mdtraj .pdb topology object.
		"""

		return self._top

	def get_traj(self):
		"""
		Return loaded trajectory.

		Returns
		-------
		traj : mdtraj.core.trajectory.Trajectory object
			Loaded mdtraj trajectory.
		"""

		return self._traj


class BaseCluster(Trajectory):
	"""
	Object for accessing generic clustering functions.

	The generic functions allow to interact with clustering results, such as cluster labels.
	In addition, one can save trajectory clusters and extract cluster leaders.

	Attributes
	----------
	cwdir : str
		Path to current working directory.
	cluster_labels : ndarray
		Numpy array (N, ) of cluster labels. N equals to number of frames.
	traj_cluster_dir : str
		Path to trajectory clusters directory.
	cluster_leader_dir : str
		Path to cluster leaders directory.
	leader_set : list
		A list of strings, where each string denotes a path to a leader.
	"""

	def __init__(self):
		""" Create a new BaseCluster object."""

		Trajectory.__init__(self)
		self._cwdir = Trajectory.get_cwdir(self)
		self._cluster_labels = None
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._leader_set = None

	def load_cluster_labels(self, path_to_npy):
		"""
		Load cluster labels.

		Parameters
		----------
		path_to_npy : str
			Path to .npy file containing Numpy array (N, ) with cluster labels.
		"""

		self._cluster_labels = np.load(path_to_npy)

		return

	def save_cluster_labels(self):
		""" Save cluster labels as Numpy array (N, ). N equals to number of frames."""

		np.save(self._title + "_cluster_labels", self._cluster_labels)

		return

	def get_cluster_labels(self):
		"""
		Return cluster labels for each frame in a trajectory.

		Returns
		-------
		cluster_labels : ndarray
			Numpy array (N, ) of cluster labels. N equals to number of frames.
		"""

		return self._cluster_labels

	def save_traj_clusters(self):
		"""
		Save each cluster as .xtc trajectory.

		The function creates a directory with trajectory clusters.
		"""

		# Create a directory where to put cluster trajectories
		self._traj_cluster_dir = os.path.join(self._cwdir, self._title + "_traj_clusters", '')
		Base._mkdir(self._traj_cluster_dir)

		#  Extract clusters into .xtc trajectories
		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			# Avoid saving HDBSCAN noise
			if cluster == - 1:
				pass
			else:
				self._traj[self._cluster_labels == cluster].save_xtc(filename=self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")

		return

	def get_path_to_traj_clusters(self):
		"""
		Get path to trajectory clusters.

		Returns
		-------
		traj_cluster_dir : str
			Path to trajectory clusters directory.
		"""

		return self._traj_cluster_dir

	@staticmethod
	def _extract_leader(path_to_top, path_to_traj, metric, atom_selection, trajnum, output_dir):
		"""
		Extract a representative leader from a given single cluster trajectory.

		Leader extraction is performed by calculating RMSD between frames and
		selecting a frame with lowest RMSD to other structures.

		Parameters
		----------
		path_to_top : str
			Path to the topology .pdb file.
		path_to_traj : str
			Path to the trajectory file (mdtraj supported extensions).
		metric : str
			Metric for extracting representative member out of a cluster.
		atom_selection : ndarray
			Numpy array (N, ) containing indices of atoms.
		trajnum : int
			Number of the trajectory.
		output_dir : str
			Path to output directory to save trajectory clusters.
		"""

		# Get RMSD matrix
		traj, rmsd_mat = _get_extract_metric(metric)(path_to_top, path_to_traj, atom_selection)

		# Calculate the sum along each and get leader index on the smallest number
		rmsd_sum = np.sum(rmsd_mat, axis=1)
		leader_index = np.where(rmsd_sum == rmsd_sum.min())[0][0]

		# Save the leader as a PDB
		traj[leader_index].save_pdb(filename=output_dir + "cluster_leader_" + str(trajnum) + ".pdb")

		return

	def save_cluster_leaders(self, metric, atom_selection):
		"""
		Save cluster leader from each cluster trajectories.

		The function creates a directory with extracted representative cluster leaders.
		"""

		# Create a directory where to put cluster leaders extracted from cluster trajectories
		self._cluster_leader_dir = os.path.join(self._cwdir, self._title + "_cluster_leaders", '')
		Trajectory._mkdir(self._cluster_leader_dir)

		# Extract a representative conformer from a given cluster trajectory.
		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			# Avoid saving HDBSCAN noise
			if cluster == - 1:
				pass
			else:
				BaseCluster._extract_leader(path_to_top=self._top,
								path_to_traj=(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
								metric=metric,
								atom_selection=atom_selection,
								trajnum=cluster,
								output_dir=self._cluster_leader_dir)

		# Initialize cluster leaders PDBs
		self._leader_set = glob.glob((self._cluster_leader_dir + "*"))

		return

	def get_path_to_cluster_leaders(self):
		"""
		Get path to cluster leader directory.

		Returns
		-------
		cluster_leader_dir : str
			Path to cluster leader directory.
		"""

		return self._cluster_leader_dir

	def get_cluster_leaders(self):
		"""
		Get cluster leaders.

		Returns
		-------
		leader_set : list
			A list of strings, where each string denotes a path to a leader.
		"""

		return self._leader_set
