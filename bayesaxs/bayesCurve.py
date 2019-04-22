import os
import re
import glob
import shutil
import subprocess
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import mdtraj as mdt
import hdbscan
import bayesChi


class Base(object):

	def __init__(self, title="Unnamed"):
		self._title = str(title).strip()
		self._cwdir = os.path.join(os.getcwd(), '')

	def set_title(self, title):
		""" Set a new title for a curve."""
		self._title = str(title)

	def get_title(self):
		""" Returns a title of a curve."""
		return self._title

	def set_cwdir(self, path):
		""" Set path for current working directory."""
		self._cwdir = os.path.normpath(os.path.join(os.getcwd(), path, ''))

	def get_cwdir(self):
		""" Get absolute path for current working directory."""
		return self._cwdir

	@staticmethod
	def _mkdir(dir_name):
		""" Create a directory."""
		if os.path.isdir(dir_name):
			print("Such folder already exists: {name}".format(name=dir_name))
			return
		else:
			os.mkdir(dir_name)


class Curve(Base):
	
	def __init__(self, path_to_file, title="Unnamed"):
		Base.__init__(self, title=title)
		self._path_to_file = path_to_file
		self._curve_data = np.loadtxt(path_to_file, skiprows=1)
		self._q = self._curve_data[:, :1]
		self._iq = self._curve_data[:, 1:2]
		self._sigma = self._curve_data[:, 2:3]
		self._fit = self._curve_data[:, 3:4]

	def __repr__(self):
		return "Curve: {}".format(self._title)

	def get_path_to_file(self):
		return self._path_to_file

	def get_dataarray(self):
		""" Return scattering curve data."""
		return self._curve_data
		
	def get_q(self):
		""" Return q values."""
		return self._q

	def get_iq(self):
		""" Return I(q) values."""
		return self._iq

	def get_logiq(self):
		""" Return log10 of I(q) values."""
		return np.log10(self._iq)

	def get_sigma(self):
		""" Return error of the curve."""
		return self._sigma

	def get_fit(self):
		""" Return fit values."""
		return self._fit

	def get_logfit(self):
		""" Return log10 of fit values."""
		return np.log10(self._fit)


class Trajectory(Base):

	def __init__(self):
		Base.__init__(self)
		self._pdb = None
		self._traj = None

	def __repr__(self):
		return "Trajectory: {}".format(self._traj)

	def load_traj(self, pdb_path, traj_path):
		""" Load a trajectory."""
		self._pdb = mdt.load_pdb(pdb_path)
		self._traj = mdt.load(traj_path, top=pdb_path)

	def get_traj(self):
		"""Return loaded trajectory as an instance of a class."""
		return self._traj


class BaseCluster(Trajectory):

	def __init__(self):
		Trajectory.__init__(self)
		self._cluster_labels = None
		self._cwdir = os.path.join(os.getcwd(), '')
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._leader_set = None

	def save_traj_clusters(self, folder_name):
		"""Save each cluster as .xtc trajectory."""
		# Create a directory where to put cluster trajectories
		self._traj_cluster_dir = os.path.join(self._cwdir, folder_name, '')
		Base._mkdir(self._traj_cluster_dir)
		#  Extract clusters into .xtc trajectories
		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			self._traj[self._cluster_labels == cluster].save_xtc(filename=self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")

	def get_cluster_labels(self):
		""" Return cluster labels for each frame in a trajectory."""
		return self._cluster_labels

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

	def save_cluster_leaders(self, folder_name):
		"""Save each cluster leader as .pdb from cluster trajectories."""
		# Create a directory where to put cluster leaders extracted from cluster trajectories
		self._cluster_leader_dir = os.path.join(self._cwdir, folder_name, '')
		Base._mkdir(self._cluster_leader_dir)
		# Extract a representative conformer from a given cluster trajectory. Skip HDBSCAN noise assignment (cluster -1)
		for cluster in range(self._cluster_labels.min() + 1, self._cluster_labels.max() + 1):
			BaseCluster._extract_leader(top=self._pdb,
							traj=(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
							trajnum=cluster,
							output_dir=self._cluster_leader_dir)
		# Initialize cluster PDBs
		BaseCluster.load_cluster_leaders(self, path_to_leaders=self._cluster_leader_dir + "*.pdb")

	def load_cluster_leaders(self, path_to_leaders):
		""" Load cluster leaders."""
		# This function could become a static class method or a part of save_cluster_leaders
		self._leader_set = glob.glob(path_to_leaders)

	def get_cluster_leaders(self):
		""" Get cluster leaders."""
		return self._leader_set


class HDBSCAN(BaseCluster):

	def __init__(self, min_cluster_size=5, metric="euclidean", core_dist_n_jobs=-1, **kwargs):
		BaseCluster.__init__(self)
		self._clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
										metric=metric,
										core_dist_n_jobs=core_dist_n_jobs, **kwargs)

	@staticmethod
	def _reshape_XYZ(traj):
		"""Reshape XYZ coordinates of a trajectory for clustering."""
		temp = traj.xyz
		frames = temp.shape[0]
		atoms = temp.shape[1]
		reshaped = temp.reshape((frames, atoms * 3))
		reshaped = reshaped.astype("float64")
		temp = []
		return reshaped

	def fit_predict(self):
		"""Perform HDBSCAN clustering."""
		traj_reshaped = HDBSCAN._reshape_XYZ(traj=self._traj)
		self._cluster_labels = self._clusterer.fit_predict(X=traj_reshaped)


class Scatter(Base):

	def __init__(self):
		Base.__init__(self)
		self._leader_set = None
		self._fit_dir = None
		self._fit_set = None
		self._pairwise_chi_matrix = None
		self._linkage_matrix = None
		self._linkage_dendogram = None
		self._fit_cluster_indices = None
		self._indices_of_clusterids = None
		self._sorted_pairwise_chi_matrix = None
		self._repfit_list = None

	def __repr__(self):
		return "Analysis: {}".format(self._title)

	def load_cluster_leaders(self, path_to_leaders):
		""" Load extracted leaders after trajectory clustering."""
		self._leader_set = glob.glob(path_to_leaders)

	def get_cluster_leaders(self):
		""" Get extracted leaders after trajectory clustering."""
		return self._leader_set

	@staticmethod
	def _system_command(command):
		""" Run a command line."""
		status = subprocess.call(command)
		return status

	def calc_scattering(self, exp_curve, folder_name):
		""" Simulate scattering for every leader using CRYSOL."""
		# Create a directory to store all CRYSOL output
		self._fit_dir = os.path.join(self._cwdir, folder_name, '')
		Base._mkdir(self._fit_dir)
		# Calculate the fits and move them to directory
		for leader in self._leader_set:
			# Run CRYSOL command
			exp_curve_file = exp_curve.get_path_to_file()
			leader_index = Scatter._get_str_int(leader)
			call_crysol = (["crysol"] + [leader] + [exp_curve_file] + ["-p"] + ["fit_" + leader_index])
			Scatter._system_command(call_crysol)
			# Move CRYSOL output to a fits directory
			shutil.move("fit_" + leader_index + ".fit", self._fit_dir)
			shutil.move("fit_" + leader_index + ".log", self._fit_dir)
		# Move CRYSOL summary to a fits directory
		shutil.move("crysol_summary.txt", self._fit_dir)
		# Initialize fits
		Scatter.load_fits(self, self._fit_dir + "*.fit")

	@staticmethod
	def _get_str_int(s):
		""" Extract integer out from a file name."""
		return re.findall("\d+", os.path.basename(s))[0]

	def load_fits(self, path_to_fits):
		""" Load fit files .fit"""
		fits = glob.glob(path_to_fits)
		self._fit_set = [Curve(fit, title=Scatter._get_str_int(fit)) for fit in fits]

	def get_crysol_summary(self):
		""" Provide a summary about CRYSOL calculations."""
		raise NotImplementedError

	def get_path_to_fits(self):
		""" Get path to fits location."""
		return self._fit_dir

	def get_fit_set(self):
		""" Get a set of scattering curves."""
		return self._fit_set

	def calc_pairwise_chi_matrix(self):
		""" Calculate a pairwise reduced chi square matrix for a set of fits."""
		self._pairwise_chi_matrix = bayesChi.pairwise_chi(self._fit_set)

	def get_pairwise_chi_matrix(self):
		""" Get a pairwise chi square matrix between theoretical scattering curves."""
		return self._pairwise_chi_matrix

	@staticmethod
	def _get_clusterids(linkage_matrix, cutoff):
		""" Get ids for each scattering cluster."""
		indx = sch.fcluster(linkage_matrix, t=cutoff, criterion='distance')
		# Generate a list of cluster fit indices
		clusterids = np.arange(1, indx.max() + 1, 1)
		# Populate a list with cluster fit indices for each cluster index
		indices_of_clusterids = []
		for clusterid in clusterids:
			set_of_indices = [i for i, x in enumerate(indx) if x == clusterid]
			indices_of_clusterids.append(set_of_indices)
		return indx, indices_of_clusterids

	def _sort_pairwise_chi_matrix_(self, pairwise_chi_matrix, linkage_matrix, cutoff):
		""" Sort a pairwise reduced chi squared matrix."""
		# Calculate a dendogram based on a linkage matrix
		self._linkage_dendogram = sch.dendrogram(linkage_matrix, orientation='left', color_threshold=cutoff)
		index = self._linkage_dendogram['leaves']
		# Sort by leaves
		sorted_matrix = pairwise_chi_matrix.copy()[index, :]
		sorted_matrix = sorted_matrix[:, index]
		return sorted_matrix

	def get_linkage_dendogram(self):
		""" Get linkage dendogram."""
		return self._linkage_dendogram

	def cluster_fits(self, cutoff_value):
		""" Perform hierarchical clustering on pairwise reduced chi squared matrix."""
		# Perform linkage clustering
		self._linkage_matrix = sch.linkage(self._pairwise_chi_matrix, method='average', metric="euclidean")
		# Define a cut off value in a range [0,1]
		cutoff = cutoff_value * max(self._linkage_matrix[:, 2])
		# Get cluster ids from clustering
		self._fit_cluster_indices, self._indices_of_clusterids = Scatter._get_clusterids(linkage_matrix=self._linkage_matrix,
																						cutoff=cutoff)
		# Sort a cluster
		self._sorted_pairwise_chi_matrix = Scatter._sort_pairwise_chi_matrix_(self, pairwise_chi_matrix=self._pairwise_chi_matrix,
																		linkage_matrix=self._linkage_matrix,
																		cutoff=cutoff)

	def get_linkage_matrix(self):
		""" Get linkage matrix after clustering."""
		return self._linkage_matrix

	def get_fit_cluster_indices(self):
		""" Get cluster indices of clustered fits."""
		return self._fit_cluster_indices

	def get_indices_of_clusterids(self):
		""" Get position indices of each clustered fits."""
		return self._indices_of_clusterids

	def get_sorted_pairwise_chi_matrix(self):
		""" Get a clustered pairwise chi matrix."""
		return self._sorted_pairwise_chi_matrix

	@staticmethod
	def _repr_distmat(array):
		"""
		Finds a representative member of an observation matrix (e.g. pair-wise chi square matrix).

		:param array: Observation matrix (n,n).
		:return: An array of boolean values for mapping a representative member.
		"""
		# Transform an observation matrix into a distance matrix in euclidean space
		distmat = squareform(pdist(array, metric="euclidean"))
		# Find the minimum value in the of the axis sum
		axis_sum = np.sum(distmat, axis=0)
		axis_sum_min = np.min(axis_sum)
		return axis_sum == axis_sum_min

	def calc_representative_fits(self):
		""" Calculate representative fit form clustered fits."""
		# Empty list for representative fits
		repfit_list = []
		# For each cluster set find a representative member and append to a list
		for clusterid_set in self._indices_of_clusterids:
			# Extract appropriate clusterid curves for the whole fit set
			clusterid_curves = [self._fit_set[i] for i in clusterid_set]
			# Calculate a pair-wise chi matrix
			pairwise_chi = bayesChi.pairwise_chi(clusterid_curves)
			# Extract a representative member (convert a list to an array, pass a boolean np array for indexing)
			repfit_of_clusterid = np.array(clusterid_set)[Scatter._repr_distmat(pairwise_chi)][0]
			# Append a representative member to a list
			repfit_list.append(repfit_of_clusterid)
		self._repfit_list = [self._fit_set[i] for i in repfit_list]

	def get_representative_fits(self):
		""" Get a list of representative fits."""
		return self._repfit_list
