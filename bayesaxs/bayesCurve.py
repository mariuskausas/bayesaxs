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
import bayesCluster


class Base(object):

	def __init__(self, title="Unnamed"):
		self._title = title
		self._cwdir = os.path.join(os.getcwd(), '')

	def set_title(self, title):
		""" Set a new title for a curve."""
		self._title = str(title).strip()

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

	def get_curve_values(self):
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

		# Initialize cluster PDBs
		BaseCluster.load_cluster_leaders(self, path_to_leaders=self._cluster_leader_dir + "*.pdb")

	def load_cluster_leaders(self, path_to_leaders):
		""" Load cluster leaders."""

		# This function could become a static class method or a part of save_cluster_leaders
		self._leader_set = glob.glob((path_to_leaders + "*"))

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
		cluster_input = bayesCluster._get_cluster_metric(metric)(traj=self._traj, **kwargs)
		self._cluster_labels = self._clusterer.fit_predict(cluster_input)


class Scatter(Base):

	def __init__(self):
		Base.__init__(self)
		self._cluster_leader_dir = None
		self._leader_set = None
		self._fit_dir = None
		self._fit_list = None
		self._crysol_log_dir = None
		self._pairwise_chi_matrix = None
		self._linkage_matrix = None
		self._linkage_dendogram = None
		self._fit_cluster_indices = None
		self._indices_of_clusterids = None
		self._sorted_pairwise_chi_matrix = None
		self._repfit_dir = None
		self._repfit_list = None

	def __repr__(self):
		return "Analysis: {}".format(self._title)

	def load_cluster_leaders(self, path_to_leaders):
		""" Load extracted leaders after trajectory clustering."""
		self._cluster_leader_dir = os.path.join(os.path.abspath(path_to_leaders), '')
		self._leader_set = glob.glob((self._cluster_leader_dir + "*"))

	def get_path_to_cluster_leaders(self):
		""" Get path to cluster leader directory."""
		return self._cluster_leader_dir

	def get_cluster_leaders(self):
		""" Get extracted leaders after trajectory clustering."""
		return self._leader_set

	@staticmethod
	def _get_str_int(s):
		""" Extract integer out from a file name."""
		return re.findall("\d+", os.path.basename(s))[0]

	def load_fits(self, path_to_fits):
		""" Load fit files .fit"""
		self._fit_dir = os.path.join(os.path.abspath(path_to_fits),'')
		fits = glob.glob((self._fit_dir + "*"))
		self._fit_list = [Curve(fit, title=Scatter._get_str_int(fit)) for fit in fits]

	@staticmethod
	def _crysol_parameters(pdb, dat, p, lm=25, fb=17, sm=0.5, ns=256, un=1, dns=0.334, dro=0, err=True, cst=True):
		""" Set CRYSOL call parameters for passing to subprocess."""

		# Define CRYSOL input parameters
		parameters = {"pdb": ["{}".format(pdb)],  				# input .pdb file
					"dat": ["{}".format(dat)],  				# input .dat file
					"p": ["-p"] + ["fit_{}".format(p)], 		# prefix for output file names
					"lm": ["-lm"] + ["{}".format(lm)],  		# maximum number of harmonics
					"fb": ["-fb"] + ["{}".format(fb)],  		# order of Fibonacci grid
					"sm": ["-sm"] + ["{}".format(sm)],  		# maximum scattering vector
					"ns": ["-ns"] + ["{}".format(ns)],  		# number of points in computed curve
					"un": ["-un"] + ["{}".format(un)],  		# angular units
					"dns": ["-dns"] + ["{}".format(dns)],  		# solvent density
					"dro": ["-dro"] + ["{}".format(dro)]}  		# contrast of hydration shell

		# Check if err and cst flags need to be set up
		if err:
			parameters["err"] = ["-err"]
		if cst:
			parameters["cst"] = ["-cst"]

		# Construct CRYSOL call with associated parameters
		crysol_call = ["crysol"]
		for key in parameters.keys():
			crysol_call += parameters.get(key, [])  # return empty list to avoid None addition

		return crysol_call

	@staticmethod
	def _system_command(command):
		""" Run a command line."""
		status = subprocess.call(command)
		return status

	def calc_scattering(self, exp_curve, **kwargs):
		""" Simulate scattering for every leader using CRYSOL."""

		# Create a directory to store CRYSOL fits
		self._fit_dir = os.path.join(self._cwdir, self._title + "_fits", '')
		Base._mkdir(self._fit_dir)

		# Create a directory to store CRYSOL summary and logs
		self._crysol_log_dir = os.path.join(self._cwdir, self._title + "_crysol_logs", '')
		Base._mkdir(self._crysol_log_dir)

		# Calculate the fits and move them to directory
		for leader in self._leader_set:
			# Run CRYSOL command
			exp_curve_file = exp_curve.get_path_to_file()
			leader_index = Scatter._get_str_int(leader)
			crysol_call = Scatter._crysol_parameters(pdb=leader, dat=exp_curve_file, p=leader_index, **kwargs)
			Scatter._system_command(crysol_call)

			# Move CRYSOL fit to a fits directory
			shutil.move("fit_" + leader_index + ".fit", self._fit_dir)

			# Move CRYSOL log to a logs directory
			shutil.move("fit_" + leader_index + ".log", self._crysol_log_dir)

		# Move CRYSOL summary to a logs directory
		shutil.move("crysol_summary.txt", self._crysol_log_dir)

		# Initialize fits
		Scatter.load_fits(self, self._fit_dir + "*.fit")

	def get_path_to_fits(self):
		""" Get path to fits location."""
		return self._fit_dir

	def get_fits(self):
		""" Get a set of scattering curves."""
		return self._fit_list

	def get_crysol_summary(self):
		""" Provide a summary about CRYSOL calculations."""
		raise NotImplementedError

	def get_path_to_crysol_logs(self):
		""" Get path to CRYSOL logs."""
		return self._crysol_log_dir

	def calc_pairwise_chi_matrix(self):
		""" Calculate a pairwise reduced chi square matrix for a set of fits."""
		self._pairwise_chi_matrix = bayesChi.pairwise_chi(self._fit_list)

	def get_pairwise_chi_matrix(self):
		""" Get a pairwise chi square matrix between theoretical scattering curves."""
		return self._pairwise_chi_matrix

	@staticmethod
	def _get_clusterids(linkage_matrix, cutoff):
		""" Get ids for each scattering cluster."""

		# Get cluster indices
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

	def get_linkage_dendogram(self):
		""" Get linkage dendogram."""
		return self._linkage_dendogram

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

		# Create a directory to store representative fits
		self._repfit_dir = os.path.join(self._cwdir, self._title + "_repfits", '')
		Base._mkdir(self._repfit_dir)

		# Empty list for representative fits
		repfit_list = []

		# For each cluster set find a representative member and append to a list
		for clusterid_set in self._indices_of_clusterids:
			# Extract appropriate clusterid curves for the whole fit set
			clusterid_curves = [self._fit_list[i] for i in clusterid_set]

			# Calculate a pair-wise chi matrix
			pairwise_chi = bayesChi.pairwise_chi(clusterid_curves)

			# Extract a representative member (convert a list to an array, pass a boolean np array for indexing)
			repfit_of_clusterid = np.array(clusterid_set)[Scatter._repr_distmat(pairwise_chi)][0]

			# Append a representative member to a list
			repfit_list.append(repfit_of_clusterid)

			# Copy a representative .fit file to a directory
			shutil.copy(self._fit_list[repfit_of_clusterid].get_path_to_file(), self._repfit_dir)

		self._repfit_list = [self._fit_list[i] for i in repfit_list]

	def get_path_to_repfits(self):
		""" Get path to representative fits location."""
		return self._repfit_dir

	def get_representative_fits(self):
		""" Get a list of representative fits."""
		return self._repfit_list
