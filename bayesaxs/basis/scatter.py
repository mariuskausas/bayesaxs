import os
import re
import glob
import shutil
import subprocess

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

import bayesaxs.basis.chi as chi
from bayesaxs.base import Base
from bayesaxs.basis.curve import Curve


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
		self._fit_dir = os.path.join(os.path.abspath(path_to_fits), '')
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

		# Define index range for cutting fit files
		exp_curve_length = exp_curve.get_curve_values().shape[0]

		# Calculate the fits and move them to directory
		for leader in self._leader_set:
			# Run CRYSOL command
			exp_curve_file = exp_curve.get_path_to_file()
			leader_index = Scatter._get_str_int(leader)
			crysol_call = Scatter._crysol_parameters(pdb=leader, dat=exp_curve_file, p=leader_index, **kwargs)
			Scatter._system_command(crysol_call)

			# Load produced fit
			fit = np.loadtxt("fit_" + leader_index + ".fit", skiprows=1)

			# Cut fit to the right shape
			fit_length = fit.shape[0]
			np.savetxt("fit_" + leader_index + ".fit", fit[fit_length - exp_curve_length:], header="CRYSOL fit")

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
		self._pairwise_chi_matrix = chi._pairwise_chi(self._fit_list)

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
			pairwise_chi = chi._pairwise_chi(clusterid_curves)

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

	def load_repfits(self, path_to_fits):
		""" Load repfit files .fit"""
		self._repfit_dir = os.path.join(os.path.abspath(path_to_fits), '')
		repfits = glob.glob((self._repfit_dir + "*"))
		self._repfit_list = [Curve(repfit, title=Scatter._get_str_int(repfit)) for repfit in repfits]
