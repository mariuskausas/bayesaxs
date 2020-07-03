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
	"""
	Scatter object for performing scattering analysis.

	The object allows you to perform the following:
		1) Load cluster leaders or calculated cluster leader fits.
		2) Perform CRYSOL calculation for cluster leaders.
		3) Cluster calculated fits and select a set of representative fits.

	Attributes
	----------
	cluster_leader_dir : str
		Path to cluster leaders directory.
	leader_set : list
		A list of strings, where each string denotes a path to a leader.
	fit_dir : str
		Path to fit directory.
	fit_list : list
		A list of bayesaxs.basis.scatter.Curve objects representing each fit.
	crysol_log_dir : str
		Path to CRYSOL log directory.
	pairwise_chi_matrix : ndarray
		Numpy array (N, N) containing pairwise reduced chi squared values,
		where N is the number of fits.
	linkage_matrix : ndarray
		The hierarchical clustering encoded as a linkage matrix.
	linkage_cutoff : float
		Cutoff value for selecting number of cluster fits and dendogram visualisation.
	linkage_dendogram : dict
		A dictionary of data structures computed to render the dendrogram.
	fit_cluster_indices : ndarray
		Numpy array (N, ) with cluster labels for each fit, where N
		equals to number of fits.
	indices_of_clusterids : list
		A list of lists, where each list represents a fit cluster
		and contains indices of each fit of that cluster.
	sorted_pairwise_chi_matrix : ndarray
		Numpy array (N, N) containing pairwise reduced chi squared values,
		where N is the number of fits. The array is sorted according clustering.
	repfit_dir : str
		Path to representative fit directory.
	repfit_list : list
		A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
	"""

	def __init__(self):
		""" Create a new Scatter object."""

		Base.__init__(self)
		self._cluster_leader_dir = None
		self._leader_set = None
		self._fit_dir = None
		self._fit_list = None
		self._crysol_log_dir = None
		self._pairwise_chi_matrix = None
		self._linkage_matrix = None
		self._linkage_cutoff = None
		self._linkage_dendogram = None
		self._fit_cluster_indices = None
		self._indices_of_clusterids = None
		self._sorted_pairwise_chi_matrix = None
		self._repfit_dir = None
		self._repfit_list = None

	def __repr__(self):
		return "Analysis: {}".format(self._title)

	def load_cluster_leaders(self, path_to_leaders):
		"""
		Load representative cluster leaders.

		Parameters
		----------
		path_to_leaders : str
			Path to cluster leaders directory.
		"""

		self._cluster_leader_dir = os.path.join(os.path.abspath(path_to_leaders), '')
		self._leader_set = glob.glob((self._cluster_leader_dir + "*"))

		return

	def get_path_to_cluster_leaders(self):
		"""
		Get path to representative cluster leaders directory.

		Returns
		-------
		out : str
			Path to cluster leaders directory.
		"""

		return self._cluster_leader_dir

	def get_cluster_leaders(self):
		"""
		Get representative cluster leaders.

		Returns
		-------
		out : list
			A list of strings, where each string denotes a path to a leader.
		"""

		return self._leader_set

	@staticmethod
	def _get_str_int(s):
		"""
		Extract integer out from a file name.

		Returns
		-------
		out : str
			Integer as a string.
		"""

		return re.findall("\d+", os.path.basename(s))[0]

	@staticmethod
	def _load_curve_objects(fits):
		""" Initialize multiple Curve objects.

		Parameters
		----------
		fits : list
			A list of strings denoting path to a .fit file.

		Returns
		-------
		curve_list : list
			A list of bayesaxs.basis.scatter.Curve objects representing each fit.
		"""

		curve_list = []
		for fit in fits:
			curve = Curve(title=Scatter._get_str_int(fit))
			curve.load_txt(fit)
			curve_list.append(curve)

		return curve_list

	def load_fits(self, path_to_fits):
		"""
		Load .fit files generated using CRYSOL.

		Parameters
		----------
		path_to_fits : str
			Path to a fit directory.
		"""

		# Define paths to fits
		self._fit_dir = os.path.join(os.path.abspath(path_to_fits), '')
		fits = glob.glob((self._fit_dir + "*"))

		# Load each fit as a Curve object
		self._fit_list = Scatter._load_curve_objects(fits)

		return

	@staticmethod
	def _crysol_parameters(pdb, dat, p, lm=25, fb=17, sm=1, ns=256, un=1, dns=0.334, dro=0, err=True, cst=True):
		"""
		Define a CRYSOL command call with relative parameters.

		Parameters
		----------
		pdb : str
			Input .pdb file.
		dat : str
			Input .dat file.
		p : str
			Prefix for output file names.
		lm : int
			Maximum number of harmonics.
		fb : int
			Order of Fibonacci grid.
		sm : int
			Maximum scattering vector.
		ns : int
			Number of points in computed curve.
		un : int
			Angular units.
		dns : int
			Solvent density.
		dro : int
			Contrast of hydration shell.
		err : bool
			Write experimental errors to the .fit file. Set to True by default.
		cst : bool
			Constant subtraction. Set to True by default.

		Returns
		-------
		crysol_command : list
			A CRYSOL command line as a list.
		"""

		# Define CRYSOL input parameters
		parameters = {"pdb": ["{}".format(pdb)],
					"dat": ["{}".format(dat)],
					"p": ["-p"] + ["fit_{}".format(p)],
					"lm": ["-lm"] + ["{}".format(lm)],
					"fb": ["-fb"] + ["{}".format(fb)],
					"sm": ["-sm"] + ["{}".format(sm)],
					"ns": ["-ns"] + ["{}".format(ns)],
					"un": ["-un"] + ["{}".format(un)],
					"dns": ["-dns"] + ["{}".format(dns)],
					"dro": ["-dro"] + ["{}".format(dro)]}

		# Check if err and cst flags need to be set up
		if err:
			parameters["err"] = ["-err"]
		if cst:
			parameters["cst"] = ["-cst"]

		# Construct CRYSOL call with associated parameters
		crysol_command = ["crysol"]
		for key in parameters.keys():
			crysol_command += parameters.get(key, [])  # return empty list to avoid None addition

		return crysol_command

	@staticmethod
	def _system_command(command):
		"""
		Run a command line using subprocess.

		Returns
		-------
		out : subprocess.call
			Run the command described by args.
		"""

		status = subprocess.call(command)

		return status

	@staticmethod
	def _crysol_fit_format(exp_curve, leader_index):
		"""
		Format CRYSOL output .fit file.

		Parameters
		----------
		exp_curve : bayesaxs.basis.scatter.Curve object
			Experimental scattering curve loaded as Curve object.
		leader_index : str
			Index number of the trajectory cluster index.
		"""

		# Load produced fit
		fit = np.loadtxt(fname="fit_" + leader_index + ".fit", skiprows=1)  # Skip CRYSOL .fit header

		# Define index range for cutting fit files
		fit_length = fit.shape[0]
		exp_curve_length = exp_curve.get_curve_values().shape[0]
		np.savetxt(fname="fit_" + leader_index + ".fit", X=fit[fit_length - exp_curve_length:])

		return

	def _crysol_clean_up(self):
		"""
		Perform clean-up after CRYSOL calculations.

		The function moves .fit, .log and crysol_summary.txt files
		into the appropriate folders.
		"""

		# Define all fits and logs
		fits = glob.glob("fit_*.fit")
		logs = glob.glob("fit_*.log")

		# Move all fits and logs
		for idx in range(len(fits)):
			shutil.move(fits[idx], self._fit_dir)
			shutil.move(logs[idx], self._crysol_log_dir)

		# Move CRYSOL summary to a logs directory
		shutil.move("crysol_summary.txt", self._crysol_log_dir)

		return

	def calc_scattering(self, exp_curve, **kwargs):
		"""
		Calculate a theoretical scattering profile for each representative cluster leaders.

		Parameters
		----------
		exp_curve : bayesaxs.basis.scatter.Curve object
			Experimental scattering curve loaded as Curve object.
		lm : int
			Maximum number of harmonics.
		fb : int
			Order of Fibonacci grid.
		sm : int
			Maximum scattering vector.
		ns : int
			Number of points in computed curve.
		un : int
			Angular units.
		dns : int
			Solvent density.
		dro : int
			Contrast of hydration shell.
		err : bool
			Write experimental errors to the .fit file. Set to True by default.
		cst : bool
			Constant subtraction. Set to True by default.
		"""

		# Create a directory to store CRYSOL fits
		self._fit_dir = os.path.join(self._cwdir, self._title + "_fits", '')
		Base._mkdir(self._fit_dir)

		# Create a directory to store CRYSOL summary and logs
		self._crysol_log_dir = os.path.join(self._cwdir, self._title + "_crysol_logs", '')
		Base._mkdir(self._crysol_log_dir)

		# Calculate the CRYSOL fits for each leader
		exp_curve_file = exp_curve.get_path_to_file()
		for leader in self._leader_set:
			leader_index = Scatter._get_str_int(leader)
			crysol_command = Scatter._crysol_parameters(pdb=leader, dat=exp_curve_file, p=leader_index, **kwargs)
			Scatter._system_command(crysol_command)
			Scatter._crysol_fit_format(exp_curve=exp_curve, leader_index=leader_index)

		# Clean-up
		Scatter._crysol_clean_up(self)

		# Initialize fits
		Scatter.load_fits(self, self._fit_dir + "*.fit")

		return

	def get_path_to_fits(self):
		"""
		Get path to fit directory.

		out : str
			Path to fit directory.
		"""

		return self._fit_dir

	def get_fits(self):
		"""
		Get a set of scattering curves.

		out : list
			A list of bayesaxs.basis.scatter.Curve objects representing each fit.
		"""

		return self._fit_list

	def get_crysol_summary(self):
		""" Provide a summary about CRYSOL calculations."""

		raise NotImplementedError

	def get_path_to_crysol_logs(self):
		"""
		Get path to CRYSOL log directory.

		out : str
			Path to CRYSOL log directory.
		"""

		return self._crysol_log_dir

	def calc_pairwise_chi_matrix(self):
		""" Calculate a pairwise reduced chi square matrix for a set of fits."""

		self._pairwise_chi_matrix = chi._pairwise_chi(self._fit_list)

		return

	def get_pairwise_chi_matrix(self):
		""" Get a pairwise reduced chi square matrix for a set of fits."""

		return self._pairwise_chi_matrix

	@staticmethod
	def _get_clusterids(linkage_matrix, cutoff):
		"""
		Get indices for each scattering cluster.

		Parameters
		----------
		linkage_matrix : ndarray
			The hierarchical clustering encoded as a linkage matrix.
		cutoff : float
			Value between [0, 1] for cluster cutoff.

		Returns
		-------
		indx : list
			Linkage leaves indices for sorting.
		indices_of_clusterids : list
			A list of lists, where each list represents a fit cluster
			and contains indices of each fit of that cluster.
		"""

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
		"""
		Sort a pairwise reduced chi squared matrix.

		Parameters
		----------
		pairwise_chi_matrix : ndarray
			Numpy array (N, N) containing pairwise reduced chi squared values,
			where N is the number of fits.
		linkage_matrix : ndarray
			The hierarchical clustering encoded as a linkage matrix.
		cutoff : float
			Value between [0, 1] for cluster cutoff.

		Returns
		-------
		sorted_matrix : ndarray
			Numpy array (N, N) containing pairwise reduced chi squared values,
			where N is the number of fits. The array is sorted according clustering.
		"""

		# Calculate a dendogram based on a linkage matrix
		self._linkage_dendogram = sch.dendrogram(linkage_matrix, orientation='left', color_threshold=cutoff)
		index = self._linkage_dendogram['leaves']

		# Sort by leaves
		sorted_matrix = pairwise_chi_matrix.copy()[index, :]
		sorted_matrix = sorted_matrix[:, index]

		return sorted_matrix

	def cluster_fits(self, method="average", metric="euclidean", cutoff_value=0.25):
		"""
		Perform hierarchical clustering on pairwise reduced chi squared matrix.

		Parameters
		----------
		method : str
			Available methods to calculate distances between newly formed clusters (scipy.cluster.hierarchy.linkage).
		metric : str
			Distance metric for calculating distances between newly formed clusters.
		cutoff_value : float
			Cluster cutoff [0, 1] of the maximum distance between newly formed clusters.
		"""

		# Perform linkage clustering
		self._linkage_matrix = sch.linkage(self._pairwise_chi_matrix, method=method, metric=metric)

		# Define a cut off value in a range [0,1]
		self._linkage_cutoff = cutoff_value * max(self._linkage_matrix[:, 2])

		# Get cluster ids from clustering
		self._fit_cluster_indices, self._indices_of_clusterids = Scatter._get_clusterids(linkage_matrix=self._linkage_matrix,
																						cutoff=self._linkage_cutoff)

		# Sort a cluster
		self._sorted_pairwise_chi_matrix = Scatter._sort_pairwise_chi_matrix_(self, pairwise_chi_matrix=self._pairwise_chi_matrix,
																			linkage_matrix=self._linkage_matrix,
																			cutoff=self._linkage_cutoff)

		return

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
		Get clustering linkage cutoff.

		Returns
		-------
		out : float
			Cutoff value for selecting number of cluster fits and dendogram visualisation.
		"""

		return self._linkage_cutoff

	def get_linkage_dendogram(self):
		"""
		Get clustering linkage dendogram.

		Returns
		-------
		out : dict
			A dictionary of data structures computed to render the dendrogram.
		"""

		return self._linkage_dendogram

	def get_fit_cluster_indices(self):
		"""
		Get cluster indices of clustered fits.

		Returns
		-------
		out : ndarray
			Numpy array (N, ) with cluster labels for each fit, where N
			equals to number of fits.
		"""

		return self._fit_cluster_indices

	def get_indices_of_clusterids(self):
		"""
		Get position indices of each clustered fits.

		Returns
		-------
		out : list
			A list of lists, where each list represents a fit cluster
			and contains indices of each fit of that cluster.
		"""

		return self._indices_of_clusterids

	def get_sorted_pairwise_chi_matrix(self):
		"""
		Get a sorted pairwise reduced chi squared matrix.

		Returns
		-------
		sorted_pairwise_chi_matrix : ndarray
			Numpy array (N, N) containing pairwise reduced chi squared values,
			where N is the number of fits. The array is sorted according clustering.
		"""

		return self._sorted_pairwise_chi_matrix

	@staticmethod
	def _repr_distmat(observation_matrix):
		"""
		Finds a representative member of an observation matrix (e.g. pair-wise chi square matrix).

		Parameters
		----------
		observation_matrix : ndarray
			Observation matrix (N, N).

		Returns
		-------
		out : ndarray
			Numpy array of boolean values for mapping a representative member.
		"""

		# Transform an observation matrix into a distance matrix in euclidean space
		distmat = squareform(pdist(observation_matrix, metric="euclidean"))

		# Find the minimum value in the of the axis sum
		axis_sum = np.sum(distmat, axis=0)
		axis_sum_min = np.min(axis_sum)

		return axis_sum == axis_sum_min

	def calc_representative_fits(self):
		""" Calculate representative fits."""

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

		return

	def get_path_to_repfits(self):
		"""
		Get path to representative fit directory.

		Returns
		-------
		out : str
			Path to representative fit directory.
		"""

		return self._repfit_dir

	def get_representative_fits(self):
		"""
		Get a list of representative fits.

		Returns
		-------
		out : list
			A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
		"""

		return self._repfit_list

	def load_representative_fits(self, path_to_fits):
		"""
		Load representative fits.

		Parameters
		----------
		path_to_fits : str
			Path to fit directory.
		"""

		# Define paths to representative fits
		self._repfit_dir = os.path.join(os.path.abspath(path_to_fits), '')
		repfits = glob.glob((self._repfit_dir + "*"))

		# Load representative fits as Curve objects
		self._repfit_list = Scatter._load_curve_objects(repfits)

		return
