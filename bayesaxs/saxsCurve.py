import re
import os
import glob
import subprocess
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import mdtraj as mdt
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns


class Curve(object):
	
	def __init__(self, path_to_file):

		# Define path, title and load all curve data

		self._path_to_file = path_to_file
		self._title = str(self._path_to_file)
		self._curve_data = np.loadtxt(path_to_file, skiprows=1)

		# Set q, I(q), sigma and fit values

		self._q = self._curve_data[:, :1]
		self._iq = self._curve_data[:, 1:2]
		self._sigma = self._curve_data[:, 2:3]
		self._fit = self._curve_data[:, 3:4]

	def __repr__(self):

		return "Curve: {}".format(self._title)

	def _get_path_to_file(self):

		return self._path_to_file

	def get_title(self):

		""" Returns a title of a curve."""

		return self._title

	def set_title(self, title):

		""" Set a new title for a curve."""

		self._title = str(title)

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


class Trajectory(object):

	def __init__(self, title="Unnamed"):

		self._title = str(title).strip()
		self._pdb = None
		self._traj = None

	def __repr__(self):

		return "Trajectory: {}".format(self._title)

	def get_title(self):

		""" Returns a title of a trajectory."""

		return self._title

	def set_title(self, title):

		""" Set a new title for a trajectory."""

		self._title = str(title)

	def load_traj(self, pdb_path, traj_path):

		""" Load a trajectory."""

		self._pdb = mdt.load_pdb(pdb_path)
		self._traj = mdt.load(traj_path, top=pdb_path)

	def get_traj(self):

		"""Return loaded trajectory as an instance of a class."""

		return self._traj


class BaseClustering(Trajectory):

	def __init__(self):

		Trajectory.__init__(self)
		self._cluster_labels = None
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._leader_set = None

	def get_cluster_labels(self):

		""" Return cluster labels for each frame in a trajectory."""

		return self._cluster_labels

	def extract_traj_clusters(self):

		"""Extract clusters as .xtc trajectory."""

		# Create a directory where to put cluster trajectories

		self._traj_cluster_dir = "./traj_clusters/"
		os.mkdir(self._traj_cluster_dir)

		# Extract clusters into .xtc trajectories

		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			self._traj[self._cluster_labels == cluster].save_xtc(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")

	def extract_cluster_leaders(self):

		"""Extract cluster leaders from cluster trajectories."""

		# Create a directory where to put cluster leaders extracted from cluster trajectories

		self._cluster_leader_dir = "./cluster_leaders/"
		os.mkdir(self._cluster_leader_dir)

		# Extract a representative conformer from a given cluster trajectory. Skip HDBSCAN noise assignment (cluster -1)

		for cluster in range(self._cluster_labels.min() + 1, self._cluster_labels.max() + 1):
			_cluster_leader(top=self._pdb,
							  traj=(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
							  trajnum=cluster,
							  output_dir=self._cluster_leader_dir)

		# Initialize cluster PDBs

		self._leader_set = glob.glob("./cluster_leaders/*")

	def load_cluster_leaders(self, path_to_leaders):

		""" Load cluster leaders."""

		self._leader_set = glob.glob(path_to_leaders)

	def get_cluster_leaders(self):

		""" Returns cluster leaders."""

		return self._leader_set


class HDBSCAN(BaseClustering):

	def __init__(self, min_cluster_size=5, metric="euclidean", core_dist_n_jobs=-1):

		BaseClustering.__init__(self)
		self._clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
										metric=metric,
										core_dist_n_jobs=core_dist_n_jobs)

	def _HDBSCAN_reshape(self, traj):

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

		# Perform clustering
		traj_reshaped = self._HDBSCAN_reshape(self._traj)
		self._cluster_labels = self._clusterer.fit_predict(traj_reshaped)


class Analysis(object):

	def __init__(self, title="Unnamed"):

		self._title = title
		self._leader_set = None
		self._path_to_fits = None
		self._fit_set = None
		self._chi_pairwise_matrix = None
		self._cluster_matrix = None
		self._fit_cluster_indices = None
		self._indices_of_clusterids = None
		self._repfit_list = None

	def __repr__(self):

		return "Analysis: {}".format(self._title)

	def get_title(self):

		""" Returns a title of a analysis."""

		return self._title

	def set_title(self, title):

		""" Set a new title for analysis."""

		self._title = str(title)

	def load_leaders(self, clusterer):

		""" Load extracted leaders after trajectory clustering."""

		self._leader_set = clusterer.get_cluster_leaders()

	def calculate_fits(self, exp_curve):

		""" Calculate theoretical scattering curves for each cluster leader using CRYSOL."""

		# Call CRYSOL and calculate theoretical scatter profile based on a set of PDB structures

		for pdb in range(len(self._leader_set)):
			crysol_command = "crysol "\
							+ self._leader_set[pdb]\
							+ " "\
							+ exp_curve._get_path_to_file()\
							+ " "\
							+ "-p "\
							+ str(int(re.findall("\d+", self._leader_set[pdb])[0]))
			process = subprocess.Popen(crysol_command.split(), stdout=subprocess.PIPE)
			process.communicate()

	def get_crysol_summary(self):

		""" Provide a summary about CRYSOL calculations."""

		pass

	def initialize_fits(self):

		""" Initialize all theoretical scattering curves."""

		# Initialize the directory where calculate fits are

		# FIXME use the name of a fit in the tile
		# FIXME sort the fits by numbers? Needs to be re-ordered possibly
		# FIXME might be redundant - check better options in the future

		self._path_to_fits = glob.glob("*.fit")

		self._fit_set = [Curve(fit) for fit in glob.glob("*.fit")]

	def get_path_to_fits(self):

		return self._path_to_fits

	def get_fit_set(self):

		""" Return a set of scattering curves."""

		# FIXME print out proper names of each fit

		return self._fit_set

	# def calc_pairwise_chi2(self):
	#
	# 	""" Calculate pairwise Chi square values between theoreatical scattering curves."""
	#
	# 	number_of_fits = len(self._fit_set)
	#
	# 	chi_pairwise_matrix = np.zeros((number_of_fits, number_of_fits))
	#
	# 	for i in range(number_of_fits):
	# 		for j in range(number_of_fits):
	# 			chi_pairwise_matrix[i:i+1, j:j+1] = _chi(self._fit_set[i].get_fit(),
	# 													self._fit_set[j].get_fit(),
	# 													self._fit_set[i].get_sigma())
	#
	# 	self._chi_pairwise_matrix = chi_pairwise_matrix

	def calc_pairwise_chi(self):

		self._chi_pairwise_matrix = _pairwise_chi(self._fit_set)

	def get_fit_pairwise_matrix(self):

		""" Return Chi square pairwise matrix between theoretical scattering curves."""

		return self._chi_pairwise_matrix

	def cluster_fits(self):

		# Perform clustering

		Y = sch.linkage(self._chi_pairwise_matrix, method='average', metric="euclidean")
		cutoff = 0.25*max(Y[:, 2])

		# Extract indices for clusters

		indx = sch.fcluster(Y, cutoff, criterion='distance')

		self._fit_cluster_indices = indx

		# Generate a list of cluster fit indices

		clusterids = np.arange(1, self._fit_cluster_indices.max() + 1, 1)

		# Populate a list with cluster fit indices for each cluster index

		indices_of_clusterids = []

		for clusterid in clusterids:
			set_of_indices = [i for i, x in enumerate(self._fit_cluster_indices) if x == clusterid]
			indices_of_clusterids.append(set_of_indices)

		self._indices_of_clusterids = indices_of_clusterids

	def cluster_fits2(self):

		# FIXME how to track the index of a fit
		# FIXME am I changing the values of a .fit_pairwise_matrix ?

		Y = sch.linkage(self._chi_pairwise_matrix)
		# The fuck this value means
		cutoff = 0.25 * max(Y[:, 2])
		Z = sch.dendrogram(Y, orientation='left', color_threshold=cutoff)
		index = Z['leaves']

		# sort the matrix

		cluster_matrix = self._chi_pairwise_matrix.copy()[index, :]
		cluster_matrix = cluster_matrix[:, index]

		self._cluster_matrix = cluster_matrix

	def get_cluster_matrix(self):

		return self._cluster_matrix

	def extract_representative_fits(self):

		# Empty list for representative fits

		repfit_list = []

		# For each cluster set find a representative member and append to a list

		for clusterid_set in self._indices_of_clusterids:

			# Extract appropriate clusterid curves for the whole fit set

			clusterid_curves = [self._fit_set[i] for i in clusterid_set]

			# Calculate a pair-wise chi matrix

			pairwise_chi = _pairwise_chi(clusterid_curves)

			# Extract a representative member (convert a list to an array, pass a boolean np array for indexing)

			repfit_of_clusterid = np.array(clusterid_set)[_repr_distmat(pairwise_chi)][0]

			# Append a representative member to a list

			repfit_list.append(repfit_of_clusterid)

		self._repfit_list = [self._fit_set[i] for i in repfit_list]

	def get_fit_cluster_indices(self):

		return self._fit_cluster_indices

	def get_indices_of_clusterids(self):

		return self._indices_of_clusterids

	def get_repfit(self):

		return self._repfit_list


# Utilities


def _chi(exp, theor, error):

	"""
	Calculate reduced chi squared.

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))

	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def _cluster_leader(top, traj, trajnum, output_dir):

	"""
	Extract a representative conformer from a given single cluster trajectory.

	"""

	# Load the trajectory

	traj = mdt.load(traj, top=top, stride=1)

	# Number of frames

	nframes = traj.n_frames

	# Create a RMSD distance matrix

	rmsd_matrix = np.zeros((nframes, nframes))

	# Calculate pairwise RMSD between each of the frame

	for i in range(nframes):
		rmsd_matrix[i:i + 1, :] = mdt.rmsd(traj, traj, i, parallel=True)

	# Calculate the sum along each row

	rmsd_sum = np.sum(rmsd_matrix, axis=1)

	# Calculate the leader index based on the smallest number

	leader_index = np.where(rmsd_sum == rmsd_sum.min())[0][0]

	# Save the leader as a PDB

	traj[leader_index].save_pdb(output_dir + "cluster_leader_" + str(trajnum) + ".pdb")


def _repr_distmat(array):

	"""
	Finds a representative member of an observation matrix (e.g. pair-wise chi square matrix).

	:param array: Observation matrix (n,n).
	:return: An array of boolean values for mapping a representative member.
	"""

	# Transform an observation matrix into a distance matrix in euclidean space

	distmat = squareform(pdist(array, metric="euclidean"))

	# Sum up values along an axis in the distance matrix

	axis_sum = np.sum(distmat, axis=0)

	# Find the minimum value in the of the axis sum

	axis_sum_min = np.min(axis_sum)

	return axis_sum == axis_sum_min


def _pairwise_chi(curves):

	"""
	Generate a pairwise chi matrix.

	:param curves: A list containing curves to iterate over.
	:return: An array containing pairwise reduced chi squared values.
	"""

	# Define the number of curves to iterate over

	number_of_curves = len(curves)

	# Generate an empty array (n,n) for a given n of curves

	pairwise_mat = np.zeros((number_of_curves, number_of_curves))

	# Perform a pairwise reduced chi squared calculation

	for i in range(number_of_curves):
		for j in range(number_of_curves):
			pairwise_mat[i:i + 1, j:j + 1] = _chi(curves[i].get_fit(),
															 curves[j].get_fit(),
															 curves[i].get_sigma())

	return pairwise_mat
















