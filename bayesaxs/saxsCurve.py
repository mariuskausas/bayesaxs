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
	
	def __init__(self, title="Unnamed"):

		self._title = title
		self._path_to_file = None
		self._curve_data = None
		self._q = None
		self._iq = None
		self._sigma = None
		self._fit = None

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

	def load_curve(self, path_to_file):

		""" Load scattering curve."""

		# FIXME skiprows set to 1. Quick fix for loading experimental and calculated fit scatter data

		self._path_to_file = path_to_file
		self._curve_data = np.loadtxt(path_to_file, skiprows=1)

		# Set q, I(q) and sigma values

		self._q = self._curve_data[:, :1]
		self._iq = self._curve_data[:, 1:2]
		self._sigma = self._curve_data[:, 2:3]

		# FIXME Needs to be check first if the 4th column exists

		self._fit = self._curve_data[:, 3:4]

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
		self._cluster_labels = None
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._leader_set = None

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

	def traj_clustering(self):
		# FIXME add options to change clustering parameters
		"""Perform HDBSCAN clustering."""

		# Format the trajectory for HDBSCAN input

		temp = self._traj.xyz
		frames = temp.shape[0]
		atoms = temp.shape[1]
		data = temp.reshape((frames, atoms * 3))
		data = data.astype("float64")

		# Delete temporary data

		temp = []

		# Initialize HDBSCAN clusterer

		clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=-1)

		# Perform clustering

		cluster_labels = clusterer.fit_predict(data)

		# Assign cluster labels to the object

		self._cluster_labels = cluster_labels


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
			cluster_leader(top=self._pdb,
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


class Analysis(object):

	def __init__(self, title="Unnamed"):

		self._title = title
		self._leader_set = None
		self._fit_set = None
		self._chi_pairwise_matrix = None

	def __repr__(self):

		return "Analysis: {}".format(self._title)

	def get_title(self):

		""" Returns a title of a analysis."""

		return self._title

	def set_title(self, title):

		""" Set a new title for analysis."""

		self._title = str(title)

	def load_leaders(self, traj):

		""" Load extracted leaders after trajectory clustering."""

		self._leader_set = traj.get_cluster_leaders()

	def calculate_fits(self, exp_curve):

		""" Calculate theoretical scattering curves for each cluster leader using CRYSOL."""

		# Call CRYSOL and calculate theoretical scatter profile based on a set of PDB structures

		for pdb in range(len(self._leader_set)):

			crysolCommand = "crysol "\
							+ self._leader_set[pdb]\
							+ " "\
							+ exp_curve._get_path_to_file()\
							+ " "\
							+ "-p "\
							+ str(int(re.findall("\d+", self._leader_set[pdb])[0]))
			process = subprocess.Popen(crysolCommand.split(), stdout=subprocess.PIPE)
			process.communicate()

	def get_crysol_summary(self):

		""" Provide a summary about CRYSOL calculations."""

		pass

	def initialize_fits(self):

		""" Initialize all theoretical scattering curves."""

		# Initialize the directory where calculate fits are

		# FIXME use the name of a fit in the tile

		self._fit_set = [Curve(fit) for fit in glob.glob("./*.fit")]

	def get_fit_set(self):

		""" Return a set of scattering curves."""

		# FIXME print out proper names of each fit

		return self._fit_set

	def calc_pairwise_chi(self):

		""" Calculate pairwise Chi square values between theoreatical scattering curves."""

		number_of_fits = len(self._fit_set)

		chi_pairwise_matrix = np.zeros((number_of_fits, number_of_fits))

		# FIXME how can one improve this + make sure that it actually does what you want
		for i in range(number_of_fits):
			for j in range(number_of_fits):
				chi_pairwise_matrix[i:i+1, j:j+1] = chi(self._fit_set()[i].get_fit(),
														self._fit_set()[j].get_fit(),
														self._fit_set()[i].get_sigma())

		self._chi_pairwise_matrix = chi_pairwise_matrix

	def get_fit_pairwise_matrix(self):

		""" Return Chi square pairwise matrix between theoretical scattering curves."""

		return self._chi_pairwise_matrix

	def clusterFits(self):

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

	def clusterFits2(self):

		# Perform clustering

		Y = sch.linkage(self._chi_pairwise_matrix, method='average', metric="euclidean")
		cutoff = 0.25*max(Y[:, 2])

		# Extract indices for clusters

		indx = sch.fcluster(Y, cutoff, criterion='distance')

		self.fit_cluster_indices = indx

		# Generate a list of cluster fit indices

		clusterids = np.arange(1, self.fit_cluster_indices.max() + 1, 1)

		# Populate a list with cluster fit indices for each cluster index

		indices_of_clusterids = []

		for clusterid in clusterids:
			set_of_indices = [i for i, x in enumerate(self.fit_cluster_indices) if x == clusterid]
			indices_of_clusterids.append(set_of_indices)

		self.indices_of_clusterids = indices_of_clusterids

	def extract_representative_fits(self):

		# Empty list for representative fits

		repfit_list = []

		for clusterid_set in self.indices_of_clusterids:
			clusterid_array = np.zeros((len(clusterid_set), len(clusterid_set)))

			for i in range(len(clusterid_set)):
				for j in range(len(clusterid_set)):
					clusterid_array[i:i+1, j:j+1] = chi(self._fit_set()[i].get_fit(),
																self._fit_set()[j].get_fit(),
																self._fit_set()[i].get_sigma())

			condensed_dist_matrix_of_clusterid = pdist(clusterid_array, metric='euclidean')

			squareform_dist_matrix_of_clusterid = squareform(condensed_dist_matrix_of_clusterid)

			squareform_dist_matrix_axis_sum_of_clusterid = np.sum(squareform_dist_matrix_of_clusterid, axis=0)

			min_squareform_dist_matrix_axis_sum_of_clusterid = np.min(squareform_dist_matrix_axis_sum_of_clusterid)

			array = np.array(clusterid_set)

			repfit_of_clusterid = array[
				squareform_dist_matrix_axis_sum_of_clusterid == min_squareform_dist_matrix_axis_sum_of_clusterid]

			repfit_list.append(repfit_of_clusterid[0])

		self.repfit_list = repfit_list

	def get_fit_cluster_indices(self):
		return self.fit_cluster_indices

	def get_indices_of_clusterids(self):
		return self.indices_of_clusterids

	def get_repfit(self):
		return self.repfit_list

### Utilities

def chi(exp, theor, error):

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))

	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def cluster_leader(top, traj, trajnum, output_dir):

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



# # Load trajectory
#
# traj = Trajectory("./data/HOIPwtzn.pdb", "./data/tinytraj_fit.xtc")
#
#
#
# # Trajectory clustering
#
# traj.traj_clustering()
#
# # Get cluster labels
#
# # print(traj.get_cluster_labels())
# #
# # plt.plot(traj.get_cluster_labels(), 'x')
# # plt.show()
#
# #
# # # Extract clusters
# #
# # traj.extract_traj_clusters()
# #
# # # Extract cluster leaders
# #
# # traj.extract_cluster_leaders()
#
#
#
# # Set cluster leaders
#
# traj.set_collectionPDB("./cluster_leaders/*")
#
# # Experimental SAXS profile
#
# expsaxs = CurveSAXS("./data/HOIP_removedNaN.dat")
#
# # Start analysis
#
# analysis = AnalysisSAXS(traj)
#
# # # Calculate fits for cluster leaders
# #
# # analysis.calcFits(expsaxs)
#
#
# # Load fits
#
# analysis.initializeFits()
#
# # Calculate pairwise chi values
#
# analysis.calcPairwiseChiFits()
#
# # Plot pairwise chi values
#
# # sns.heatmap(analysis.get_fit_pairwise_matrix())
# # plt.show()
#
# # Cluster fits
#
# analysis.clusterFits2()
#
# # Plot clustered fits
#
# # sns.heatmap(analysis.get_cluster_matrix())
# # plt.show()
#
# # Fit cluster indices
#
# print(type(analysis.get_fit_cluster_indices()))
#
# print(analysis.get_fit_cluster_indices())
#
# print(len(analysis.get_collectionFits()))
#
# print(analysis.get_indices_of_clusterids())
# #
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[6].get_fit(log=True), 'r')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[13].get_fit(log=True), 'r')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[14].get_fit(log=True), 'r')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[15].get_fit(log=True), 'r')
# #
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[3].get_fit(log=True), 'b')
# #
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[0].get_fit(log=True), 'k')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[5].get_fit(log=True), 'k')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[8].get_fit(log=True), 'k')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[10].get_fit(log=True), 'k')
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[11].get_fit(log=True), 'k')
# # plt.show()
#
# analysis.extract_representative_fits()
#
# print(analysis.get_repfit())
#
#
# #
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[1].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[3].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[13].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[12].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[8].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[4].get_fit(log=True))
# # plt.plot(analysis.get_collectionFits()[0].get_q(), analysis.get_collectionFits()[7].get_fit(log=True))
# # plt.show()