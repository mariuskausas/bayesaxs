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


class CurveSAXS(object):
	
	def __init__(self, filename):
		self.filename = filename
		# FIXME skiprows set to 1. Quick fix for loading experimental and calculated fit scatter data
		self.dataarray = np.loadtxt(filename, skiprows=1)

	def get_filename(self):
		return self.filename
	
	def get_dataarray(self):
		return self.dataarray
		
	def get_q(self):
		return self.dataarray[:, :1]

	def get_iq(self, log=False):

		# FIXME get the best way to deal with flags

		if log is True:
			return np.log10(self.dataarray[:, 1:2])

		return self.dataarray[:, 1:2]
	
	def get_logiq(self):
		# Possibly redudant
		return np.log10(self.dataarray[:, 1:2])

	def get_sigma(self):
		return self.dataarray[:, 2:3]

	def get_fit(self, log=False):
		# FIXME write a test to check if the 4 column with fit information does exists or not
		if log is True:
			return np.log10(self.dataarray[:, 3:4])

		return self.dataarray[:, 3:4]


class Trajectory(object):

	def __init__(self, title="Unkown"):

		"""
		Initialize Trajectory class by providing path to your topology file .pdb and trajectory file .xtc.

		"""
		# FIXME understand how to correctly define attributes outside the __init__
		self._title = str(title).strip()
		self._pdb = None
		self._traj = None
		self._cluster_labels = None
		self._traj_cluster_dir = None
		self._cluster_leader_dir = None
		self._collectionPDB = None

	def __repr__(self):

		return "{0}: Number of clusters {1}".format(self._title, self._cluster_labels.max())

	def __str__(self):

		return self.__class__.__name__ + ' ' + self._title

	def getModel(self):

		"""Returns self."""

		return self

	def setTitle(self, title):

		""" Set a new title."""

		self._title = str(title)

	def loadTraj(self, pdb_path, traj_path):

		""" Load a trajectory"""

		self._pdb = mdt.load_pdb(pdb_path)
		self._traj = mdt.load(traj_path, top=pdb_path)

	def getTraj(self):

		"""Return loaded trajectory as an instance of a class."""

		return self._traj

	def trajClustering(self):
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


	def getClusterLabels(self):

		""" Return cluster labels for each frame in a trajectory."""

		return self._cluster_labels


	def extractTrajClusters(self):

		"""Extract clusters as .xtc trajectory."""

		# Create a directory where to put cluster trajectories

		self._traj_cluster_dir = "./traj_clusters/"

		os.mkdir(self._traj_cluster_dir)

		# Extract clusters into .xtc trajectories

		for cluster in range(self._cluster_labels.min(), self._cluster_labels.max() + 1):
			self._traj[self._cluster_labels == cluster].save_xtc(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")


	def extractClusterLeaders(self):

		"""Extract cluster leaders from cluster trajectories."""

		# Create a directory where to put cluster leaders extracted from cluster trajectories

		self._cluster_leader_dir = "./cluster_leaders/"

		os.mkdir(self._cluster_leader_dir)

		# Extract a representative conformer from a given cluster trajectory. Skip HDBSCAN noise assignment (cluster -1)

		for cluster in range(self._cluster_labels.min() + 1, self._cluster_labels.max() + 1):
			clustering_leader(top=self._pdb,
							  traj=(self._traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
							  trajnum=cluster,
							  output_dir=self._cluster_leader_dir)

		# Initialize cluster PDBs

		self._collectionPDB = glob.glob("./cluster_leaders/*")

	def loadClusterLeaders(self, dir_to_pdbs):

		""" Load cluster leaders"""

		self._collectionPDB = glob.glob(dir_to_pdbs)

	def getClusterLeaders(self):

		""" Returns cluster leaders """

		return self._collectionPDB


class AnalysisSAXS(object):

	def __init__(self, trajectory):
		self.collectionPDB = trajectory.get_collectionPDB()

	def calcFits(self, expSAXS):

		# Call CRYSOL and calculate theoretical scatter profile based on a set of PDB structures

		for pdb in range(len(self.collectionPDB)):

			crysolCommand = "crysol "\
							+ self.collectionPDB[pdb]\
							+ " "\
							+ expSAXS.get_filename()\
							+ " "\
							+ "-p "\
							+ str(int(re.findall("\d+", self.collectionPDB[pdb])[0]))
			process = subprocess.Popen(crysolCommand.split(), stdout=subprocess.PIPE)
			process.communicate()

	def initializeFits(self):
		# Initialize the directory where calculate fits are
		self.collectionFits = [CurveSAXS(fit) for fit in glob.glob("./*.fit")]

	def get_collectionFits(self):
		return self.collectionFits

	def calcPairwiseChiFits(self):

		number_of_fits = len(self.collectionFits)

		fit_pairwise_matrix = np.zeros((number_of_fits, number_of_fits))

		# FIXME how can one improve this + make sure that it actually does what you want
		for i in range(number_of_fits):
			for j in range(number_of_fits):
				fit_pairwise_matrix[i:i+1, j:j+1] = chi(self.get_collectionFits()[i].get_fit(),
														self.get_collectionFits()[j].get_fit(),
														self.get_collectionFits()[i].get_sigma())

		self.fit_pairwise_matrix = fit_pairwise_matrix

	def get_fit_pairwise_matrix(self):
		return self.fit_pairwise_matrix

	def clusterFits(self):

		# FIXME how to track the index of a fit
		# FIXME am I changing the values of a .fit_pairwise_matrix ?

		Y = sch.linkage(self.fit_pairwise_matrix)
		# The fuck this value means
		cutoff = 0.25 * max(Y[:, 2])
		Z = sch.dendrogram(Y, orientation='left', color_threshold=cutoff)
		index = Z['leaves']

		# sort the matrix

		cluster_matrix = self.fit_pairwise_matrix.copy()[index, :]
		cluster_matrix = cluster_matrix[:, index]

		self.cluster_matrix = cluster_matrix


	def get_cluster_matrix(self):
		return self.cluster_matrix


	def _test(self):
		pass

	def clusterFits2(self):

		# Perform clustering

		Y = sch.linkage(self.fit_pairwise_matrix, method='average', metric="euclidean")
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
					clusterid_array[i:i+1, j:j+1] = chi(self.get_collectionFits()[i].get_fit(),
																self.get_collectionFits()[j].get_fit(),
																self.get_collectionFits()[i].get_sigma())

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


def clustering_leader(top, traj, trajnum, output_dir):

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