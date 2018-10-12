import re
import os
import glob
import subprocess
import numpy as np
import scipy.cluster.hierarchy as sch
import mdtraj as mdt
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns


# Helper functions


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

		if log == True:
			return np.log10(self.dataarray[:, 1:2])

		return self.dataarray[:, 1:2]
	
	def get_logiq(self):
		# Possibly redudant
		return np.log10(self.dataarray[:, 1:2])

	def get_sigma(self):
		return self.dataarray[:, 2:3]

	def get_fit(self, log=False):
		# FIXME write a test to check if the 4 column with fit information does exists or not
		if log == True:
			return np.log10(self.dataarray[:, 3:4])

		return self.dataarray[:, 3:4]




class Trajectory1(object):

	def __init__(self, pdb_path, traj_path):

		"""
		Initialize Trajectory class by providing path to your topology file .pdb and trajectory file .xtc.

		"""

		self.pdb = mdt.load_pdb(pdb_path)
		self.traj = mdt.load(traj_path, top=pdb_path)

	def get_trajectory(self):

		"""
		Return loaded trajectory as an instance of a class.

		"""

		return self.traj

	def traj_clustering(self):

		"""
		Perform HDBSCAN clustering.

		"""

		# Format the trajectory for HDBSCAN input

		temp = self.traj.xyz
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

		self.cluster_labels = cluster_labels


	def get_cluster_labels(self):

		"""
		Return cluster labels for each frame in a trajectory.

		"""

		return self.cluster_labels


	def extract_traj_clusters(self):

		"""
		Extract clusters as .xtc trajectory.

		"""

		# Create a directory where to put cluster trajectories

		self.traj_cluster_dir = "./traj_clusters/"

		os.mkdir(self.traj_cluster_dir)

		# Extract clusters into .xtc trajectories

		for cluster in range(self.cluster_labels.min(), self.cluster_labels.max() + 1):
			self.traj[self.cluster_labels == cluster].save_xtc(self.traj_cluster_dir + "cluster_" + str(cluster) + ".xtc")


	def extract_cluster_leaders(self):

		"""
		Extract cluster leaders from cluster trajectories.

		"""

		# Create a directory where to put cluster leaders extracted from cluster trajectories

		self.cluster_leader_dir = "./cluster_leaders/"

		os.mkdir(self.cluster_leader_dir)

		# Extract a representative conformer from a given cluster trajectory. Skip HDBSCAN noise assignment (cluster -1)

		for cluster in range(self.cluster_labels.min() + 1, self.cluster_labels.max() + 1):
			clustering_leader(top=self.pdb,
							  traj=(self.traj_cluster_dir + "cluster_" + str(cluster) + ".xtc"),
							  trajnum=cluster,
							  output_dir=self.cluster_leader_dir)

		# Initialize cluster PDBs

		self.collectionPDB = glob.glob("./cluster_leaders/*")

	def get_collectionPDB(self):
		return self.collectionPDB


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



#curve = CurveSAXS("./data/HOIP_removedNaN_eom.fit")
#fit = CurveSAXS("./data/HOIP_removedNaN_HOIPwt_open.fit")

#saxsPlots.plot_saxsCurve(fit.get_q(), fit.get_iq(log=True), fit.get_fit(log=True), plot_fit=False)

#print(saxsChi.chi(curve.get_iq(), curve.get_fit(), curve.get_sigma()))
#print(saxsChi.chi(fit.get_iq(), fit.get_fit(), fit.get_sigma()))


# # Load experimental data
#
# expcurve = CurveSAXS("./data/HOIP_removedNaN.dat")
# #saxsPlots.plot_saxsCurve(expcurve.get_q(), expcurve.get_logiq(), plot_fit=False)
# print(expcurve.get_filename())
#
#
# # Load PDB files
#
# pdbs = Trajectory2("./data/pdbs/*")
# #print(pdbs.get_collectionPDB())
#
# analysis = AnalysisSAXS(pdbs)
#
# #analysis.calcFits(expcurve)
# analysis.initializeFits()
# print(expcurve.get_dataarray().shape)
# print(analysis.get_collectionFits()[0].get_dataarray().shape)
#
# print(analysis.get_fit_pairwise_matrix())
# print(analysis.get_cluster_matrix())

# for fit in analysis.get_collectionFits():
# 	print(chi(fit.get_iq(), fit.get_fit(), fit.get_sigma()))
# 	saxsPlots.plot_saxsCurve(fit.get_q(), fit.get_iq(log=True), fit.get_fit(log=True), plot_fit=True)

# analysis.calcPairwiseChiFits()
# sns.heatmap(analysis.get_fit_pairwise_matrix())
# plt.show()
#
# analysis.clusterFits()
# sns.heatmap(analysis.get_cluster_matrix())
# plt.show()





