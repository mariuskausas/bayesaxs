import re
import glob
import subprocess
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
import saxsPlots
#import saxsChi

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

class Trajectory(object):

	def __init__(self, path2PDB):
		self.path2PDB = path2PDB
		self.collectionPDB = glob.glob(path2PDB)

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



#curve = CurveSAXS("./data/HOIP_removedNaN_eom.fit")
#fit = CurveSAXS("./data/HOIP_removedNaN_HOIPwt_open.fit")

#saxsPlots.plot_saxsCurve(fit.get_q(), fit.get_iq(log=True), fit.get_fit(log=True), plot_fit=False)

#print(saxsChi.chi(curve.get_iq(), curve.get_fit(), curve.get_sigma()))
#print(saxsChi.chi(fit.get_iq(), fit.get_fit(), fit.get_sigma()))


# Load experimental data

expcurve = CurveSAXS("./data/HOIP_removedNaN.dat")
#saxsPlots.plot_saxsCurve(expcurve.get_q(), expcurve.get_logiq(), plot_fit=False)
print(expcurve.get_filename())


# Load PDB files

pdbs = Trajectory("./data/pdbs/*")
#print(pdbs.get_collectionPDB())

analysis = AnalysisSAXS(pdbs)

#analysis.calcFits(expcurve)
analysis.initializeFits()
print(expcurve.get_dataarray().shape)
print(analysis.get_collectionFits()[0].get_dataarray().shape)

print(analysis.get_fit_pairwise_matrix())
print(analysis.get_cluster_matrix())




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





