import numpy as np
import saxsPlots
import saxsChi

class saxsCurve(object):
	
	def __init__(self, filename):
		self.filename = filename
		self.dataarray = np.loadtxt(filename)

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


curve = saxsCurve("./data/HOIP_removedNaN_eom.fit")
fit = saxsCurve("./data/HOIP_removedNaN_HOIPwt_open.fit")

saxsPlots.plot_saxsCurve(fit.get_q(), fit.get_iq(log=True), fit.get_fit(log=True), plot_fit=True)

print(saxsChi.chi(curve.get_iq(), curve.get_fit(), curve.get_sigma()))
print(saxsChi.chi(fit.get_iq(), fit.get_fit(), fit.get_sigma()))





