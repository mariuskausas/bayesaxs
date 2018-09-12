import pandas as pd
import numpy as np

class saxsCurve(object):
	
	def __init__(self, filename):
		self.filename = filename
		self.dataarray = np.loadtxt(filename, skiprows=1)

	def get_filename(self):
		return self.filename
	
	def get_dataarray(self):
		return self.dataarray
		
	def get_q(self):
		return self.dataarray[:,:1]

	def get_iq(self):
		return self.dataarray[:,1:2]
	
	def get_logiq(self):
		return np.log10(self.dataarray[:,1:2])

	def get_sigma(self):
		return self.dataarray[:,2:3]


curve = saxsCurve("HOIP_removedNaN.dat")

