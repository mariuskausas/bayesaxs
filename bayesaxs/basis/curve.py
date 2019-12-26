import numpy as np

from bayesaxs.base import Base


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
