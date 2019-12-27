import numpy as np

from bayesaxs.base import Base


class Curve(Base):
	"""
	Object to handle scattering files.

	The Curve object allows loading experimental scattering files.
	First instance, to load a file containing three columns
	describing scattering angle, intensity and error.

	In other instances, the Curve object allows loading theoretical scattering files.
	Such, files have an additional fourth column, with scattering intensities of a fit.

	Attributes
	----------
	path_to_file : str
		Full or relative path to a scattering file.
	curve_data : array
		Numpy array ((N, 3) or (N, 4)) describing scattering file.
	q : array
		Numpy array (N, 1) of scattering angles.
	iq : array
		Numpy array (N, 1) of experimental intensities.
	sigma : array
		Numpy array (N, 1) of experimental errors.
	fit : array
		Numpy array (N, 1) of theoretical intensities.
	"""

	def __init__(self, path_to_file, title="Unnamed"):
		""" Create a new Curve object."""
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
		""" Return path to the scattering file.

		Returns
		-------
		out : String of a path to a file.
		"""
		return self._path_to_file

	def get_curve_values(self):
		""" Return scattering curve data.

		Returns
		-------
		out : array
			Numpy array ((N, 3) or (N, 4)) describing scattering file.
		"""
		return self._curve_data

	def get_q(self):
		""" Return q values.
v
		Returns
		-------
		out: array
			Numpy array (N, 1) of scattering angles.
		"""
		return self._q

	def get_iq(self):
		""" Return experimental intensities.

		Returns
		-------
		out: array
			Numpy array (N, 1) of experimental intensities.
		"""
		return self._iq

	def get_logiq(self):
		""" Return log10 of experimental intensities.

		Returns
		-------
		out : array
			Numpy array (N, 1) of log10 experimental intensities.
		"""
		return np.log10(self._iq)

	def get_sigma(self):
		""" Return experimental errors.

		Returns
		-------
		out : array
			Numpy array (N, 1) of experimental errors.
		"""
		return self._sigma

	def get_fit(self):
		""" Return fit values.

		Returns
		-------
		out : array
			Numpy array (N, 1) of theoretical intensities.
		"""
		return self._fit

	def get_logfit(self):
		""" Return log10 of theoretical intensities.

		Returns
		-------
		out : array
			Numpy array (N, 1) of log10 theoretical intensities.
		"""
		return np.log10(self._fit)
