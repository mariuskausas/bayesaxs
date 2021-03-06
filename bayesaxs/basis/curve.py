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
	curve_data : ndarray
		Numpy array ((N, 3) or (N, 4)) describing scattering file.
	path_to_file : str
		Full or relative path to a scattering file.
	q : ndarray
		Numpy array (N, 1) of scattering angles.
	iq : ndarray
		Numpy array (N, 1) of experimental intensities.
	sigma : ndarray
		Numpy array (N, 1) of experimental errors.
	fit : ndarray
		Numpy array (N, 1) of theoretical intensities.
	"""

	def __init__(self, title="Unnamed"):
		""" Create a new Curve object."""

		Base.__init__(self, title=title)
		self._curve_data = None
		self._path_to_file = None
		self._q = None
		self._iq = None
		self._sigma = None
		self._fit = None

	def __repr__(self):
		return "Curve: {}".format(self._title)

	def _initialize_curve_values(self):
		""" Initialize curve scattering angles, intensities, errors and fit values. """

		self._q = self._curve_data[:, :1]
		self._iq = self._curve_data[:, 1:2]
		self._sigma = self._curve_data[:, 2:3]
		self._fit = self._curve_data[:, 3:4]

		return

	def load_txt(self, path_to_file):
		"""
		Load a scattering file.

		Parameters
		----------
		path_to_file : str
			Full or relative path to a scattering file.
		"""

		self._curve_data = np.loadtxt(path_to_file)
		self._path_to_file = path_to_file
		Curve._initialize_curve_values(self)

		return

	def load_curve_data(self, curve_data):
		"""
		Load Numpy array into Curve object.

		Parameters
		----------
		curve_data : ndarray
			Numpy array ((N, 3) or (N, 4)) describing scattering file.
		"""

		self._curve_data = curve_data
		Curve._initialize_curve_values(self)

		return

	def get_path_to_file(self):
		"""
		Get path to the scattering file.

		Returns
		-------
		out : String of a path to a file.
		"""

		return self._path_to_file

	def save_txt(self, output_name):
		"""
		Save Curve columns as a .fit file.

		Parameters
		----------
		output_name : str
			Output file name.
		"""

		np.savetxt(fname=output_name + ".fit", X=self._curve_data)

		return

	def get_curve_values(self):
		"""
		Get scattering curve data.

		Returns
		-------
		out : ndarray
			Numpy array ((N, 3) or (N, 4)) describing scattering file.
		"""

		return self._curve_data

	def get_q(self):
		"""
		Get q values.

		Returns
		-------
		out: ndarray
			Numpy array (N, 1) of scattering angles.
		"""

		return self._q

	def get_iq(self):
		"""
		Get experimental intensities.

		Returns
		-------
		out: ndarray
			Numpy array (N, 1) of experimental intensities.
		"""

		return self._iq

	def get_logiq(self):
		"""
		Get log10 of experimental intensities.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of log10 experimental intensities.
		"""

		return np.log10(self._iq)

	def get_sigma(self):
		"""
		Get experimental errors.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of experimental errors.
		"""

		return self._sigma

	def get_fit(self):
		"""
		Get fit values.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of theoretical intensities.
		"""

		return self._fit

	def get_logfit(self):
		"""
		Get log10 of theoretical intensities.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of log10 theoretical intensities.
		"""

		return np.log10(self._fit)
