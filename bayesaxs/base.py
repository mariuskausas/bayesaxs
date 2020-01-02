import os


class Base(object):
	"""
	Base object designed for providing generic functionality.

	The Base object has the functionality to set title,
	set working directories and make directories.

	Attributes
	----------
	title : str
		Title of the object.
	cwdir : str
		Current working directory.
	"""

	def __init__(self, title="Unnamed"):
		"""
		Create a new Base object."""

		self._title = title
		self._cwdir = os.path.join(os.getcwd(), '')

	def set_title(self, title):
		"""
		Set a new title for a curve.

		Parameters
		-------
		title : str
			Title of the object.
		"""

		self._title = str(title).strip()
		return

	def get_title(self):
		"""
		Returns a title of a curve.

		Returns
		-------
		out : str
			Title of the object.
		"""

		return self._title

	def set_cwdir(self, path):
		"""
		Set path for current working directory.

		Parameters
		----------
		path : str
			Path of current working directory.
		"""

		self._cwdir = os.path.normpath(os.path.join(os.getcwd(), path, ''))
		return

	def get_cwdir(self):
		"""
		Get path for current working directory.

		Returns
		-------
		out : str
			Path of current working directory.
		"""

		return self._cwdir

	@staticmethod
	def _mkdir(dir_name):
		"""
		Create a directory.

		Parameters
		----------
		dir_name : str
			Name of the directory.
		"""

		if os.path.isdir(dir_name):
			print("Such folder already exists: {name}".format(name=dir_name))
			return
		else:
			os.mkdir(dir_name)
		return
