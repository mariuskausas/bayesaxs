import os


class Base(object):

	def __init__(self, title="Unnamed"):
		self._title = title
		self._cwdir = os.path.join(os.getcwd(), '')

	def set_title(self, title):
		""" Set a new title for a curve."""
		self._title = str(title).strip()

	def get_title(self):
		""" Returns a title of a curve."""
		return self._title

	def set_cwdir(self, path):
		""" Set path for current working directory."""
		self._cwdir = os.path.normpath(os.path.join(os.getcwd(), path, ''))

	def get_cwdir(self):
		""" Get absolute path for current working directory."""
		return self._cwdir

	@staticmethod
	def _mkdir(dir_name):
		""" Create a directory."""
		if os.path.isdir(dir_name):
			print("Such folder already exists: {name}".format(name=dir_name))
			return
		else:
			os.mkdir(dir_name)
