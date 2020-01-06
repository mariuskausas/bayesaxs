import os
import pickle


def save_pickle(file_name, data):
	"""
	Save object as a pickle file.

	Pickling is performed using highest protocol.

	Parameters
	----------
	file_name : str
		Name of the pickle file.
	data : dict
		Inference results as a dictionary.
	"""

	with open(file_name + '.pkl', 'wb') as f:
		pickle.dump(obj=data, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path_to_file):
	"""
	Load pickle file.

	Parameters
	----------
	path_to_file : str
		Path to a pickle file.

	Returns
	-------
	out :
		Loaded pickle file.
	"""

	file_extension = os.path.splitext(path_to_file)[1]

	if file_extension == ".pkl":
		with open(path_to_file, 'rb') as f:
			return pickle.load(f)
	else:
		raise ValueError("File extension is not recognised. Use only '.pkl'")
