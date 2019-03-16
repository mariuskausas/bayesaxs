import numpy as np
import theano as tt


def chi_np(exp, theor, error):

	"""
	Calculate reduced chi squared (numpy method).

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))
	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def chi2_tt(exp, theor, sigma):

	"""
	Calculate chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2))
	return tt.tensor.sum(chi_value)


def chi2red_tt(exp, theor, sigma):

	"""
	Calculate reduced chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2)) / (
				exp.size - 1)
	return tt.tensor.sum(chi_value)


def pairwise_chi(curves):

	"""
	Calculate a pairwise chi matrix.

	:param curves: A list containing curves to iterate over.
	:return: An array containing pairwise reduced chi squared values.
	"""

	# Generate an empty array (n,n) for a given n of curves
	number_of_curves = len(curves)
	pairwise_mat = np.zeros((number_of_curves, number_of_curves))

	# Perform a pairwise reduced chi squared calculation
	for i in range(number_of_curves):
		for j in range(number_of_curves):
			pairwise_mat[i:i + 1, j:j + 1] = chi_np(curves[i].get_fit(),
																curves[j].get_fit(),
																curves[i].get_sigma())

	return pairwise_mat