import numpy as np
import theano as tt


def _chi2_np(exp, theor, sigma):
	""" Calculate chi squared (numpy method)."""
	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.
	chi2 = np.sum(np.power(np.divide((exp - theor), sigma, out=np.zeros_like(exp-theor), where=sigma != 0), 2))
	return chi2


def _chi2red_np(exp, theor, sigma):
	""" Calculate reduced chi squared (numpy method)."""
	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.
	nominator = np.sum(np.power(np.divide((exp - theor), sigma, out=np.zeros_like(exp-theor), where=sigma != 0), 2))
	chi2red = np.divide(nominator, (exp.size - 1))
	return chi2red


def _chi2_tt(exp, theor, sigma):
	""" Calculate chi squared (theano method)."""
	chi2 = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2))
	return chi2


def _chi2red_tt(exp, theor, sigma):
	""" Calculate reduced chi squared (theano method)."""
	chi2red = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2)) / (exp.size - 1)
	return chi2red


def _pairwise_chi(curves):
	""" Calculate a pairwise reduced chi squared matrix.

	:param curves: A list containing curves to iterate over.
	:return: An array containing pairwise reduced chi squared values.
	"""

	# Generate an empty array (n,n) for a given n of curves
	number_of_curves = len(curves)
	pairwise_mat = np.zeros((number_of_curves, number_of_curves))

	# Perform a pairwise reduced chi squared calculation
	for i in range(number_of_curves):
		for j in range(number_of_curves):
			pairwise_mat[i:i + 1, j:j + 1] = _chi2red_np(curves[i].get_fit(), curves[j].get_fit(), curves[i].get_sigma())

	return pairwise_mat
