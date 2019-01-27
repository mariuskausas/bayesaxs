import numpy as np
import pymc3 as pm
import theano as tt
import matplotlib.pyplot as plt

from saxsCurve import *

# Utilities


def _chi_np(exp, theor, error):

	"""
	Calculate reduced chi squared (numpy method).

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))
	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def _chi_tt(exp, theor, error):

	"""
	Calculate reduced chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / error, 2)) / (
				exp.size - 1)
	return tt.tensor.sum(chi_value)


def summarize(trace):

		""" Simple summary function to return weights for a two state scattering model."""

		# Prepare a dictionary to store trace information
		summary = {}
		varnames = trace.varnames

		# For each trace variable extract mean and standard deviation
		for var in varnames:
			_trace = trace[var]
			mu = _trace.mean()
			sd = _trace.std()

			summary[var] = dict(zip(['mu', 'sd'], [mu, sd]))

		return summary['w1']['mu'], summary['w2']['mu']


class BayesModel(object):

	def __init__(self, title="BayesModel"):

		self._title = title
		self._curves = None
		self._trace = None

	def __repr__(self):

		return "BayesModel: {}".format(self._title)

	def get_title(self):

		""" Returns a title of a Bayesian model."""

		return self._title

	def set_title(self, title):

		""" Set a new title for Bayesian model."""

		self._title = str(title)

	def load_curves(self, curves):

		""" Load a set of representative curves."""

		self._curves = curves

	def get_curves(self):

		""" Return a set of representative curves."""

		return self._curves

	def two_state_inference(self, curve1, curve2):

		""" Bayesian inference of a two state scattering model."""

		with pm.Model():

			# Prepare weights
			w1 = pm.Uniform("w1", 0, 1)
			w2 = pm.Deterministic("w2", 1 - w1)
			lam = pm.Uniform("lam", 0, 10)

			# Prepare a weighted scattering curve
			composite = curve1.get_fit() * w1 + curve2.get_fit() * w2

			# Chi square value for composite vs experimental data
			chi_value = _chi_tt(curve1.get_iq()[5:], composite[5:], curve1.get_sigma()[5:])

			# Likelihood
			target = pm.Exponential("Chi", lam=lam, observed=chi_value)

			# Sample
			step = pm.Metropolis()
			self._trace = pm.sample(2000, step=step, njobs=1)

			return BayesModelResults(self._trace).two_state_summary()

	def run_two_state_inference(self):

		""" Perform a pairwises Bayesian inference for all given curves."""

		# Initialize an array
		n = len(self._curves)
		array = np.zeros((n, n))
		count = 1

		# Populate upper array triangle with inferred chi square values
		for i in range(n):
			# Variable ount is used to populate an upper triangle
			for j in range(count, n):

				# Define two scattering curves
				curve1 = self._curves[i]
				curve2 = self._curves[j]

				# Bayesian inference
				w1, w2 = BayesModel.two_state_inference(self, curve1, curve2)

				# Chi square value calculation
				array[i][j] = _chi_np(curve1.get_iq(),
							(curve1.get_fit() * w1 + curve2.get_fit() * w2),
							curve1.get_sigma())

			# Variable ount is used to populate an upper triangle
			count += 1

		return array


class BayesModelResults(object):

	def __init__(self, trace):
		self._trace = trace

	def two_state_summary(self):

		""" Simple summary function to return weights for a two state scattering model."""

		# Prepare a dictionary to store trace information
		summary = {}
		varnames = self._trace.varnames

		# For each trace variable extract mean and standard deviation
		for var in varnames:
			_trace = self._trace[var]
			mu = _trace.mean()
			sd = _trace.std()

			summary[var] = dict(zip(['mu', 'sd'], [mu, sd]))

		return summary['w1']['mu'], summary['w2']['mu']










