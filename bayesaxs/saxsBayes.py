from itertools import combinations
from six import string_types
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


def _chi_tt2(exp, theor, error):

	"""
	Calculate reduced chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / error, 2))
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

	def multi_state_inference(self, curves):

		with pm.Model():
			num_curves = len(curves)

			# Prepare weights
			w = pm.Dirichlet("w", a=np.ones(num_curves), shape=num_curves)
			lam = pm.Uniform("lam", 0, 10)

			# Calculate composite curve
			composite2 = np.sum([(curves[i].get_fit() * w[i]) for i in range(num_curves)], axis=0)
			chi_value = _chi_tt(curves[0].get_iq()[5:], composite2[5:], curves[0].get_sigma()[5:])

			# Likelihood
			target = pm.Exponential("Chi", lam=lam, observed=chi_value)

			# Sample
			step = pm.Metropolis()
			trace = pm.sample(2000, step=step, njobs=1)

			return BayesModelResults(trace).multi_state_summary()

	def run_two_state_inference(self):

		""" Perform a pairwises Bayesian inference for all given curves."""

		# Initialize an array
		n = len(self._curves)
		array = np.zeros((n, n))
		count = 1

		# Populate upper array triangle with inferred chi square values
		for i in range(n):
			# Variable count is used to populate an upper triangle
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

	def run_multi_state_inference(self, states):

		with open('multi_state{0}.txt'.format(str(states)), 'w') as f:

				combs = combinations(self._curves, states)

				for comb in combs:

					# Bayesian inference
					mu, sd = BayesModel.multi_state_inference(self, comb)

					# Composite
					composite = np.sum([(comb[i].get_fit() * mu[i]) for i in range(states)], axis=0)

					# Chi value calculation
					chi_value = _chi_np(comb[0].get_iq(),
							composite,
							comb[0].get_sigma())

					print("-----------------------------", file=f)
					print("Simulation", file=f)
					print("                             ", file=f)
					print("Chi square value: {}".format(np.round(chi_value, 3)), file=f)
					for indx, curve in enumerate(comb):
						print(curve, "mu: {} sd: {}".format(np.round(mu[indx], 3), np.round(sd[indx], 3)), file=f)
					print("                             ", file=f)
					print("-----------------------------", file=f)


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

	def multi_state_summary(self):

		""" Simple summary function to return weights and standard deviations for a multi state scattering model."""

		mu = self._trace['w'].mean(axis=0)
		sd = self._trace['w'].std(axis=0)

		return mu, sd


class BayesModel2(object):

	def __init__(self, curves):

		self._curves = curves
		self._shape = len(curves)
		self._exp_curve = self._curves[0].get_iq()
		self._exp_sigma = self._curves[0].get_sigma()
		self._model = pm.Model()
		self._dirichlet = None
		self._weighted_curve = None
		self._chi2 = None
		self._likelihood = None
		self._trace = None


class BayesSampling(BayesModel2):

	def __init__(self, curves):

		BayesModel2.__init__(self, curves)
		self._initialize_variables()

	@staticmethod
	def _composite(curves, weights, shape):
		return np.sum([(curves[i].get_fit() * weights[i]) for i in range(shape)], axis=0)

	def _initialize_variables(self):

		with self._model:

			# Initialize dirichlet weights
			self._dirichlet = pm.Dirichlet("w", a=np.ones(self._shape), shape=self._shape)

			# Calculate a weighted curve
			self._weighted_curve = BayesSampling._composite(self._curves, self._dirichlet, self._shape)
			self._chi2 = _chi_tt2(self._exp_curve[5:], self._weighted_curve[5:], self._exp_sigma[5:])

			# Set likelihood in a form of exp(-chi2)
			self._likelihood = pm.Exponential("lam", lam=1, observed=self._chi2)

	def sample(self, step, num_samples, chains=1, njobs=1):

		with self._model:

			# Set the MCMC sampler
			if isinstance(step, string_types):
				step = {
					'nuts': pm.NUTS(),
					'metropolis': pm.Metropolis()
				}[step.lower()]

			self._trace = pm.sample(num_samples, step=step, chains=chains, njobs=njobs)








