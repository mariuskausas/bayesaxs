from itertools import combinations
import numpy as np
import pymc3 as pm
import theano as tt

from saxsCurve import *


# Utilities


def _chi_np(exp, theor, sigma):

	"""
	Calculate reduced chi squared (numpy method).

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), sigma, out=np.zeros_like(exp-theor), where=sigma != 0), 2))
	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def _chi2_tt(exp, theor, sigma):

	"""
	Calculate chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2))
	return tt.tensor.sum(chi_value)


def _chi2red_tt(exp, theor, sigma):

	"""
	Calculate reduced chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2)) / (
				exp.size - 1)
	return tt.tensor.sum(chi_value)


class BayesModel(object):

	def __init__(self, curves, title="BayesModel"):

		self._title = title
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


class BayesSampling(BayesModel):

	def __init__(self, curves):

		BayesModel.__init__(self, curves)
		self._initialize_variables()

	def _initialize_variables(self):

		with self._model:

			# Initialize dirichlet weights
			self._dirichlet = pm.Dirichlet("w", a=np.ones(self._shape), shape=self._shape)

			# Calculate a weighted curve
			self._weighted_curve = BayesSampling._composite(curves=self._curves,
															weights=self._dirichlet,
															shape=self._shape)
			self._chi2 = _chi2_tt(exp=self._exp_curve[5:],
								theor=self._weighted_curve[5:],
								sigma=self._exp_sigma[5:])

			# Set likelihood in a form of exp(-chi2)
			self._likelihood = pm.Exponential("lam", lam=1, observed=self._chi2)

	@staticmethod
	def _composite(curves, weights, shape):

		return np.sum([(curves[i].get_fit() * weights[i]) for i in range(shape)], axis=0)

	def _inference(self, step, num_samples, chains=1, njobs=1):

		with self._model:

			# Set the MCMC sampler
			if isinstance(step, str):
				step = {
					'nuts': pm.NUTS(),
					'metropolis': pm.Metropolis()
				}[step.lower()]

			self._trace = pm.sample(num_samples, step=step, chains=chains, njobs=njobs)

	def _weight_summary(self):

		""" Simple summary function to return weights and standard deviations for a multi state scattering model."""

		mu = self._trace['w'].mean(axis=0)
		sd = self._trace['w'].std(axis=0)

		return mu, sd

	def _summary_output(self, mu, sd, final_chi2):

		print("-----------------------------")
		print("Simulation")
		print("                             ")
		print("Chi square value: {}".format(np.round(final_chi2, 3)))
		for indx, curve in enumerate(self._curves):
			print(curve, "mu: {} sd: {}".format(np.round(mu[indx], 3), np.round(sd[indx], 3)))
		print("                             ")
		print("-----------------------------")

	def _summary(self):

		# Get the weights
		mu, sd = BayesSampling._weight_summary(self)

		# Calculate the weighted curve chi2red
		final_weighted_curve = np.sum([(self._curves[i].get_fit() * mu[i]) for i in range(self._shape)], axis=0)
		final_chi2 = _chi_np(exp=self._exp_curve, theor=final_weighted_curve, sigma=self._exp_sigma)

		# Print the summary output
		BayesSampling._summary_output(self, mu=mu, sd=sd, final_chi2=final_chi2)

	@staticmethod
	def _inference_single(curves, **kwargs):

		single_state = BayesSampling(curves)
		single_state._inference(**kwargs)
		single_state._summary()

	def inference_multiple(self, states, **kwargs):

		combs = combinations(self._curves, states)
		for comb in combs:
			BayesSampling._inference_single(comb, **kwargs)









