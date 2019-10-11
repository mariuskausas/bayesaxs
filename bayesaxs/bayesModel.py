from itertools import combinations
import numpy as np
import pymc3 as pm
from bayesCurve import Base
import bayesChi


class BayesModel(Base):

	def __init__(self, title="BayesModel"):
		Base.__init__(self, title=title)
		self._title = title
		self._curves = None
		self._shape = None
		self._exp_q = None
		self._exp_iq = None
		self._exp_sigma = None
		self._model = pm.Model()
		self._pm_weights = None
		self._pm_weighted_curve = None
		self._pm_chi2 = None
		self._pm_likelihood = None
		self._pm_trace = None

	def __repr__(self):
		return "BayesModel: {}".format(self._title)

	def load_curves(self, curves):
		""" Load a set of representative curves."""
		self._curves = curves
		self._shape = len(self._curves)
		self._exp_q = self._curves[0].get_q()
		self._exp_iq = self._curves[0].get_iq()
		self._exp_sigma = self._curves[0].get_sigma()

	def get_curves(self):
		""" Return a set of representative curves."""
		return self._curves

	def get_exp_q(self):
		return self._exp_q

	def get_exp_iq(self):
		""" Return I(q) values for experimental curve."""
		return self._exp_iq

	def get_exp_sigma(self):
		""" Return sigma values for experimental curve."""
		return self._exp_sigma

	@staticmethod
	def _composite(curves, weights, shape):
		""" Calculate a composite curve."""
		return np.sum([(curves[i].get_fit() * weights[i]) for i in range(shape)], axis=0)

	def _initialize_parameters(self):
		""" Initialize weigths, weighted curve and set likelihood"""

		# Initialize PyMC3 model
		with self._model:
			# Initialize weights (Dirichlet distribution)
			self._pm_weights = pm.Dirichlet("w", a=np.ones(self._shape), shape=self._shape)

			# Calculate a weighted curve
			self._pm_weighted_curve = BayesModel._composite(curves=self._curves, weights=self._pm_weights, shape=self._shape)
			self._pm_chi2 = bayesChi.chi2_tt(exp=self._exp_iq, theor=self._pm_weighted_curve, sigma=self._exp_sigma)

			# Set likelihood in a form of exp(-chi2/2)
			self._pm_likelihood = pm.Exponential("lam", lam=1, observed=(self._pm_chi2 / 2.0))

	def _sample(self, step, num_samples, chains=1):
		""" Perform MCMC sampling."""

		# Finalize PyMC3 model
		with self._model:
			# Set the MCMC sampler
			if isinstance(step, str):
				step = {
					'nuts': pm.NUTS(),
					'metropolis': pm.Metropolis()
				}[step.lower()]

			# Perform MCMC sampling
			self._pm_trace = pm.sample(num_samples, step=step, chains=chains)

	def _sample_summary(self):
		""" Return MCMC sampling summary."""

		# Set empty sample dictionary
		sample_summary = {}

		# Get the weights and sd
		weights = self._pm_trace['w'].mean(axis=0)
		sd = self._pm_trace['w'].std(axis=0)

		# Calculate the optimized curve
		opt_curve = np.sum([(self._curves[i].get_fit() * weights[i]) for i in range(self._shape)], axis=0)

		# Calculate chi2red for an optimized curve
		opt_chi2 = bayesChi.chi2red_np(exp=self._exp_iq, theor=opt_curve, sigma=self._exp_sigma)

		# Update sample summary
		sample_summary["w"] = weights
		sample_summary["sd"] = sd
		sample_summary["opt_curve"] = opt_curve
		sample_summary["opt_chi2"] = opt_chi2
		sample_summary["trace"] = self._pm_trace

		return sample_summary

	def inference_single_combination(self, curves, **kwargs):
		""" Infer weights for a single combination of scattering curves."""

		# Set up a BayesModel
		single_state = BayesModel()

		# Load scattering curves
		single_state.load_curves(curves)

		# Initialization of BayesModel parameters and sampling
		single_state._initialize_parameters()
		single_state._sample(**kwargs)

		return single_state._sample_summary()

	def inference_single_basis(self, n_states, **kwargs):
		""" Infer weights for a set of unique scattering curve combinations (basis)."""

		# Set empty states dictionary
		basis_summary = {}

		# Define unique combinations of scattering states
		combs = combinations(self._curves, n_states)

		# Perform sampling for each combination
		for comb in combs:
			# Set combination title (e.g. 1:2:3)
			comb_title = ":".join([curve.get_title() for curve in comb])

			# Perform BayesModel sampling
			basis_summary[comb_title] = BayesModel.inference_single_combination(self, comb, **kwargs)

		return basis_summary

	def inference_all_basis(self, **kwargs):
		""" Infer weights for all basis sets."""

		# Set empty states dictionary
		basis_summary = {}

		# Number of basis sets
		basis_sizes = [basis_size + 2 for basis_size in range(len(self._curves) - 1)]

		# Perform sampling for each basis size
		for basis_size in basis_sizes:
			basis_summary[basis_size] = BayesModel.inference_single_basis(self, n_states=basis_size, **kwargs)

		return basis_summary
