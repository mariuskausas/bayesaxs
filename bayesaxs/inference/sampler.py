from itertools import combinations

import numpy as np
import pymc3 as pm
import theano as tt

from bayesaxs.base import Base
import bayesaxs.basis.chi as chi
from bayesaxs.basis.curve import Curve


def _chi2_tt(exp, theor, sigma):
	"""
	Calculate chi squared (Theano method).

	This implementation is specific for PyMC3 sampling.

	This implementation does not catch division by zeros.

	Parameters
	----------
	exp : ndarray
		Numpy array (N, 1) of experimental intensities.
	theor : ndarray
		Numpy array (N, 1) of theoretical intensities.
	sigma : ndarray
		Numpy array (N, 1) of experimental errors.

	Returns
	-------
	chi2 : theano.tensor
		Chi squared value.
	"""

	chi2 = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2))

	return chi2


def _chi2red_tt(exp, theor, sigma):
	"""
	Calculate reduced chi squared (Theano method).

	This implementation is specific for PyMC3 sampling.

	This implementation does not catch division by zeros.

	Parameters
	----------
	exp : ndarray
		Numpy array (N, 1) of experimental intensities.
	theor : ndarray
		Numpy array (N, 1) of theoretical intensities.
	sigma : ndarray
		Numpy array (N, 1) of experimental errors.

	Returns
	-------
	chi2red : theano.tensor
		Reduced chi squared value.
	"""

	chi2red = tt.tensor.sum(tt.tensor.power((exp - theor) / sigma, 2)) / (exp.size - 1)

	return chi2red


def _get_chi(chi_type):
	"""
	Get type of chi-squared for likelihood.

	Parameters
	----------
	chi_type : str
		Type of chi-squared for likelihood.

	Returns
	-------
	out : func
		Function for calculating chi-squared (Theano method).
	"""

	chis = {"nonred": _chi2_tt,
			"red": _chi2red_tt}

	return chis[chi_type]


def _l1_regularization(chi2, alpha, weights):
	"""
	L1 weight regularization.

	Parameters
	----------
	chi2 : theano.tensor
		Chi squared value as Theano tensor.
	alpha : pymc3.distributions.distribution.Continuous
		PyMC3 Uniform distribution for regularization parameter.
	weights : pymc3.distributions.distribution.Continuous
		PyMC3 Dirichlet weight distribution.

	Returns
	-------
	out : theano.tensor
		Regularized likelihood.
	"""

	return (chi2 / 2.0) + alpha * tt.tensor.sum(tt.tensor.abs_(weights))


def _l2_regularization(chi2, alpha, weights):
	"""
	L2 weight regularization.

	Parameters
	----------
	chi2 : theano.tensor
		Chi squared value as Theano tensor.
	alpha : pymc3.distributions.distribution.Continuous
		PyMC3 Uniform distribution for regularization parameter.
	weights : pymc3.distributions.distribution.Continuous
		PyMC3 Dirichlet weight distribution.

	Returns
	-------
	out : theano.tensor
		Regularized likelihood.
	"""

	return (chi2 / 2.0) + alpha * tt.tensor.sum(tt.tensor.power(weights, 2))


def _get_regularization(reg_type):
	"""
	Get regularized likelihood.

	"l1" and "l2" are L1 and L2 regularizations, respectively.

	Parameters
	----------
	reg_type : str
		Regularization type.

	Returns
	-------
	out : function
		Regularization function.
	"""

	regularizations = {"l1": _l1_regularization,
					"l2": _l2_regularization}

	return regularizations[reg_type]


class Sampler(Base):
	"""
	Sampler object for performing weight Bayesian Monte Carlo.

	The object allows you to perform the following:
		1) Load Curve objects.
		2) Perform weight inference for single combination of Curve objects.
		3) Perform weight inference for single basis combination.
		4) Perform weight inference for all basis combinations.
		5) Perform summary analysis.

	Attributes
	----------
	title : str
		Title of the object.
	curves : list
		A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
	shape : int
		Number of bayesaxs.basis.scatter.Curve objects.
	exp_q : ndarray
		Numpy array (N, 1) of scattering angles.
	exp_iq : ndarray
		Numpy array (N, 1) of experimental intensities.
	exp_sigma : ndarray
		Numpy array (N, 1) of experimental errors.
	model : pymc3.model.InitContextMeta
		PyMC3 context manager.
	chi_type : str
		Type of chi-squared for likelihood.
	pm_weights : pymc3.distributions.distribution.Continuous
		PyMC3 Dirichlet weight distribution.
	pm_weighted_curve : ndarray
		Numpy array (N, 1) of weighted theoretical intensities.
	pm_chi2 : theano.tensor
		Chi squared value as Theano tensor.
	likelihood : theano.tensor
		Initial likelihood form.
	pm_likelihood : pymc3.distributions.distribution.Continuous
		PyMC3 Exponential log-likelihood.
	pm_alpha : pymc3.distributions.distribution.Continuous
		PyMC3 Uniform distribution for regularization parameter.
	pm_trace : pymc3.backends.base.MultiTrace
		PyMC3 Monte Carlo trace.
	"""

	def __init__(self, title="Unnamed"):
		""" Create a new Sampler object."""

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
		self._likelihood = None
		self._pm_likelihood = None
		self._pm_alpha = None
		self._pm_trace = None

	def __repr__(self):
		return "Sampler: {}".format(self._title)

	def load_curves(self, curves):
		"""
		Load a list of representative fits.

		Parameters
		----------
		curves : list
			A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
		"""

		self._curves = curves
		self._shape = len(self._curves)
		self._exp_q = self._curves[0].get_q()
		self._exp_iq = self._curves[0].get_iq()
		self._exp_sigma = self._curves[0].get_sigma()

		return

	def get_curves(self):
		"""
		Get a list of representative fits.

		Returns
		-------
		out : list
			A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
		"""

		return self._curves

	def get_exp_q(self):
		"""
		Get scattering angles.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of experimental intensities.
		"""

		return self._exp_q

	def get_exp_iq(self):
		"""
		Get experimental intensities.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of experimental intensities.
		"""

		return self._exp_iq

	def get_exp_sigma(self):
		"""
		Get experimental errors.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of experimental errors.
		"""

		return self._exp_sigma

	@staticmethod
	def _weighted_curve(curves, weights, shape):
		"""
		Calculate a weighted curve.

		Parameters
		----------
		curves : list
			A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
		weights : theano.tensor or ndarray
			A set of weights for producing a weighted curve.
		shape : int
			Number of bayesaxs.basis.scatter.Curve objects.

		Returns
		-------
		out : ndarray
			Numpy array (N, 1) of weighted theoretical intensities.
		"""

		return np.sum([(curves[i].get_fit() * weights[i]) for i in range(shape)], axis=0)

	def _initialize_parameters(self):
		"""
		Initialize Sampler parameters.

		Setup PyMC3 context manager, define Dirichlet prior for weights,
		define a weighted curve and set exponential log-likelihood.
		"""

		# Initialize PyMC3 model
		with self._model:
			# Initialize weights (Dirichlet distribution)
			self._pm_weights = pm.Dirichlet("w", a=np.ones(self._shape), shape=self._shape)

			# Calculate a weighted curve
			self._pm_weighted_curve = Sampler._weighted_curve(curves=self._curves, weights=self._pm_weights, shape=self._shape)
			self._pm_chi2 = _chi2_tt(exp=self._exp_iq, theor=self._pm_weighted_curve, sigma=self._exp_sigma)
			
			# Set likelihood in a form of exp(-chi2)
			self._likelihood = self._pm_chi2

	def _regularize_likelihood(self, reg_type):
		"""
		Regularize likelihood.

		Parameters
		----------
		reg_type : str
			Regularization type.
		"""

		# Update PyMC3 model
		with self._model:
			# Prior for regularization
			self._pm_alpha = pm.Uniform("alpha", lower=0, upper=1000000, shape=1)

			# Set regularized likelihood
			self._likelihood = _get_regularization(reg_type)(self._pm_chi2, self._pm_alpha, self._pm_weights)

	def _initialize_likelihood(self):
		""" Initialize PyMC3 likelihood."""

		# Update PyMC3 model
		with self._model:
			self._pm_likelihood = pm.Exponential("lam", lam=1, observed=self._likelihood)

	def _sample(self, step="nuts", num_samples=50000, chains=1):
		"""
		Perform Bayesian Monte Carlo weight inference.

		Parameters
		----------
		step : pymc3.step_methods
			PyMC3 step method.
		num_samples : int
			The number of samples to draw.
		chains : int
			The number of chains to sample.
		"""

		# Finalize PyMC3 model
		with self._model:
			# Set the Monte Carlo sampler
			if isinstance(step, str):
				step = {
					'nuts': pm.step_methods.NUTS(),
					'metropolis': pm.step_methods.Metropolis()
				}[step.lower()]

			# Perform Monte Carlo sampling
			self._pm_trace = pm.sample(draws=num_samples, step=step, chains=chains)

	def _sample_summary(self):
		"""
		Return Bayesian Monte Carlo weight inference summary.

		Returns
		-------
		out : dict
			Inference summary as a dictionary.
		"""

		# Set empty sample dictionary
		sample_summary = {}

		# Get the weights and sd
		weights = self._pm_trace['w'].mean(axis=0)
		sd = self._pm_trace['w'].std(axis=0)

		# Calculate the optimized curve and chi2red
		opt_curve = Sampler._weighted_curve(curves=self._curves, weights=weights, shape=self._shape)
		opt_chi2red = chi._chi2red_np(exp=self._exp_iq, theor=opt_curve, sigma=self._exp_sigma)

		# Prepare a Curve object
		opt_curve_array = np.hstack((self._exp_q, self._exp_iq, self._exp_sigma, opt_curve))
		curve = Curve("opt_curve")
		curve.load_curve_data(opt_curve_array)

		# Update sample summary
		sample_summary["wopt"] = weights
		sample_summary["sd"] = sd
		sample_summary["opt_curve"] = curve
		sample_summary["opt_chi2red"] = opt_chi2red
		sample_summary["trace"] = self._pm_trace

		return sample_summary

	def inference_single_combination(self, reg_type=None, **kwargs):
		"""
		Infer weights for a single combination of scattering curves.

		Parameters
		----------
		curves : list
			A list of bayesaxs.basis.scatter.Curve objects containing each representative fit.
		reg_type : str
			Option to turn on L1 or L2 regularization.

		Returns
		-------
		out : dict
			Inference summary for a single combination as a dictionary.
		"""

		# Set up a Sampler
		single_state = Sampler()

		# Load scattering curves
		single_state.load_curves(self._curves)

		# Initialization of basic Sampler parameters
		single_state._initialize_parameters()

		# Likelihood regularization
		if reg_type is not None:
			single_state._regularize_likelihood(reg_type=reg_type)

		# Initialization of likelihood
		single_state._initialize_likelihood()

		# Bayesian Monte Carlo inference
		single_state._sample(**kwargs)

		return single_state._sample_summary()

	def inference_single_basis(self, n_states,**kwargs):
		"""
		Infer weights for a single basis-set.

		Parameters
		----------
		n_states : int
			Size of the basis-set

		Returns
		-------
		out : dict
			Inference summary for a single basis-set as a dictionary.
		"""

		# Set empty states dictionary
		basis_summary = {}

		# Define unique combinations of scattering states
		combs = combinations(self._curves, n_states)

		# Perform Bayesian Monte Carlo inference for each combination
		for comb in combs:
			# Set combination title (e.g. 1:2:3)
			comb_title = ":".join([curve.get_title() for curve in comb])
			sampler = Sampler()
			sampler.load_curves(comb)
			basis_summary[comb_title] = sampler.inference_single_combination(**kwargs)

		return basis_summary

	def inference_all_basis(self, **kwargs):
		"""
		Infer weights for all basis-sets.

		Returns
		-------
		out : dict
			Inference summary for a all basis-sets as a dictionary.
		"""

		# Set empty states dictionary
		basis_summary = {}

		# Number of basis sets
		basis_sizes = [basis_size + 2 for basis_size in range(len(self._curves) - 1)]

		# Perform Bayesian Monte Carlo inference for each basis size
		for basis_size in basis_sizes:
			basis_summary[basis_size] = Sampler.inference_single_basis(self, n_states=basis_size, **kwargs)

		return basis_summary
