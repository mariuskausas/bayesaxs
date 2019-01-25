import numpy as np
import pymc3 as pm
import theano as tt
import matplotlib.pyplot as plt

from saxsCurve import *

exp_curve = Curve("./data/HOIP_removedNaN.dat")

analysis = Analysis()
analysis.initialize_fits()

#
# curve1 = analysis.get_fit_set()[4]
# curve2 = analysis.get_fit_set()[13]

# Chi-square calculation


def chi1(exp, theor, error):
	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / error, 2)) / (
				exp.size - 1)  ## exp.size to be swapped with Shannon Channel
	return tt.tensor.sum(chi_value)


def chi2(exp, theor, error):

	"""
	Calculate reduced chi squared.

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))

	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)


def run(curve1, curve2):

	with pm.Model() as model:

		# Uniform variables
		w1 = pm.Uniform("w1", 0, 1)
		w2 = pm.Deterministic("w2", 1 - w1)
		lam = pm.Uniform("lam", 0, 10)

		# Preparing theoretical scattering data input
		composite = curve1.get_fit() * w1 + curve2.get_fit() * w2

		# Chi square value for composite vs experimental data
		chi_value = chi1(curve1.get_iq()[5:500], composite[5:500], curve1.get_sigma()[5:500])

		# Likelihood
		target = pm.Exponential("Chi", lam=lam, observed=chi_value)

		# Sample
		step = pm.Metropolis()
		trace = pm.sample(2000, step=step, njobs=1)

	return trace

def summarize(trace):

	summary = {}
	varnames = trace.varnames

	for var in varnames:
		_trace = trace[var]
		mu = _trace.mean()
		sd = _trace.std()

		summary[var] = dict(zip(['mu', 'sd'], [mu, sd]))

	return summary['w1']['mu'], summary['w2']['mu']

def run_all(curves):

	n = len(curves)
	array = np.zeros((n, n))
	count = 1

	for i in range(n):
		for j in range(count, n):

			curve1 = curves[i]
			curve2 = curves[j]

			trace = run(curve1, curve2)

			w1, w2 = summarize(trace)

			chi_value = chi2(curve1.get_iq(),
							   (curve1.get_fit() * w1 + curve2.get_fit() * w2),
							   curve1.get_sigma())

			array[i][j] = chi_value

		count += 1

	return array










