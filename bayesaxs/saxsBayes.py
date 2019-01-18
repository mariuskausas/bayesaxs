import numpy as np
import pymc3 as pm
import theano as tt
import matplotlib.pyplot as plt

from saxsCurve import *

analysis = Analysis()
analysis.initialize_fits()

curve1 = analysis.get_fit_set()[4]
curve2 = analysis.get_fit_set()[13]

# Chi-square calculation
def chi(exp, theor, error):
	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / error, 2)) / (
				exp.size - 1)  ## exp.size to be swapped with Shannon Channel
	return tt.tensor.sum(chi_value)

def _chi(exp, theor, error):

	"""
	Calculate reduced chi squared.

	"""

	# Catch division by zero errors. First do the division, then provide a zero array with the same size as the
	# original array. Finish by populating zero array with values and skip those that had a zero in a denominator.

	nominator = np.sum(np.power(np.divide((exp - theor), error, out=np.zeros_like(exp-theor), where=error != 0), 2))

	chi = np.divide(nominator, (exp.size - 1))

	return np.sum(chi)

# print(_chi(curve1.get_iq(), curve1.get_fit(), curve1.get_sigma()))
#
# plt.plot(curve1.get_q()[5:350], curve1.get_logiq()[5:350], label="exp")
# plt.plot(curve1.get_q()[5:350], curve1.get_logfit()[5:350], label="fit")
# plt.plot(curve2.get_q(), curve2.get_logfit(), label="fit")
# plt.show()
#
# w1=0.2
# w2=0.8
#
# composite = curve1.get_fit()[5:350] * w1 + curve2.get_fit()[5:350] * w2
#
# print(composite)

# chi_value = _chi(curve1.get_iq()[5:350], curve1.get_fit()[5:350], curve1.get_sigma()[5:350])

# print(chi_value)





# arr = np.linspace(0, 1, 51)
# arr1 = np.ones((51))
# arr2 = arr1 - arr
# list_of_chi = []
#
# for i in range(arr1.size):
#     composite = curve1.get_fit()[5:200] * arr1[i] + curve2.get_fit()[5:200] * arr2[i]
#     list_of_chi.append(_chi(curve1.get_iq()[5:200], composite, curve1.get_sigma()[5:200]))
#
# index = list_of_chi.index(np.min(list_of_chi))
# print("Min Chi^2 value:", np.min(list_of_chi))
# print("w1:", arr[index])
# print("w2:", arr2[index])






with pm.Model() as model:

	# Uniform variables
	w1 = pm.Uniform("w1", 0, 1)
	w2 = pm.Deterministic("w2", 1 - w1)
	lam = pm.Uniform("lam", 0, 10)

	# Preparing theoretical scattering data input
	composite = curve1.get_fit() * w1 + curve2.get_fit() * w2

	# Chi square value for composite vs experimental data
	chi_value = chi(curve1.get_iq()[5:300], composite[5:300], curve1.get_sigma()[5:300])

	# Likelihood
	target = pm.Exponential("Chi", lam=lam, observed=chi_value)

	# Sample
	step = pm.Metropolis()
	trace = pm.sample(2000, step=step, njobs=1)

pm.plot_posterior(trace)
plt.show()

# summary = {}
# varnames = trace.varnames
#
# for var in varnames:
# 	_trace = trace[var]
# 	mu = _trace.mean()
# 	sd = _trace.std()
#
# 	summary[var] = dict(zip(['mu', 'sd'], [mu, sd]))
#
# print(summary)



def summarize(trace):

	summary = pm.summary(trace).T

	stats = ['mu', 'sd']
	vars = summary.keys()








