from saxsCurve import *
import saxsBayes
from saxsBayes import *
import matplotlib.pyplot as plt
import seaborn as sns


def _chi_tt(exp, theor, error):

	"""
	Calculate reduced chi squared (theano method).

	"""

	chi_value = tt.tensor.sum(tt.tensor.power((exp - theor) / error, 2)) / (
				exp.size - 1)
	return tt.tensor.sum(chi_value)


# Initialize fits

analysis = Analysis()

analysis.initialize_fits()

analysis.calc_pairwise_chi()

analysis.cluster_fits()

analysis.extract_representative_fits()

print(analysis.get_repfit())



# Bayesian testing

# bayesmodel = BayesModel()
#
# bayesmodel.load_curves(analysis.get_repfit())

# bayesmodel.run_multi_state_inference(states=4)

# array = bayesmodel.run_two_state_inference()
#
# sns.heatmap(array, annot=True, fmt=".1f")
# plt.savefig("multi_state.png", dpi=300)
# plt.show()
#plt.savefig("multi_state.png", dpi=600)

sample = BayesSampling(analysis.get_repfit())
sample.sample(step="metropolis", num_samples=50000)

print(pm.summary(sample._trace))


