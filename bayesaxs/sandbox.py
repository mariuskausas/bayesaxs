from saxsCurve import *
from saxsBayes import *
import matplotlib.pyplot as plt

analysis = Analysis()

analysis.initialize_fits()

analysis.calc_pairwise_chi()

analysis.cluster_fits()

analysis.extract_representative_fits()

#print(analysis.get_fit_set())

print(analysis.get_repfit())



# Bayesian testing

curve1 = analysis.get_repfit()[1]
curve2 = analysis.get_repfit()[4]

# curve1 = analysis.get_fit_set()[3]
# curve2 = analysis.get_fit_set()[4]

trace = run(curve1, curve2)

w1, w2 = summarize(trace)

print(pm.summary(trace).T)

print(chi2(curve1.get_iq(),
	 curve1.get_fit() * w1 + curve2.get_fit() * w2,
	 curve1.get_sigma()))
#
# print(analysis.get_repfit())
#
# array = run_all(analysis.get_repfit())
#
# sns.heatmap(array, annot=True)
# plt.show()

