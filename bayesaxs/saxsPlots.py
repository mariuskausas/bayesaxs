from saxsCurve import *
from saxsBayes import *
import matplotlib.pyplot as plt

def plot_saxsCurve(q, iq, fit=None, plot_fit=False):

	if plot_fit == True:
		plt.plot(q, iq)
		plt.plot(q, fit)
		plt.show()

	else:
		plt.plot(q, iq)
		plt.show()

analysis = Analysis()

analysis.initialize_fits()

analysis.calc_pairwise_chi()

analysis.cluster_fits()

analysis.extract_representative_fits()


colors = ["tab:blue", "tab:red", "tab:orange", "tab:brown", "tab:green", "tab:grey", "tab:olive"]
curves = analysis.get_fit_set()
curves1 = analysis.get_repfit()


plt.figure(figsize=(10, 12.5))

plt.subplot(211)

for indx, cluster in enumerate(analysis.get_indices_of_clusterids()):
	for curve in cluster:
		plt.plot(curves[curve].get_q(),
				 curves[curve].get_logfit(),
				 label=str(curves[curve].get_title()),
				 color=colors[indx],
				 linewidth=2)
plt.xlim(0, 0.5)
plt.ylim(-4, -0.8)
plt.ylabel('$log_{10}$ $(I_{q})$', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend()

plt.subplot(212)

for indx, curve in enumerate(curves1):
		plt.plot(curves1[indx].get_q(),
				 curves1[indx].get_logfit(),
				 label=str(curves1[indx].get_title()),
				 color=colors[indx],
				 linewidth=2)
plt.xlim(0, 0.5)
plt.ylim(-4, -0.8)
plt.xlabel('$q$ $(\AA^{-1})$', fontsize=20)
plt.ylabel('$log_{10}$ $(I_{q})$', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend()
plt.savefig("selected_clusters.png", dpi=300)
plt.show()