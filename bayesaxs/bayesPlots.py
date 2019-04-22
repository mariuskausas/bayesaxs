from bayesCurve import *
import matplotlib.pyplot as plt


def plot_fit(curve):

	q = curve.get_q()
	exp = curve.get_iq()
	fit = curve.get_fit()
	name = curve

	fig = plt.figure(figsize=[6, 6])
	ax = fig.add_subplot(111)

	ax.plot(q, np.log10(exp), label="Exp", color="tab:grey", linewidth=2)
	ax.plot(q, np.log10(fit), label=name, color="tab:blue", linewidth=2)
	ax.set_xlabel("$q$")
	ax.set_ylabel("$log_{10}(I_{q})$")
	ax.legend(loc="upper right")
	ax.set_title("Theoretical fit")

	plt.show()
	plt.close()


# analysis = Scatter()
#
# analysis.initialize_fits()
#
# analysis.calc_pairwise_chi()
#
# analysis.cluster_fits()
#
# analysis.extract_representative_fits()
#
#
# colors = ["tab:blue", "tab:red", "tab:orange", "tab:brown", "tab:green", "tab:grey", "tab:olive"]
# curves = analysis.get_fit_set()
# curves1 = analysis.get_repfit()
#
#
# plt.figure(figsize=(10, 12.5))
#
# plt.subplot(211)
#
# for indx, cluster in enumerate(analysis.get_indices_of_clusterids()):
# 	for curve in cluster:
# 		plt.plot(curves[curve].get_q(),
# 				 curves[curve].get_logfit(),
# 				 label=str(curves[curve].get_title()),
# 				 color=colors[indx],
# 				 linewidth=2)
# plt.xlim(0, 0.5)
# plt.ylim(-4, -0.8)
# plt.ylabel('$log_{10}$ $(I_{q})$', fontsize=20)
# plt.tick_params(labelsize=20)
# plt.legend()
#
# plt.subplot(212)
#
# for indx, curve in enumerate(curves1):
# 		plt.plot(curves1[indx].get_q(),
# 				 curves1[indx].get_logfit(),
# 				 label=str(curves1[indx].get_title()),
# 				 color=colors[indx],
# 				 linewidth=2)
# plt.xlim(0, 0.5)
# plt.ylim(-4, -0.8)
# plt.xlabel('$q$ $(\AA^{-1})$', fontsize=20)
# plt.ylabel('$log_{10}$ $(I_{q})$', fontsize=20)
# plt.tick_params(labelsize=20)
# plt.legend()
# plt.savefig("selected_clusters.png", dpi=300)
# plt.show()