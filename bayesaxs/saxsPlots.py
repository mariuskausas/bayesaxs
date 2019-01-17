import matplotlib.pyplot as plt

def plot_saxsCurve(q, iq, fit=None, plot_fit=False):

	if plot_fit == True:
		plt.plot(q, iq)
		plt.plot(q, fit)
		plt.show()

	else:
		plt.plot(q, iq)
		plt.show()


colors = ["tab:blue", "tab:red", "tab:orange", "tab:brown", "tab:green", "tab:grey", "tab:olive"]
curves = analysis.get_fit_set()

plt.figure(figsize=(10, 10))
for indx, cluster in enumerate(analysis.get_indices_of_clusterids()):
	for curve in cluster:
		plt.plot(curves[curve].get_q(),
				 curves[curve].get_logfit(),
				 label=str(curves[curve].get_title()),
				 color=colors[indx],
				 linewidth=2)
plt.xlim(0, 0.15)
plt.ylim(-2.75, -0.8)
plt.legend()
plt.savefig("cluster_fits.png", dpi=600)
plt.show()