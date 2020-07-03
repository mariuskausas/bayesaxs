import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

tick_params = dict(labelsize=22, length=10, width=1)


def _get_curve_data(curve):
	"""
	Prepare Curve data for plotting.

	Parameters
	----------
	curve : bayesaxs.basis.scatter.Curve object
		Curve object containing experimental and fit scattering values.

	Returns
	-------
	q : ndarray
		Numpy array (N, ) with scattering angles.
	exp : ndarray
		Numpy array (N, ) with experimental intensities.
	sigma : ndarray
		Numpy array (N, ) with experimental intensities.
	fit : ndarray
		Numpy array (N, ) with theoretical intensities.
	"""

	q = curve.get_q().squeeze()
	exp = curve.get_iq().squeeze()
	sigma = curve.get_sigma().squeeze()
	fit = curve.get_fit().squeeze()

	return q, exp, sigma, fit


def plot_single_fit(curve):
	""" Plot single theoretical fit and residuals to the experimental data."""

	fs = tick_params["labelsize"]
	fig = plt.figure(figsize=[8, 5])

	q, exp, sigma, fit = _get_curve_data(curve)
	ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
	ax1.plot(q, exp, label="Exp", color="k", linewidth=2, zorder=1)
	ax1.fill_between(q, exp-sigma, exp+sigma, color='black', alpha=0.2, label='Error', zorder=2)
	ax1.plot(q, fit, label="Fit : {}".format(curve.get_title()), color="tab:red", linewidth=3, zorder=3)
	ax1.semilogy(nonposy="clip")
	ax1.set_xticklabels([])
	ax1.tick_params(**tick_params)
	ax1.set_ylabel("$I(q)$", fontsize=fs)

	ax1.legend(ncol=2, loc="upper right", fontsize=14)

	residuals = (exp - fit) / sigma

	ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
	ax2.axhline(y=0, xmin=0, xmax=1, ls='--', color="k", linewidth=3, zorder=2)
	ax2.scatter(q, residuals, s=6, color="tab:red", zorder=1, marker='o', alpha=1)
	ax2.tick_params(**tick_params)
	ax2.set_xlabel("$q$", fontsize=fs)


def plot_multiple_fits(curves):
	"""
	Plot multiple fits to the experimental data.

	Parameters
	----------
	curves : list
		A list of bayesaxs.basis.scatter.Curve objects.
	"""

	fs = tick_params["labelsize"]
	fig = plt.figure(figsize=[8, 5])
	ax = fig.add_subplot(111)

	q, exp, sigma, fit = _get_curve_data(curves[0])
	ax.plot(q, exp, label="Exp", color="k", linewidth=2, zorder=1)
	ax.fill_between(q, exp-sigma, exp+sigma, color='black', alpha=0.2, label='Error', zorder=2)

	for curve in curves:
		q, exp, sigma, fit = _get_curve_data(curve)
		ax.plot(q, fit, label="Fit : {}".format(curve.get_title()), linewidth=3, zorder=3)

	ax.semilogy(nonposy="clip")

	ax.tick_params(**tick_params)

	ax.set_xlabel("$q$", fontsize=fs)
	ax.set_ylabel("$I(q)$", fontsize=fs)

	ax.legend(loc="upper right", fontsize=14)


def plot_dendogram(scatter, orientation="left", **kwargs):
	"""
	Plot a dendogram.

	Parameters
	----------
	scatter : bayesaxs.basis.scatter.Scatter
		A bayesaxs.basis.scatter.Scatter object.
	orientation : str
		The direction to plot the dendrogram.
	"""

	fig = plt.figure(figsize=[8, 8])
	ax = fig.add_subplot(111)

	sch.dendrogram(scatter.get_linkage_matrix(),
				color_threshold=scatter.get_linkage_cutoff(),
				orientation=orientation,
				**kwargs)


def plot_clusters_vs_scatters(scatter, path_to_cluster_labels):
	"""
	Plot a time-series of cluster labels coloured
	according to representative fits.

	Parameters
	----------
	scatter : bayesaxs.basis.scatter.Scatter
		A bayesaxs.basis.scatter.Scatter object.
	path_to_cluster_labels : str
		Path to the cluster label .npy file.
	"""

	# Associate each structural cluster with scattering fit cluster
	curve_pairs = list(zip(scatter.get_fits(), scatter.get_fit_cluster_indices()))
	curve_pairs_dict = {}
	for idx, curve in enumerate(curve_pairs):
		curve_pairs_dict[int(curve_pairs[idx][0].get_title())] = int(curve_pairs[idx][1])

	# Load cluster labels
	cluster_labels = np.load(path_to_cluster_labels)

	# Dolors for clusters
	n_curves = scatter.get_fit_cluster_indices().max()
	cmap = plt.get_cmap('tab20')
	colors = [cmap(i) for i in np.linspace(0, 1, n_curves)]

	# Define a sequence of colors
	sequence_of_colors = []
	for cluster_label in cluster_labels:
		if cluster_label in curve_pairs_dict.keys():
			sequence_of_colors.append(colors[curve_pairs_dict[cluster_label] - 1])
		else:
			sequence_of_colors.append("k")

	# For each cluster assign a representative fit label
	sequence_of_repfit_labels = []
	for cluster_label in cluster_labels:
		if cluster_label in curve_pairs_dict.keys():
			t = scatter.get_representative_fits()[curve_pairs_dict[cluster_label] - 1]
			sequence_of_repfit_labels.append(t.get_title())
		else:
			sequence_of_repfit_labels.append('-1')

	# Define a set of colors for each representative fit label
	colors_and_repfit_labels = list(zip(sequence_of_colors, sequence_of_repfit_labels))
	colors_and_repfit_labels_dict = {}
	for idx in range(len(colors_and_repfit_labels)):
		colors_and_repfit_labels_dict[colors_and_repfit_labels[idx][0]] = colors_and_repfit_labels[idx][1]

	# Custom labels for plotting
	custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
	custom_labels = [colors_and_repfit_labels_dict[color] for color in colors]

	# Plot
	fs = tick_params["labelsize"]
	fig = plt.figure(figsize=[8, 5])
	ax = fig.add_subplot(111)

	ax.scatter(range(cluster_labels.shape[0]), cluster_labels, s=50, c=sequence_of_colors)
	ax.set_xlabel("Frame", fontsize=fs)
	ax.set_ylabel("Cluster", fontsize=fs)
	ax.legend(custom_lines, custom_labels, loc='right', bbox_to_anchor=(1.1, 0.5), fontsize=fs - 10)
	ax.tick_params(labelsize=fs)


def plot_weights(combination):
	"""
	Plot weights for combination of fits.

	Parameters
	----------
	combination : dict
		Inference summary for a single combination as a dictionary.
	"""

	# Parse results
	comb = list(combination.keys())[0]
	clusters = comb.split(":")
	w = combination[comb]["wopt"]
	sd = combination[comb]["sd"]

	# Plot weights
	fs = tick_params["labelsize"]
	fig = plt.figure(figsize=[8, 2])
	ax = fig.add_subplot(111)

	ax.bar(clusters, w, yerr=sd, color="tab:grey", edgecolor="black")
	ax.set_ylim(0, 1)
	ax.set_xlabel("Cluster number", fontsize=fs)
	ax.set_ylabel("Weight", fontsize=fs)
	ax.tick_params(labelsize=fs - 6)
