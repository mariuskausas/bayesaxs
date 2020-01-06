import matplotlib.pyplot as plt


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
	ax1.semilogy(nonposy="mask")
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

	plt.show()

	return


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

	ax.semilogy(nonposy="mask")

	ax.tick_params(**tick_params)

	ax.set_xlabel("$q$", fontsize=fs)
	ax.set_ylabel("$I(q)$", fontsize=fs)

	ax.legend(loc="upper right", fontsize=14)

	plt.show()

	return
