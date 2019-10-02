from bayesCurve import *
import matplotlib.pyplot as plt
import seaborn as sns


def _get_curve_data(curve):

	q = curve.get_q().squeeze()
	exp = curve.get_iq().squeeze()
	sigma = curve.get_sigma().squeeze()
	fit = curve.get_fit().squeeze()

	return q, exp, sigma, fit


def plot_single_scatter(curve):

	fs = 22
	fig = plt.figure(figsize=[6, 4])
	ax = fig.add_subplot(111)

	q, exp, sigma, fit = _get_curve_data(curve)
	ax.plot(q, exp, label="Exp", color="k", linewidth=2, zorder=1)
	ax.fill_between(q, exp-sigma, exp+sigma, color='black', alpha=0.2, label='Error', zorder=2)
	ax.plot(q, fit, label=curve.get_title(), linewidth=2, zorder=3)
	ax.semilogy(nonposy="mask")

	ax.tick_params(labelsize=12)

	ax.set_xlabel("$q$", fontsize=fs)
	ax.set_ylabel("$I(q)$", fontsize=fs)

	ax.grid()

	ax.legend(ncol=2, loc="upper right", fontsize=10)

	plt.show()
	return


def plot_multi_scatters(curves):

	fs = 22
	fig = plt.figure(figsize=[6, 4])
	ax = fig.add_subplot(111)

	for curve in curves:

		q, exp, sigma, fit = _get_curve_data(curve)
		ax.plot(q, fit, label=curve.get_title(), linewidth=2, zorder=3)

	ax.semilogy(nonposy="mask")

	ax.tick_params(labelsize=12)

	ax.set_xlabel("$q$", fontsize=fs)
	ax.set_ylabel("$I(q)$", fontsize=fs)

	ax.grid()

	ax.legend(loc="upper right", fontsize=10)

	plt.show()
	return


def plot_heatmap(heatmap):

	sns.heatmap(heatmap)
	plt.show()
	return
