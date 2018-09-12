import matplotlib.pyplot as plt

def plot_saxsCurve(q, iq, fit=None, plot_fit=False):

	if plot_fit == True:
		plt.plot(q, iq)
		plt.plot(q, fit)
		plt.show()

	else:
		plt.plot(q, iq)
		plt.show()