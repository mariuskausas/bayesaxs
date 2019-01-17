from saxsCurve import *
import matplotlib.pyplot as plt

analysis = Analysis()

analysis.initialize_fits()

analysis.calc_pairwise_chi()

analysis.cluster_fits()

analysis.extract_representative_fits()

print(analysis.get_fit_set())

print(analysis.get_repfit())



