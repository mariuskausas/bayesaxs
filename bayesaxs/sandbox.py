import os
import glob
import shutil
from bayesCurve import *
from bayesModel import *
from bayesPlots import *
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize fits

# curve = Curve("./data/HOIP_removedNaN.dat")
#
# analysis = Scatter()
# analysis.load_fits("fits/")
# print(analysis.get_fit_set())
# analysis.load_leaders(glob.glob("./cluster_leaders/*"))
# analysis.simulate_scattering(curve)
# analysis.calc_pairwise_chi()
# analysis.cluster_fits()
# analysis.extract_representative_fits()
# print(analysis.get_repfit())
#
# # Bayesian testing
#
# sample = BayesSampling(analysis.get_repfit())
# sample.inference_multiple(2, step="metropolis", num_samples=1000)

# shutil.rmtree("./fits")
# shutil.rmtree("./cluster_leaders/")
# shutil.rmtree("./traj_clusters/")
#
# clusterer = HDBSCAN()
# print(clusterer.get_cwdir())
# print(os.path.isdir("a"))
# print(os.path.join(clusterer.get_cwdir(), "folder_name", ''))
# clusterer.load_traj("./data/HOIPwtzn.pdb", "./data/tinytraj_fit.xtc")
# clusterer.fit_predict()
# clusterer.save_traj_clusters("a")
# clusterer.save_cluster_leaders("b")
# print(clusterer.get_cluster_leaders())
#
# curve = Curve("./data/HOIP_removedNaN.dat")
# print(curve.get_title())
analysis = Scatter()
# analysis.load_cluster_leaders("./b/*")
# print(analysis.get_cluster_leaders())
# analysis.simulate_scattering(curve, "fits")
# print(analysis.get_fit_set())
# analysis.get_crysol_summary()
analysis.load_fits("./fits/*.fit")
print(analysis.get_fit_set())
analysis.calc_pairwise_chi_matrix()
print(analysis.get_pairwise_chi_matrix())

#
analysis.cluster_fits(0.25)
# # print(analysis.get_fit_cluster_indices())
# # print(analysis.get_indices_of_clusterids())
print(analysis.get_sorted_pairwise_chi_matrix())

# print(analysis.get_linkage_matrix().max())
# # plt.show()

# print(analysis.get_fit_set()[10].get_q())

#plot_single_scatter(analysis.get_fit_set()[0])
#plot_multi_scatters(analysis.get_fit_set())

