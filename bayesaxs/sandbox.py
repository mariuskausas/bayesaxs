import os
import glob
import shutil
from bayesCurve import *
from bayesModel import *

# Initialize fits

# curve = Curve("./data/HOIP_removedNaN.dat")
#
# analysis = Analysis()
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
curve = Curve("./data/HOIP_removedNaN.dat")
# print(curve.get_title())
analysis = Analysis()
analysis.load_cluster_leaders("./b/*")
print(analysis.get_cluster_leaders())
analysis.simulate_scattering(curve, "fits")
print(analysis.get_fit_set())


