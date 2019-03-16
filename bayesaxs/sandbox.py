import os
import shutil
from bayesCurve import *
from bayesModel import *

# # Initialize fits
#
# analysis = Analysis()
# analysis.initialize_fits()
# analysis.calc_pairwise_chi()
# analysis.cluster_fits()
# analysis.extract_representative_fits()
#
# print(analysis.get_repfit())
#
# # Bayesian testing
#
# sample = BayesSampling(analysis.get_repfit())
# sample.inference_multiple(3, step="metropolis", num_samples=1000)

shutil.rmtree("./fits")
shutil.rmtree("./cluster_leaders/")
shutil.rmtree("./traj_clusters/")

clusterer = HDBSCAN()
clusterer.load_traj("./data/HOIPwtzn.pdb", "./data/tinytraj_fit.xtc")
clusterer.fit_predict()
clusterer.save_traj_clusters()
clusterer.save_cluster_leaders()

curve = Curve("./data/HOIP_removedNaN.dat")
analysis = Analysis()
analysis.load_leaders(clusterer)
analysis.simulate_scattering(curve)
print(analysis.get_fit_set())


