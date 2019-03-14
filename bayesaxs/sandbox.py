from saxsCurve import *
from saxsBayes import *


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


clusterer = HDBSCAN()
clusterer.load_traj("./data/HOIPwtzn.pdb", "./data/tinytraj_fit.xtc")
clusterer.fit_predict()
clusterer.extract_traj_clusters()
clusterer.extract_cluster_leaders()



