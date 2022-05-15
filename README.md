bayesaxs
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/bayesaxs.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/bayesaxs)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/bayesaxs/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/bayesaxs/branch/master)

Python-based implementation of Basis-Set Supported SAXS (BSS-SAXS).

Analysis package for determination of relative probability distributions of protein conformational states by integrating molecular dynamics simulations and solution scattering data through a Maximum Parsimony approach. 

References: 

- [Multidomain assembled states of Hck tyrosine kinase in solution](https://www.pnas.org/content/107/36/15757)
- [Determining Atomistic SAXS Models of Tri-Ubiquitin Chains from Bayesian Analysis of Accelerated Molecular Dynamics Simulations](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00059)
- [BEES: Bayesian Ensemble Estimation from SAS](https://www.cell.com/biophysj/fulltext/S0006-3495%2819%2930513-2)

### Installation 

Install `conda` environment:

```
conda env create --file environment.yml
```

Activate `conda` environment:

```
conda activate bayesaxs
```

Install `bayesaxs` package:

```
pip install .
```

### Tutorial

```python
# Load the library
import bayesaxs as bs

# Initialize HDBSCAN clusterer
clustering = bs.HDBSCAN()

# # Load the trajectory
clustering.load_traj(top_path="topology.pdb", traj_path="trajectory.xtc")

# Select CA atoms
ca_atoms = clustering.get_traj().topology.select("name CA")

# Perform clustering on pairwise interdomain distances
clustering.fit_predict(metric="distances", atom_selection=ca_atoms)

# Save cluster labels
np.save("cluster_labels", clustering.get_cluster_labels())

# Load cluster labels
clustering.load_cluster_labels("cluster_labels.npy")

# Save traj clusters and extract cluster leaders
clustering.save_traj_clusters()
clustering.save_cluster_leaders(metric="distances", atom_selection=ca_atoms)

# Initialize experimental curve
curve = bs.Curve()
curve.load_txt("SAXS_curve.dat")

# Scattering clustering
analysis = bs.Scatter()
analysis.load_cluster_leaders("./cluster_leaders/")

analysis.calc_scattering(curve)
analysis.load_fits("./fits/")

analysis.calc_pairwise_chi_matrix()
analysis.cluster_fits(method="ward", metric="euclidean", cutoff_value=0.15)
analysis.calc_representative_fits()

# Bayesian Markov chain Monte Carlo sampling
repfits = bs.Scatter()
repfits.load_representative_fits("./repfits/")

sampler = bs.Sampler()
sampler.load_curves(repfits.get_representative_fits())
n_curves = len(sampler.get_curves())
states = sampler.inference_single_basis(n_states=n_curves,
                                        step="metropolis",
                                        num_samples=50000,
                                        chains=1,
                                        burn=1000)
bs.save_pickle("basis_set", states)
```


### Copyright

Copyright (c) 2022, Marius Kausas

#### Acknowledgements
 
Project based on the 
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
