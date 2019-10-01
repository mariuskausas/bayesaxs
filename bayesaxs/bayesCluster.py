from itertools import combinations


def get_cluster_metrics(self):
	""" Get metrics for clustering."""

	metrics = {"xyz": ClusterMetrics._cluster_XYZ,
				"distances": ClusterMetrics._cluster_distances,
				"DRID": ClusterMetrics._cluster_drid}

	return metrics


def _cluster_XYZ(traj):
	""" Prepare XYZ coordinates of a trajectory for clustering."""
	temp = traj.xyz
	frames = temp.shape[0]
	atoms = temp.shape[1]
	reshaped = temp.reshape((frames, atoms * 3))
	reshaped = reshaped.astype("float64")
	temp = []
	return reshaped


def _cluster_distances(traj, atom_selection="name CA"):
	""" Calculate pair-wise atom distances of a trajectory for clustering."""
	selected_atoms = traj.topology.select(atom_selection)
	atom_pairs = list(combinations(selected_atoms, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)
	return pairwise_distances


def _cluster_drid(traj, atom_selection="name CA"):
	""" Calulate DRID representation of a trajectory for clustering."""
	selected_atoms = traj.topology.select(atom_selection)
	return mdt.compute_drid(traj=traj, atom_indices=selected_atoms)