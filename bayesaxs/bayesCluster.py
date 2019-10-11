from itertools import combinations
import mdtraj as mdt


def _get_cluster_metric(metric):
	""" Get metrics for clustering."""
	cluster_metrics = {"xyz": _cluster_XYZ,
				"distances": _cluster_distances,
				"DRID": _cluster_drid}
	return cluster_metrics[metric]


def _cluster_XYZ(traj, atom_selection):
	""" Prepare XYZ coordinates of a trajectory for clustering."""
	temp = traj.xyz[:, atom_selection]
	frames = temp.shape[0]
	atoms = temp.shape[1]
	reshaped_XYZ = temp.reshape((frames, atoms * 3))
	reshaped_XYZ = reshaped_XYZ.astype("float64")
	temp = []
	return reshaped_XYZ


def _cluster_distances(traj, atom_selection):
	""" Calculate pair-wise atom distances of a trajectory for clustering."""
	atom_pairs = list(combinations(atom_selection, 2))
	pairwise_distances = mdt.compute_distances(traj=traj, atom_pairs=atom_pairs)
	return pairwise_distances


def _cluster_drid(traj, atom_selection):
	""" Calulate DRID representation of a trajectory for clustering."""
	drid_distances = mdt.compute_drid(traj=traj, atom_indices=atom_selection)
	return drid_distances
