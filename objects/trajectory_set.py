import numpy as np
import copy
from trajectory import Trajectory
 
import warnings
warnings.filterwarnings("error")


class TrajectorySet:

    def __init__(self, traj_pathlet_ids,
                        traj_pathlet_edges,
                        pathlet_dict,
                        representability=True):
        
        self.traj_pathlet_ids = traj_pathlet_ids
        self.traj_pathlet_edges = traj_pathlet_edges
        self.representability = representability

        self.trajectories = dict()

        t_ids = self.traj_pathlet_ids
        traj_pathlet_ids = self.traj_pathlet_ids.values()
        traj_pathlet_edges = self.traj_pathlet_edges.values()

        for t_id, pathlet_ids, pathlet_edges in zip(t_ids, traj_pathlet_ids, traj_pathlet_edges):
            T = Trajectory(t_id, pathlet_ids, pathlet_edges)
            T.compute_pathlet_cov_dict(pathlet_dict)
            T.compute_weighted_pathlet_cov_dict(pathlet_dict)
            self.trajectories[t_id] = T

        self.ORIG_TRAJS = copy.deepcopy(self.trajectories)


    def get_traj_lost(self, percent=False):
        abs_lost = len(self.ORIG_TRAJS) - len(self.trajectories)
        return abs_lost if not percent else abs_lost/len(self.ORIG_TRAJS)

    def get_representativeness_dict(self, weighted=False):
        return {t_id:traj.get_representability(weighted) for t_id, traj in self.trajectories.items()}

    def set_trajectories(self, trajectories):
        traj_rep = int(self.representability)*0.99
        self.trajectories = {t_id:traj for t_id, traj in trajectories.items() if traj.get_representability(weighted=False) > traj_rep}

    def compute_pathlet_coverages_of_all_trajs(self, pathlet_dict):

        for traj_id in self.trajectories:

            new_traj = self.trajectories[traj_id]
            new_traj.compute_pathlet_cov_dict(pathlet_dict)
            new_traj.compute_weighted_pathlet_cov_dict(pathlet_dict)
            self.trajectories[traj_id] = new_traj

    def compute_ave_pathlets_rep(self):
        return np.mean([len(traj.pathlet_cov_dict) for t_id, traj in self.trajectories.items()])

    def ave_representability(self, weighted=False):
        return np.mean([traj.get_representability(weighted) for t_id, traj in self.trajectories.items()])

    def reset_trajectories(self):
        self.trajectories = copy.deepcopy(self.ORIG_TRAJS)