import numpy as np
import copy
import random as rnd

from objects.pathlet import Pathlet
from objects.pathlet_graph import PathletGraph
from objects.trajectory import Trajectory
from objects.trajectory_set import TrajectorySet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec

import warnings
warnings.filterwarnings("error")

class PathletGraphEnvironment(py_environment.PyEnvironment):

    def __init__(self,
                edge_dict,
                edge_rev_dict,
                traj_dict,
                traj_edge_dict,
                traj_coords,
                weighted=False,
                k=10,
                traj_loss_max=0.25,
                rep_threshold=0.8,
                alpha=[1/4, 1/4, 1/4, 1/4],
                representability=True):

        self.pg = PathletGraph(edge_dict, edge_rev_dict, weighted=weighted, k=k)
        self.ts = TrajectorySet(traj_dict, traj_edge_dict, self.pg.pathlet_dict, representability)
        trajectory_dictionary = self.ts.trajectories

        for traj_id, traj in trajectory_dictionary.items():
            pathlet_cov_dict = traj.pathlet_cov_dict
            for pathlet_id, pathlet_cov in pathlet_cov_dict.items():
                pathlet = self.pg.pathlet_dict[pathlet_id]
                pathlet.traj_cov_dict[traj_id] = pathlet_cov
                self.pg.pathlet_dict[pathlet_id] = pathlet

        if weighted:
            for p_id, pathlet in copy.deepcopy(self.pg.pathlet_dict).items():
                traj_count = len(pathlet.traj_cov_dict)         
                pathlet_importance = traj_count/len(trajectory_dictionary)     
                pathlet.weight = pathlet_importance
                self.pg.pathlet_dict[p_id] = pathlet
        self.weighted = weighted

        self.ts.compute_pathlet_coverages_of_all_trajs(self.pg.pathlet_dict)
        self.pg.set_current_pathlet()
        self.representability = representability
            
        self.PATHLET_GRAPH_SIZE = len(edge_dict)
        self.REPRESENTATION = self.ts.compute_ave_pathlets_rep()
        self.TRAJ_COUNT = len(self.ts.trajectories)
        self.TRAJ_LOSS_MAX = int(self.TRAJ_COUNT*traj_loss_max)
        self.ALPHA = alpha
        self.REP_THRESHOLD = rep_threshold
        
        self.processed_pathlets = set()
        self.processed_pathlets_len1 = set()
        self.begin_flag = True

        self.MAX_ACTION = max(len(val) for val in self.pg.pathlet_neighbor_dict.values())
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.MAX_ACTION, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(2,), dtype=np.int32, name='observation')
        self._state = np.array([self.PATHLET_GRAPH_SIZE, self.TRAJ_COUNT], dtype=np.int32)
        self._episode_ended = False
        
        self.epi = 0
        self.reward_val = 0
        self.step_count = 0

        self.S_init, self.phi_init, self.traj_loss_init, self.representability_init = self.PATHLET_GRAPH_SIZE, self.REPRESENTATION, 0, 0

        self.S_prev, self.S_curr = self.PATHLET_GRAPH_SIZE, self.PATHLET_GRAPH_SIZE
        self.phi_prev, self.phi_curr = self.REPRESENTATION, self.REPRESENTATION
        self.traj_loss_prev, self.traj_loss_curr = 0, 0
        self.representability_prev, self.representability_curr = 0, 0


        self.pathlet_objs_prev, self.pathlet_objs_curr = None, None
        self.traj_objs_prev, self.traj_objs_curr = None, None
        self.processed_pathlets_prev, self.processed_pathlets_curr = None, None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        
        if self.begin_flag:
            self.begin_flag = False
        else:
            self.pg.reset_pathlet_graph()
            self.ts.reset_trajectories()
            for traj_id, traj in self.ts.trajectories.items():
                pathlet_cov_dict = traj.pathlet_cov_dict
                for pathlet_id, pathlet_cov in pathlet_cov_dict.items():
                    pathlet = self.pg.pathlet_dict[pathlet_id]
                    pathlet.traj_cov_dict[traj_id] = pathlet_cov
                    self.pg.pathlet_dict[pathlet_id] = pathlet

            if self.weighted:
                traj_dict = self.ts.trajectories
                for p_id, pathlet in copy.deepcopy(self.pg.pathlet_dict).items():
                    traj_count = len(pathlet.traj_cov_dict)         
                    pathlet_importance = traj_count/len(traj_dict)     
                    pathlet.weight = pathlet_importance
                    self.pg.pathlet_dict[p_id] = pathlet


            self.ts.compute_pathlet_coverages_of_all_trajs(self.pg.pathlet_dict)
            self.pg.set_current_pathlet()
        
        self.epi += 1
        self.step_count = 0
        self.reward_val = 0

        self._episode_ended = False
        self._state = np.array([self.PATHLET_GRAPH_SIZE, 0], dtype=np.int32)
        
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        if self.step_count == 0 and action == 0:
            action = self.__change_init_action()

        if action == 0:

            self.pg.processed_pathlets.add(self.pg.current_pathlet_id)

            self.processed_pathlets_prev = self.processed_pathlets_curr
            self.processed_pathlets_curr = self.pg.processed_pathlets

            if len( set(self.pg.pathlet_dict.keys()).difference(self.pg.processed_pathlets) )  == 0: 
                self._episode_ended = True

            else:
                self.pg.set_current_pathlet()
                self.S_prev = self.S_curr
                self.S_curr = -np.inf

                self.phi_prev = self.phi_curr
                self.phi_curr = -np.inf

                self.traj_loss_prev = self.traj_loss_curr
                self.traj_loss_curr = -np.inf
                
                self.representability_prev = self.representability_curr
                self.representability_curr = -np.inf

        else:
            if len(self.pg.pathlet_dict[self.pg.current_pathlet_id].neighbors) == 0:
                
                if len( set(self.pg.pathlet_dict.keys()).difference(self.pg.processed_pathlets) )  == 0: 
                    self._episode_ended = True
                else:
                    self.pg.set_current_pathlet()

            else:

                neigh_ix = self.get_action_neigh_ix_mapping(self.pg.pathlet_dict[self.pg.current_pathlet_id].neighbors, action)
                p_id = list(self.pg.pathlet_dict[self.pg.current_pathlet_id].neighbors)[neigh_ix]
                new_p_id, traj_dict_upd = self.pg.merge(self.pg.current_pathlet_id,
                                                        p_id,
                                                        self.ts.trajectories)
                if new_p_id is not None:
                    self.ts.set_trajectories(traj_dict_upd)
                
                self.pg.set_current_pathlet(new_p_id)

        self.step_count += 1

        self.S_prev = self.S_curr
        self.S_curr = len(self.pg.pathlet_dict)

        self.phi_prev = self.phi_curr
        self.phi_curr = self.ts.compute_ave_pathlets_rep()

        self.traj_loss_prev = self.traj_loss_curr
        self.traj_loss_curr = self.ts.get_traj_lost(percent=True)
        
        self.representability_prev = self.representability_curr
        self.representability_curr = self.ts.ave_representability(weighted=self.pg.weighted)
        
        self.pathlet_objs_prev = self.pathlet_objs_curr
        self.pathlet_objs_curr = self.pg.pathlet_dict

        self.traj_objs_prev = self.traj_objs_curr
        self.traj_objs_curr = self.ts.trajectories

        self.processed_pathlets_prev = self.processed_pathlets_curr
        self.processed_pathlets_curr = self.pg.processed_pathlets


        if not self.representability:
            if len([k for k, v in self.ts.get_representativeness_dict(weighted=self.pg.weighted).items() if v < 1.0])/len(self.ts.ORIG_TRAJS) >= self.TRAJ_LOSS_MAX:
                self.__undo_merge()
                self._episode_ended = True
        else:
            if self.ts.get_traj_lost() >= self.TRAJ_LOSS_MAX or np.mean(list(self.ts.get_representativeness_dict(weighted=True).values())) < self.REP_THRESHOLD:
                self.__undo_merge()
                self._episode_ended = True

        S_prev, S_curr = self.S_prev, self.S_curr
        phi_prev, phi_curr = self.phi_prev, self.phi_curr
        traj_loss_prev, traj_loss_curr = self.traj_loss_prev, self.traj_loss_curr
        representability_prev, representability_curr = self.representability_prev, self.representability_curr

        S1 = 0 if S_prev == -np.inf or S_curr == -np.inf else (S_prev - S_curr)/S_prev
        S2 = 0 if phi_prev == -np.inf or phi_curr == -np.inf else (phi_prev - phi_curr)/phi_prev
        S3 = 0 if traj_loss_curr == 0 or traj_loss_prev == -np.inf or (traj_loss_prev == traj_loss_curr and traj_loss_prev != 0) else (traj_loss_prev - traj_loss_curr)/max(1,S_prev)
        S4 = 0 if representability_prev == 0 or representability_prev == -np.inf or representability_curr == -np.inf else (representability_prev - representability_curr)/representability_prev

        a1, a2, a3, a4 = self.ALPHA
        reward = a1*S1 + a2*S2 + a3*S3 - a4*S4
        self.reward_val += reward

        if self._episode_ended:
            
            S1 = (self.S_init - S_curr)/self.S_init
            S2 = (self.phi_init - phi_curr)/self.phi_init
            S3 = 1-traj_loss_curr
            S4 = representability_curr

            extra_reward = a1*S1 + a2*S2 + a3*S3 - a4*S4
            self.reward_val += extra_reward

            return ts.termination(self._state, self.reward_val)
        
        else:
            return ts.transition(self._state, reward=self.reward_val, discount=0.99)
    
    def episode_attributes(self):
        return {
            '|S|' : len(self.pg.pathlet_dict),
            'phi' : self.ts.compute_ave_pathlets_rep(),
            'ave_repr': self.ts.ave_representability(),
            'ave_w_repr': self.ts.ave_representability(weighted=True),
            'L_traj' : self.ts.get_traj_lost(),
            'L_traj_%' : self.ts.get_traj_lost(percent=True),
            'R' : self.get_reward(),
            'S' : list(self.pg.pathlet_dict.keys())
        }
    
    def get_reward(self):
        return self.reward_val

    def __change_init_action(self):
        return rnd.randint(1, self.MAX_ACTION)
    
    def get_action_neigh_ix_mapping(self, neighbors, action):
        num_of_neighbors = len(neighbors)
        max_action = self.MAX_ACTION
        interval = max_action/num_of_neighbors

        lb = [i*interval for i in range(max_action)]
        ub = [(i+1)*interval for i in range(max_action)]
        bin_bools = [lb[i] < action <= ub[i] for i in range(max_action)]
        
        return np.where(bin_bools)[0][0]

    def __undo_merge(self):

        self.pg.pathlet_dict = self.pathlet_objs_prev
        self.pg.processed_pathlets = self.processed_pathlets_prev
        self.ts.set_trajectories(self.traj_objs_prev)