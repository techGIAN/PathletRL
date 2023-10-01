import copy
import networkx as nx
import random as rnd

from pathlet import Pathlet

import warnings
warnings.filterwarnings("error")

class PathletGraph:

    def __init__(self,
                edge_dict,
                edge_rev_dict,
                weighted=False,
                k=10):

        G = nx.Graph()
        G.add_edges_from(edge_dict.values())
        edge_id_dict = {tuple(e_val):{'id':str(e_id)} for e_id, e_val in edge_dict.items()} 
        nx.set_edge_attributes(G, edge_id_dict)
        self.k_order = k
        self.G = G
        self.weighted = weighted

        self.ORIG_PATHLETS = dict()
        self.pathlet_dict = dict()
        self.pathlet_neighbor_dict = dict()
        self.pathlet_lengths_dict = dict()
        self.__init_pathlets()
    
        self.current_pathlet_id = ''
        self.processed_pathlets = set()
        self.processed_pathlets_len1 = set()
        self.set_current_pathlet()

    def __init_pathlets(self):

        for item in self.G.edges:
            pathlet = Pathlet(p_id=self.G.edges[item]['id'],
                              edge_path=[list(item)],
                              neighbors=None,
                              traj_cov_dict=dict(),
                              p_len=1)
            self.pathlet_dict[self.G.edges[item]['id']] = pathlet
        
        for p_id, pathlet in copy.deepcopy(self.pathlet_dict).items():
            p_neighbors = self.get_neighbor_pathlets(p_id, pathlet.edge_path)
            pathlet.neighbors = p_neighbors
            self.pathlet_neighbor_dict[p_id] = p_neighbors

            pathlet.weight = 1/len(self.pathlet_dict)
            self.pathlet_dict[p_id] = pathlet

            self.pathlet_lengths_dict[p_id] = pathlet.p_len

        self.ORIG_PATHLETS = copy.deepcopy(self.pathlet_dict)
        self.ORIG_PATHLETS_NBORS = copy.deepcopy(self.pathlet_neighbor_dict)


    # ======================= SETTER METHODS =======================

    def set_current_pathlet(self, curr_pathlet_id=None):

        if curr_pathlet_id is None:
            pathlets_to_choose_from = set(self.pathlet_dict.keys()).difference(self.processed_pathlets)
            self.current_pathlet_id = rnd.sample(list(pathlets_to_choose_from), k=1)[0]
        else:
            self.current_pathlet_id = curr_pathlet_id

    def add_processed_pathlets(self, p_id):
        self.processed_pathlets.add(p_id)

    def set_processed_pathlets(self, pp):
        self.processed_pathlets = pp

    def get_neighbor_pathlets(self, my_p_id, edge_path):

        head_node, tail_node = edge_path[0][0], edge_path[-1][-1]
        head_edge, tail_edge = edge_path[0], edge_path[-1]

        neighbor_pathlet_ids = set()
        for p_id, pathlet in self.pathlet_dict.items():

            pathlet_edge_path = pathlet.edge_path
            other_head_node, other_tail_node = pathlet_edge_path[0][0], pathlet_edge_path[-1][-1]

            if p_id != my_p_id and (head_node == other_head_node or \
                tail_node == other_head_node or head_node == other_tail_node or \
                tail_node == other_tail_node):
                neighbor_pathlet_ids.add(p_id)

        return neighbor_pathlet_ids

    def reset_pathlet_graph(self):
        
        self.processed_pathlets = set()
        self.processed_pathlets_len1 = set()
        self.pathlet_dict= copy.deepcopy(self.ORIG_PATHLETS)
        self.pathlet_neighbor_dict = copy.deepcopy(self.ORIG_PATHLETS_NBORS)
        self.pathlet_dict= self.ORIG_PATHLETS
        self.pathlet_neighbor_dict = self.ORIG_PATHLETS_NBORS
        self.pathlet_lengths_dict = {k:1 for k, v in self.pathlet_dict.items()}
        self.set_current_pathlet()
    

    # ======================= MERGE METHODS =======================

    def merge(self, p1_id, p2_id, trajectory_dictionary):

        merged = False
        pathlet1 = self.pathlet_dict[p1_id]
        pathlet2 = self.pathlet_dict[p2_id]

        if p2_id not in self.pathlet_neighbor_dict[p1_id] or pathlet1.p_len + pathlet2.p_len > self.k_order:
            return None, trajectory_dictionary
        
        new_id = '_'.join([p1_id, p2_id])
        
        p1_path = pathlet1.edge_path
        p1_rev_path = pathlet1.rev_path
        p2_path = pathlet2.edge_path
        p2_rev_path = pathlet2.rev_path
        
        tail_node = p1_path[-1][-1]
        head_node = p2_path[0][0]
        head_rev_node = p2_rev_path[0][0]
        
        if tail_node == head_node:
            traj_dict_upd = self.__merge_action(trajectory_dictionary, p1_id, p2_id, new_id, p1_path, p2_path)
            merged = True
        elif tail_node == head_rev_node:
            traj_dict_upd = self.__merge_action(trajectory_dictionary, p1_id, p2_id, new_id, p1_path, p2_rev_path)
            merged = True
            
        p1_path = pathlet1.rev_path
        tail_node = p1_path[-1][-1]
        
        if tail_node == head_node and not merged:
            traj_dict_upd = self.__merge_action(trajectory_dictionary, p1_id, p2_id, new_id, p1_path, p2_path)
            merged = True
        elif tail_node == head_rev_node and not merged:
            traj_dict_upd = self.__merge_action(trajectory_dictionary, p1_id, p2_id, new_id, p1_path, p2_rev_path)
            merged = True
            
        if merged:
            if pathlet1.p_len == 1:
                self.processed_pathlets_len1.add(p1_id)
            if pathlet2.p_len == 1:
                self.processed_pathlets_len1.add(p2_id)
            self.pathlet_lengths_dict[new_id] = pathlet1.p_len + pathlet2.p_len
            return new_id, traj_dict_upd
        else:
            raise ValueError('Investigate why it did not merge.')
    
    def __merge_action(self, trajectory_dictionary, p1_id, p2_id, new_id, edge_path1, edge_path2):

        new_edge_path = edge_path1 + edge_path2
        pathlet1 = self.pathlet_dict[p1_id]
        pathlet2 = self.pathlet_dict[p2_id]

        new_pathlet = Pathlet(p_id=new_id,
                                edge_path=new_edge_path,
                                neighbors=None,
                                traj_cov_dict=dict(),
                                p_len=pathlet1.p_len + pathlet2.p_len)

        self.pathlet_dict[new_id] = new_pathlet

        self.pathlet_dict.pop(p1_id)
        self.pathlet_dict.pop(p2_id)

        for p_id, pathlet in copy.deepcopy(self.pathlet_dict).items():
            p_neighbors = self.get_neighbor_pathlets(p_id, pathlet.edge_path)
            pathlet.neighbors = p_neighbors
            self.pathlet_neighbor_dict[p_id] = p_neighbors
            self.pathlet_dict[p_id] = pathlet

        trajectory_dictionary_updated = copy.deepcopy(trajectory_dictionary)
        for traj_id in trajectory_dictionary:
            traj = trajectory_dictionary[traj_id]
            pathlet_cov_dict = traj.pathlet_cov_dict
            if '.'.join([p1_id, p2_id]) in '.'.join(list(pathlet_cov_dict.keys())) or \
                '.'.join([p2_id, p1_id]) in '.'.join(list(pathlet_cov_dict.keys())):
                p1c = 0 if p1_id not in pathlet_cov_dict else pathlet_cov_dict[p1_id]
                p2c = 0 if p2_id not in pathlet_cov_dict else pathlet_cov_dict[p2_id]
                pathlet_cov_dict[new_id] = p1c + p2c
            if p1_id in pathlet_cov_dict:
                pathlet_cov_dict.pop(p1_id)
            if p2_id in pathlet_cov_dict:
                pathlet_cov_dict.pop(p2_id)

            traj.pathlet_cov_dict = pathlet_cov_dict
            traj.traj_pathlet_ids = list(pathlet_cov_dict.keys())

            upd_edges = []
            counter = 0
            flag = False
            edges = traj.traj_pathlet_edges
            for i in range(len(edges)):
                if flag:
                    flag = False
                    continue
                if counter == len(edges)-1:
                    upd_edges.append(edges[i])
                    break
                if i < len(edges)-1 and ([edges[i]] == pathlet1.edge_path or [edges[i]] == pathlet1.rev_path or \
                    [edges[i]] == pathlet2.edge_path or [edges[i]] == pathlet2.rev_path) and \
                    ([edges[i+1]] == pathlet1.edge_path or [edges[i+1]] == pathlet1.rev_path or \
                    [edges[i+1]] == pathlet2.edge_path or [edges[i+1]] == pathlet2.rev_path) :
                    new_mer = [edges[i], edges[i+1]]
                    upd_edges.append(new_mer)
                    flag = True
                else:
                    upd_edges.append(edges[i])
            traj.set_traj_pathlet_edges(upd_edges)
            trajectory_dictionary_updated[traj_id] = traj
        
        for traj_id, traj in trajectory_dictionary_updated.items():
            pathlet_cov_dict = traj.pathlet_cov_dict
            for pathlet_id, pathlet_cov in pathlet_cov_dict.items():
                pathlet = self.pathlet_dict[pathlet_id]
                pathlet.traj_cov_dict[traj_id] = pathlet_cov
                self.pathlet_dict[pathlet_id] = pathlet

        for p_id, pathlet in self.pathlet_dict.items():
            p_neighbors = self.get_neighbor_pathlets(p_id, pathlet.edge_path)
            pathlet.neighbors = p_neighbors 
            self.pathlet_neighbor_dict[p_id] = p_neighbors
            self.pathlet_dict[p_id] = pathlet


        for t_id, traj in trajectory_dictionary_updated.items():
            pathlet_cov_dict = traj.pathlet_cov_dict
            traj.update_weighted_pathlet_cov_dict(pathlet_cov_dict)
        return trajectory_dictionary_updated
