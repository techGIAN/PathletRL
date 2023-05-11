import warnings
warnings.filterwarnings("error")

class Trajectory:

    def __init__(self, traj_id, 
                        traj_pathlet_ids,
                        traj_pathlet_edges,
                        traj_coords,
                        traj_times=None):
        
        self.traj_id = traj_id

        if len(traj_coords) != len(traj_pathlet_ids) or len(traj_coords) != len(traj_pathlet_edges):
            raise ValueError('Input lengths are not correct.')

        self.traj_coords = traj_coords
        self.traj_pathlet_ids = traj_pathlet_ids
        self.traj_pathlet_edges = traj_pathlet_edges

        self.traj_times = traj_times if traj_times is not None else list(range(1, len(traj_coords)+1))
    
        self.TRAJ_LEN = len(traj_pathlet_ids)        # trajectory length based on pathlet lengths
        self.pathlet_cov_dict = dict()
        self.weighted_pathlet_cov_dict = dict()

        self.__TRAJ_WEIGHTED_SUM_COVERAGE = 0

    def get_representability(self, weighted=True):

        if weighted:
            return -1 if self.__TRAJ_WEIGHTED_SUM_COVERAGE == 0 else sum(self.weighted_pathlet_cov_dict.values())/self.__TRAJ_WEIGHTED_SUM_COVERAGE
        return sum(self.pathlet_cov_dict.values()) 

    def set_traj_pathlet_edges(self, traj_pathlet_edges):

        new_traj_pathlet_edges = []
        for item in traj_pathlet_edges:
            if type(item[0]) == list:
                new_l = [item[0][0]]
                for l in item:
                    new_l.append(l[1])
                new_traj_pathlet_edges.append(new_l)
            else:
                new_traj_pathlet_edges.append(item)
        self.traj_pathlet_edges = new_traj_pathlet_edges


    def compute_weighted_pathlet_cov_dict(self, pathlet_dict):

        new_dict = dict()
        for p_id, coverage in self.pathlet_cov_dict.items():
            weight = pathlet_dict[p_id].weight       # normalized weight of pathlet
            new_dict[p_id] = weight*coverage
        self.__TRAJ_WEIGHTED_SUM_COVERAGE = sum(new_dict.values())
        
        self.weighted_pathlet_cov_dict = {p_id : wc for p_id, wc in new_dict.items()}
        

    def update_weighted_pathlet_cov_dict(self, pathlet_cov_dict):

        merged_p_id = [p_id for p_id in pathlet_cov_dict if p_id not in self.weighted_pathlet_cov_dict]
        if len(merged_p_id) > 0:
            arr = merged_p_id[0].split('_')
            pid1 = '_'.join(arr[0:-1])
            pid2 = arr[-1]

            pid2_ver2 = '_'.join(arr[1:])
            pid1_ver2 = arr[0]
            try:
                self.weighted_pathlet_cov_dict[merged_p_id[0]] = self.weighted_pathlet_cov_dict[pid1] + self.weighted_pathlet_cov_dict[pid2]
                self.weighted_pathlet_cov_dict.pop(pid1)
                self.weighted_pathlet_cov_dict.pop(pid2)
            except:
                self.weighted_pathlet_cov_dict[merged_p_id[0]] = self.weighted_pathlet_cov_dict[pid1_ver2] + self.weighted_pathlet_cov_dict[pid2_ver2]
                self.weighted_pathlet_cov_dict.pop(pid1_ver2)
                self.weighted_pathlet_cov_dict.pop(pid2_ver2)

        else:
            self.weighted_pathlet_cov_dict = {p_id:val for p_id, val in self.weighted_pathlet_cov_dict.items() if p_id in pathlet_cov_dict}


    def compute_pathlet_cov_dict(self, pathlet_dict):

        new_dict = dict()
        for p_id in self.traj_pathlet_ids:
            if p_id not in pathlet_dict:
                continue
            new_dict[p_id] = pathlet_dict[p_id].p_len/self.TRAJ_LEN
        
        self.pathlet_cov_dict = new_dict