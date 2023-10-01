class Pathlet:
    
    def __init__(self, p_id, edge_path, neighbors, traj_cov_dict, p_len=1):
        self.p_id = p_id
        self.edge_path = edge_path
        self.rev_path = [item[::-1] for item in self.edge_path][::-1]
        self.neighbors = neighbors
        self.p_len = p_len
        self.traj_cov_dict = traj_cov_dict
        self.weight = -1
        
    def attributes(self):
        return {'p_id': self.p_id,
               'edge_path': self.edge_path,
               'rev_path': self.rev_path,
               'neighbors': self.neighbors,
               'length': self.p_len,
               'traj_cov_dict': self.traj_cov_dict}