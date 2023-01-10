import numpy as np

class Sequence():
    def __init__(self, obs, act) -> None:
        self.observations = np.array(obs).astype('float32')
        self.actions = np.array(act).astype('float32')
    
    def get_single_obs(self):
        return self.observations, self.actions
    
    def get_paired_obs(self):
        return np.concatenate((self.observations[0:-1], self.observations[1:]), 1), self.actions[1:]
    
