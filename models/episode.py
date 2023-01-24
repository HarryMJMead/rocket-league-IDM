import numpy as np

class Episode():
    def __init__(self, obs, act, act_num, ad_data) -> None:
        self.observations = np.array(obs).astype('float32')
        self.actions = np.array(act).astype('float32')
        self.act_num = np.array(act_num).astype('int')
        self.add_data = np.array(ad_data).astype('int')
    
