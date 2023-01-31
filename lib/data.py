import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

from lib.episode import Episode
from lib.data_modification import just_player, rel_velocity_just_player, remove_on_ground, add_change, corrupt



class EpisodeDataset(Dataset):
    def __init__(self, data_paths, width: int, include_change = False, corrupt=False):
        self.episodes = []

        print('Loading Episode Data')
        for data_path in data_paths:
            self.episodes.extend([np.load(data_path + "/" + batch_path, allow_pickle=True)['arr_0'] for batch_path in os.listdir(data_path)])
        self.episodes = np.concatenate(self.episodes)
        print('Finished Loading Episode Data')

        self.width = width
        self.include_change = include_change
        self.corrupt = corrupt
    
    def __len__(self):
            return len(self.episodes)


    def __getitem__(self, idx):
        ep = self.episodes[idx]

        obs, act, act_num, add_data = ep.observations, ep. actions, ep.act_nums, ep.add_data
        
        #obs, act, act_num, add_data = remove_on_ground(obs, act, act_num, add_data)
        if self.corrupt:
            obs = corrupt(obs)

        obs = rel_velocity_just_player(obs)
        if self.include_change:
            obs, act, act_num, add_data = add_change(obs, act, act_num, add_data)

        stacked_obs = np.stack([obs[i:-self.width*2+1+i if -self.width*2+1+i != 0 else None] for i in range(self.width*2)], 1)
        label = act[self.width-1:-self.width] + np.array([1, 1, 1, 1, 1, 0, 0, 0])

        add_data = add_data[self.width-1:-self.width]

        return stacked_obs, label, add_data