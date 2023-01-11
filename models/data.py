import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

from models.sequence import Sequence


class RandomEpisodes(Dataset):
    def __init__(self, data_path: str):
        print('Loading Episode Data')
        self.episodes = np.concatenate([np.load(data_path + "\\" + batch_path, allow_pickle=True)['arr_0'][0::1] for batch_path in os.listdir(data_path)[0:]])
        print('Finished Loading Episode Data')


    def __len__(self):
            return len(self.episodes)


    def __getitem__(self, idx):
        _obs, act =  self.episodes[idx].get_single_obs()

        change = _obs[1:] - _obs[0:-1]

        obs = np.concatenate((_obs[1:], change*100/np.array([1, 1, 1, 5, 5, 5, 5, 5, 5, 2, 2, 2, 20, 20, 20, 2])), 1)

        return torch.Tensor(obs), torch.Tensor(act[1:])
