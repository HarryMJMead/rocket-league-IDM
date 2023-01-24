import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

from models.episode import Episode
from utils import quat_to_rot_mtx


class EpisodeDataset(Dataset):
    def __init__(self, data_path: str,):
        print('Loading Episode Data')
        self.episodes = np.concatenate([np.load(data_path + "/" + batch_path, allow_pickle=True)['arr_0'][0::1] for batch_path in os.listdir(data_path)[0:]])
        print('Finished Loading Episode Data')


    def __len__(self):
            return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        _obs, _act, _act_nums, _add_data = ep.observations, ep.actions, ep.act_nums, ep.add_data

        modified_obs = np.zeros((len(_obs), 51))
        modified_obs[:, 10:13] = _obs[:, 9:12]
        modified_obs[:, 13:22] = quat_to_rot_mtx(_obs[:, 12:16])
        modified_obs[:, 22:30] = _obs[:, 16:24]

        ball_close = np.linalg.norm(_obs[:, 0:3] - _obs[:, 9:12], axis=1) < (100*2.5/2300)
        modified_obs[ball_close, 0:9] = _obs[ball_close, 0:9]
        modified_obs[ball_close, 9] = 1

        car_close = np.linalg.norm(_obs[:, 24:27] - _obs[:, 9:12], axis=1) < (100*2.5/2300)
        modified_obs[car_close, 30:33] = _obs[car_close, 24:27]
        modified_obs[car_close, 33:42] = quat_to_rot_mtx(_obs[car_close, 27:31])
        modified_obs[car_close, 42:50] = _obs[car_close, 31:39]
        modified_obs[car_close, 50] = 1
        

        #return torch.Tensor(modified_obs[2:]), torch.Tensor(np.concatenate((_act[:-2], _add_data[:-2]), 1))
        return torch.Tensor(modified_obs[2:]), torch.Tensor(np.concatenate((_act_nums[:-2].reshape(-1, 1), _add_data[:-2]), 1))
