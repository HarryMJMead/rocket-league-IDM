import torch
import numpy as np

from lib.episode import Episode
from lib.utils import quats_to_rot_mtx

def just_player(obs: np.ndarray) -> np.ndarray:
    return obs[:, 9:24]


def rel_velocity_just_player(obs: np.ndarray) -> np.ndarray:
    rel_obs = obs[:,9:24]

    quat = rel_obs[:, 3:7]
    lin_vel = rel_obs[:, 7:10]
    ang_vel = rel_obs[:, 10:13]

    rot_mtx = quats_to_rot_mtx(quat)
    forward = rot_mtx[:, :, 0]
    up = rot_mtx[:, :, 2]

    rel_lin_vel = np.array([np.matmul(l_v, rot_mtx[i]) for i, l_v in enumerate(lin_vel)])
    rel_ang_vel = np.array([np.matmul(a_v, rot_mtx[i]) for i, a_v in enumerate(ang_vel)])

    rel_obs = np.concatenate([rel_obs[:, 0:3], forward, up, rel_obs[:, 7:15], rel_lin_vel, rel_ang_vel], axis=1)

    return np.float32(rel_obs)


def remove_on_ground(obs, act, act_num, add_data):
    not_on_ground = add_data[:, 0] == 0

    if np.sum(not_on_ground) != 0:
        return obs[not_on_ground], act[not_on_ground], act_num[not_on_ground], add_data[not_on_ground]
    
    else:
        return obs[0:1], act[0:1], act_num[0:1], add_data[0:1]

def add_change(obs, act, act_num, add_data):
    change = (obs[1:] - obs[:-1]) * 100 / np.array([1, 1, 1, 5, 5, 5, 5, 5, 5, 2, 2, 2, 20, 20, 20, 2, 1, 2, 2, 2, 20, 20, 20])

    return np.float32(np.concatenate((obs[:-1], change), 1)), act[:-1], act_num[:-1], add_data[:-1]