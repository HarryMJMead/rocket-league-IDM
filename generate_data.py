import numpy as np
from time import time
import csv

import environment.create_env as rlenv
from models.sequence import Sequence

env = rlenv.make()

def generate_action(on_ground):
    continuous_action = np.random.rand(5)*2 - 1
    if on_ground:
        continuous_action[2:5] = np.zeros(3)
    else:
        continuous_action[1] = 0
    
    jump = np.random.rand(1) > 0.95

    boost_drift = np.random.rand(2) > 0.5

    return np.concatenate((continuous_action, jump, boost_drift))

def play_episode():
    obs = env.reset() 
    done = False

    ep_acts = [[], []]
    ep_obs = [[], []]

    prev_on_ground = [one_obs[1] for one_obs in obs]
    actions = [generate_action(on_ground) for on_ground in prev_on_ground]

    while not done:
        # action1 = np.concatenate((np.random.rand(5)*2 - 1, np.random.randint(0, 2, 3)))
        # action2 = np.concatenate((np.random.rand(5)*2 - 1, np.random.randint(0, 2, 3)))

        # actions = np.asarray([action1, action2])

        obs, _ , done, _ = env.step(actions)

        for i in range(2):
            ep_acts[i].append(actions[i])
            ep_obs[i].append(obs[i][0])
        
        temp_on_ground = [one_obs[1] for one_obs in obs]
        if temp_on_ground != prev_on_ground or np.random.rand() > 0.9:
            actions = [generate_action(on_ground) for on_ground in prev_on_ground]
            prev_on_ground = temp_on_ground


    # with open("out0.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(ep_obs[0])
    
    # with open("out1.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(ep_obs[1])
    
    # with open("act0.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(ep_acts[0])
    
    # with open("act1.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(ep_acts[1])

    return ep_obs, ep_acts

def generate_episodes(count: int):
    episodes = [None] * count * 2

    for i in range(0, count*2, 2):
        ep_obs, ep_acts = play_episode()
        
        for j in range(2):
            episodes[i+j] = Sequence(ep_obs[j], ep_acts[j])

    np_ep = np.array(episodes, dtype=object)
    np.savez_compressed('Episode_Data/Semi_Random_Episodes/' + str(round(time())), np_ep)


initial_time = time()
for x in range(300):
    start_time = time()
    
    generate_episodes(100)
    
    time_taken = time() - start_time
    total_time = time() - initial_time

    print('Completed: ', x, ', Batch Time: ', time_taken, ', Total Time: ', total_time)
