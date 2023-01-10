import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import gym
import wandb
import os
import random

from models.LSTM import LSTMnetwork
from models.sequence import Sequence
from models.data import RandomEpisodes


TEST_PATH = 'Episode_Data\Test_Data'
test_dataset = RandomEpisodes(TEST_PATH)
Test_Data_Loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = LSTMnetwork()
gpumodel = model.cuda()
gpumodel.load_state_dict(torch.load('Trained Networks/semi_random_car_data_only_best_test.pt'))
gpumodel.eval()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(gpumodel.parameters(), lr=0.001)

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = torch.mean(differences_squared, 0)
   return mean_diff



total_loss = 0
split_loss = np.zeros(5)
obs_count = 0

for _obs, _acts in Test_Data_Loader:
    obs, acts = _obs[0], _acts[0]
    obs = obs.cuda() #[43:62 for just single agent state data]
    acts = acts[:, 0:5].cuda()

    gpumodel.hidden1 = (torch.zeros(1,1,gpumodel.hidden_size).cuda(),
                    torch.zeros(1,1,gpumodel.hidden_size).cuda())
    
    gpumodel.hidden2 = (torch.zeros(1,1,gpumodel.hidden_size).cuda(),
                    torch.zeros(1,1,gpumodel.hidden_size).cuda())
    
    #y_pred = gpumodel(torch.cat((idv_obs, idv_prev_obs), 1))
    with torch.no_grad():
        y_pred = gpumodel(obs)
    
    loss = criterion(y_pred, acts)

    total_loss += loss.item()*len(obs)
    obs_count += len(obs)

    for i in range(acts.size(0)):
        print('Prediction: ', y_pred[i])
        print('Actual: ', acts[i])
        print()
    
    split_loss += root_mean_squared_error(acts, y_pred).cpu().numpy()*len(obs)
    break

print(split_loss / obs_count)
print(total_loss / obs_count)

