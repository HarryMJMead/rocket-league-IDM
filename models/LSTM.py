import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np

class LSTMnetwork(nn.Module):
    def __init__(self,input_size=32,hidden_size=256,output_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Add an LSTM layer:
        self.lstm1 = nn.LSTM(input_size,hidden_size)
        self.lstm2 = nn.LSTM(hidden_size,hidden_size)
        
        # Add a fully-connected layer:
        self.linearNN = nn.Sequential(
              nn.Linear(hidden_size, hidden_size // 2),
              nn.ReLU(),
              nn.Linear(hidden_size // 2, output_size)
            )
        
        # Initialize h0 and c0:
        self.hidden1 = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))
        
        self.hidden2 = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))
        


    def forward(self,seq):
        lstm_out, self.hidden1 = self.lstm1(
            seq.view(len(seq),1,-1), self.hidden1)

        lstm_out, self.hidden2 = self.lstm2(
            lstm_out.view(len(lstm_out),1,-1), self.hidden2)

        pred = self.linearNN(lstm_out.view(len(seq),-1))
        return pred

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    

class LSTM_Model():
    def __init__(self, Train_Data_Loader : DataLoader, Test_Data_Loader: DataLoader):
        model = LSTMnetwork()
        #print(count_parameters(model))

        self.gpumodel = model.cuda()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.gpumodel.parameters(), lr=0.001)

        self.Test_Data_Loader = Test_Data_Loader

        self.Train_Data_Loader = Train_Data_Loader
    

    def load(self, model_path):
        self.gpumodel.load_state_dict(torch.load(model_path + '.pt'))
    

    def save(self, model_path):
        torch.save(self.gpumodel.state_dict(), model_path + ".pt")
    
    def share_memory(self):
        self.gpumodel.share_memory()
    

    def train(self):
        total_loss = 0
        obs_count = 0

        for _obs, _acts in self.Train_Data_Loader:
            obs, acts = _obs[0], _acts[0]
            obs = obs.cuda() #[43:62 for just single agent state data]
            acts = acts[:, 0:5].cuda()

            self.optimizer.zero_grad()
            self.gpumodel.hidden1 = (torch.zeros(1,1,self.gpumodel.hidden_size).cuda(),
                            torch.zeros(1,1,self.gpumodel.hidden_size).cuda())
            
            self.gpumodel.hidden2 = (torch.zeros(1,1,self.gpumodel.hidden_size).cuda(),
                            torch.zeros(1,1,self.gpumodel.hidden_size).cuda())
            
            #y_pred = self.gpumodel(torch.cat((idv_obs, idv_prev_obs), 1))
            y_pred = self.gpumodel(obs)
            
            loss = self.criterion(y_pred, acts)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()*len(obs)
            obs_count += len(obs)
        
        return total_loss / obs_count
    

    def test(self):
        total_loss = 0
        obs_count = 0

        for _obs, _acts in self.Test_Data_Loader:
            obs, acts = _obs[0], _acts[0]
            obs = obs.cuda() #[43:62 for just single agent state data]
            acts = acts[:, 0:5].cuda()

            self.gpumodel.hidden1 = (torch.zeros(1,1,self.gpumodel.hidden_size).cuda(),
                            torch.zeros(1,1,self.gpumodel.hidden_size).cuda())
            
            self.gpumodel.hidden2 = (torch.zeros(1,1,self.gpumodel.hidden_size).cuda(),
                            torch.zeros(1,1,self.gpumodel.hidden_size).cuda())
            
            #y_pred = self.gpumodel(torch.cat((idv_obs, idv_prev_obs), 1))
            with torch.no_grad():
                y_pred = self.gpumodel(obs)
            
            loss = self.criterion(y_pred, acts)

            total_loss += loss.item()*len(obs)
            obs_count += len(obs)
        
        return total_loss / obs_count
        