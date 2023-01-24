import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
from time import time
import wandb

class LSTM_Net(nn.Module):
    def __init__(self,input_size=32,lstm_size=512,hidden_size=1024):
        super().__init__()
        self.lstm_size = lstm_size
        
        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size,lstm_size)
        
        # Add a fully-connected layer:
        self.linearNN = nn.Sequential(
              nn.Linear(lstm_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
            #   nn.Linear(hidden_size, hidden_size),
            #   nn.ReLU(),
            )
        
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1,1,self.lstm_size),
                       torch.zeros(1,1,self.lstm_size))
        

        self.action = nn.Linear(hidden_size, 90)
        self.ground = nn.Linear(hidden_size, 2)
        self.jump = nn.Linear(hidden_size, 2)
        self.flip = nn.Linear(hidden_size, 2)

    def forward(self,seq):
        lstm_out, self.hidden1 = self.lstm(seq.view(seq.shape[0], 1, -1), self.hidden)

        pred = self.linearNN(lstm_out.view(seq.shape[0], -1))

        action = F.log_softmax(self.action(pred), dim=1)
        ground = F.log_softmax(self.ground(pred), dim=1)
        jump = F.log_softmax(self.jump(pred), dim=1)
        flip = F.log_softmax(self.flip(pred), dim=1)

        return action, ground, jump, flip

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    

class LSTM_Model():
    def __init__(self, Train_Data_Loader : DataLoader, Test_Data_Loader: DataLoader):
        model = LSTM_Net(input_size=51)

        print(count_parameters(model))

        self.gpumodel = model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.gpumodel.parameters(), lr=0.0001)

        self.Test_Data_Loader = Test_Data_Loader
        self.Train_Data_Loader = Train_Data_Loader
    

    def load(self, model_path):
        self.gpumodel.load_state_dict(torch.load(model_path + '.pt'))
    

    def save(self, model_path):
        torch.save(self.gpumodel.state_dict(), model_path + ".pt")
    
    def share_memory(self):
        self.gpumodel.share_memory()
    

    def train(self, log_results = False):
        LOSS_MODIFIER = [20, 1, 1, 1]

        total_loss = 0
        obs_count = 0

        start_time = time()
        batch_time = start_time
        batch = 0

        total_correct = torch.zeros(4).cuda()


        for _obs, _labels in self.Train_Data_Loader:
            obs, labels = torch.flatten(_obs, 0, 1), torch.flatten(_labels, 0, 1)
            if obs.shape[0] != 0:
                obs = obs.cuda() #[43:62 for just single agent state data]
                labels = labels.long().cuda()

                self.optimizer.zero_grad()

                self.gpumodel.hidden = (torch.zeros(1, 1, self.gpumodel.lstm_size).cuda(), 
                                torch.zeros(1, 1, self.gpumodel.lstm_size).cuda())
                
                predictions = self.gpumodel(obs)
                pred_values = torch.zeros((len(obs), 4)).cuda()
                
                loss = 0
                for i in range(4):
                    loss += self.criterion(predictions[i], labels[:, i])*LOSS_MODIFIER[i]
                    pred_values[:, i] = torch.argmax(predictions[i], dim=1)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()*len(obs)
                obs_count += len(obs)

                correct = pred_values == labels
                total_correct += torch.sum(correct, 0)

                batch += 1
                if batch % 1000 == 0:
                    idv_accuracy = total_correct.cpu().numpy() / obs_count

                    print('Batch: ', batch, ', Loss: ', total_loss / obs_count, ', Rate: ', obs_count / (time() - batch_time), ', Time: ', time() - start_time)
                    print('Action Accuracy: ', idv_accuracy[0], 'On Ground Accuracy: ', idv_accuracy[1], 
                    'Has Jump Accuracy: ', idv_accuracy[2], 'Has Flip Accuracy: ', idv_accuracy[3])
                    if log_results:
                        wandb.log({"loss": total_loss/obs_count, "accuracy/action": idv_accuracy[0], "accuracy/on_ground": idv_accuracy[1],
                        "accuracy/has_jump": idv_accuracy[2], "accuracy/has_flip": idv_accuracy[3]})
                    
                    
                    total_correct = torch.zeros(4).cuda()
                    batch_time = time()
                    total_loss = 0
                    obs_count = 0

        return total_loss / obs_count
    
    def test(self):
        LOSS_MODIFIER = [20, 1, 1, 1]

        total_loss = 0
        obs_count = 0

        start_time = time()
        batch_time = start_time
        batch = 0

        total_correct = torch.zeros(4).cuda()


        for _obs, _labels in self.Test_Data_Loader:
            obs, labels = torch.flatten(_obs, 0, 1), torch.flatten(_labels, 0, 1)
            if obs.shape[0] != 0:
                obs = obs.cuda() #[43:62 for just single agent state data]
                labels = labels.long().cuda()

                self.gpumodel.hidden = (torch.zeros(1, 1, self.gpumodel.lstm_size).cuda(), 
                                torch.zeros(1, 1, self.gpumodel.lstm_size).cuda())
                
                with torch.no_grad():
                    predictions = self.gpumodel(obs)
                pred_values = torch.zeros((len(obs), 4)).cuda()
                
                loss = 0
                for i in range(4):
                    loss += self.criterion(predictions[i], labels[:, i])*LOSS_MODIFIER[i]
                    pred_values[:, i] = torch.argmax(predictions[i], dim=1)


                total_loss += loss.item()*len(obs)
                obs_count += len(obs)

                correct = pred_values == labels
                total_correct += torch.sum(correct, 0)


        idv_accuracy = total_correct.cpu().numpy() / obs_count

        print('Batch: ', batch, ', Loss: ', total_loss / obs_count, ', Rate: ', obs_count / (time() - batch_time), ', Time: ', time() - start_time)
        print('Action Accuracy: ', idv_accuracy[0], 'On Ground Accuracy: ', idv_accuracy[1], 
        'Has Jump Accuracy: ', idv_accuracy[2], 'Has Flip Accuracy: ', idv_accuracy[3])
 

        return total_loss / obs_count
        