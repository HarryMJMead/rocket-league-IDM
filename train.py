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
from time import time

from models.LSTM import LSTM_Model
from models.sequence import Sequence
from models.data import RandomEpisodes

wandb.init(project="Inverse Dynamics Model", entity="harrymead")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 3000,
  "batch_size": 10
}

TRAIN_PATH = 'Episode_Data\Semi_Random_Episodes'
TEST_PATH = 'Episode_Data\Semi_Random_Test_Data'

MODEL_PATH = 'Trained Networks/semi_random_car_data_only'


if __name__ == "__main__":
    train_dataset = RandomEpisodes(TRAIN_PATH)
    test_dataset = RandomEpisodes(TEST_PATH)

    Train_Data_Loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    Test_Data_Loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    model = LSTM_Model(Train_Data_Loader, Test_Data_Loader)
    #model.load(MODEL_PATH)



    averaged_loss = 1/3
    saved_average_loss = 1/3

    saved_test_loss = 1/3

    since_last_improve = 0

    for epoch in range(20):
        start_time = time()
        loss = model.train()

        test_loss = model.test()
        if test_loss < saved_test_loss:
            since_last_improve = 0
            model.save(MODEL_PATH + '_best_test')
            saved_test_loss = test_loss


        change_time = time() - start_time

        averaged_loss += 0.1*(loss - averaged_loss)
        
        print('Epoch: ', epoch, 'Loss: ', loss, ', Test Loss:', test_loss, ', Time: ', change_time)
        wandb.log({"loss": loss, "averaged_loss": averaged_loss, "test_loss": test_loss})

    model.save(MODEL_PATH)