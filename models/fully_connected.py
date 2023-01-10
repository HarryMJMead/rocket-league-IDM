import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self,input_size=162,hidden_size=256,output_size=5):
        super().__init__()

        self.linearNN = nn.Sequential(
              nn.Linear(input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size // 2),
              nn.ReLU(),
              nn.Linear(hidden_size // 2, output_size)
            )

    def forward(self,seq):
        pred = self.linearNN(seq)

        return pred