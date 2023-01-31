import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, obs_size=21, obs_width=2, hidden_size=256):
        super().__init__()
        self.input_size = obs_size*obs_width*2

        self.linearNN = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU()
            )
        
        self.throttle = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.steer = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.pitch = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.yaw = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.roll = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.jump = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.boost = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.drift = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )

    def forward(self,seq):
        pred = self.linearNN(seq.view(-1, self.input_size))

        throttle = F.log_softmax(self.throttle(pred), dim=1)
        steer = F.log_softmax(self.steer(pred), dim=1)
        pitch = F.log_softmax(self.pitch(pred), dim=1)
        yaw = F.log_softmax(self.yaw(pred), dim=1)
        roll = F.log_softmax(self.roll(pred), dim=1)
        jump = F.log_softmax(self.jump(pred), dim=1)
        boost = F.log_softmax(self.boost(pred), dim=1)
        drift = F.log_softmax(self.drift(pred), dim=1)
        
        return throttle, steer, pitch, yaw, roll, jump, boost, drift


class ConvNet(nn.Module):
    def __init__(self,obs_size=21, obs_width=2, conv_number = 6, hidden_size=256):
        super().__init__()
        self.input_size = obs_size*conv_number
        self.obs_size = obs_size
        self.obs_width = obs_width

        self.conv = nn.Conv2d(1, conv_number, (obs_width*2, 1), 1)

        self.linearNN = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU()
            )
        
        self.throttle = nn.Linear(hidden_size, 3)

    def forward(self,seq):
        x = F.relu(self.conv(seq.view(-1, 1, self.obs_width*2, self.obs_size)))

        pred = self.linearNN(x.view(-1, self.input_size))

        out = F.log_softmax(self.throttle(pred), dim=1)
        
        return out

class FullyConnected2(nn.Module):
    def __init__(self, obs_size=21, obs_width=2, hidden_size=256):
        super().__init__()
        self.input_size = obs_size*obs_width*2

        self.linearNN = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
            )
        
        self.throttle = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.steer = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.pitch = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.yaw = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.roll = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.jump = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.boost = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.drift = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )

    def forward(self,seq):
        pred = self.linearNN(seq.view(-1, self.input_size))

        throttle = F.log_softmax(self.throttle(pred), dim=1)
        steer = F.log_softmax(self.steer(pred), dim=1)
        pitch = F.log_softmax(self.pitch(pred), dim=1)
        yaw = F.log_softmax(self.yaw(pred), dim=1)
        roll = F.log_softmax(self.roll(pred), dim=1)
        jump = F.log_softmax(self.jump(pred), dim=1)
        boost = F.log_softmax(self.boost(pred), dim=1)
        drift = F.log_softmax(self.drift(pred), dim=1)
        
        return throttle, steer, pitch, yaw, roll, jump, boost, drift

class FullyConnected3(nn.Module):
    def __init__(self, obs_size=21, obs_width=2, hidden_size=256):
        super().__init__()
        self.input_size = obs_size*obs_width*2
        
        self.throttle = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.steer = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.pitch = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.yaw = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.roll = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 3)
        )
        self.jump = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.boost = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )
        self.drift = nn.Sequential(
              nn.Linear(self.input_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 2)
        )

    def forward(self,seq):
        pred = seq.view(-1, self.input_size)

        throttle = F.log_softmax(self.throttle(pred), dim=1)
        steer = F.log_softmax(self.steer(pred), dim=1)
        pitch = F.log_softmax(self.pitch(pred), dim=1)
        yaw = F.log_softmax(self.yaw(pred), dim=1)
        roll = F.log_softmax(self.roll(pred), dim=1)
        jump = F.log_softmax(self.jump(pred), dim=1)
        boost = F.log_softmax(self.boost(pred), dim=1)
        drift = F.log_softmax(self.drift(pred), dim=1)
        
        return throttle, steer, pitch, yaw, roll, jump, boost, drift