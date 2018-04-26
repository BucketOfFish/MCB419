import gym
from RL_agent import RLAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

####################
# NEURAL NET FOR Q #
####################

env = gym.make('MountainCar-v0')
n_input_neurons = 3

# env = gym.make('CartPole-v0')
# n_input_neurons = 5

n_hidden_neurons = 10
learning_rate = 0.001
weight_decay = 0.01

class Quality_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(n_input_neurons, n_hidden_neurons)
        self.hidden = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.output = nn.Linear(n_hidden_neurons, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

############
# TRAINING #
############

class Agent(RLAgent):
    def init_model(self):
        self.env = env
        self.quality_function = Quality_Net()
        self.quality_optimizer = optim.Adam(self.quality_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lossFunction = nn.MSELoss()
    
Learner = Agent()
Learner.learn()
