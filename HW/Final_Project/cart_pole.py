import gym
from RL_agent import RLAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############
# SETTINGS #
############

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
n_input_neurons = 5

n_hidden_neurons = 10
learning_rate = 0.001
weight_decay = 0.01

####################
# NEURAL NET FOR Q #
####################

class Quality_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(n_input_neurons, n_hidden_neurons)
        self.hidden = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.output = nn.Linear(n_hidden_neurons, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = self.output(x)
        return x

class CartAgent(RLAgent):
    def init_model(self):
        self.env = env
        self.net = Quality_Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lossFunction = nn.MSELoss()
    
CartLearner = CartAgent()
CartLearner.learn()
