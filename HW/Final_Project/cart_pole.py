import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###############
# ENVIRONMENT #
###############

env = gym.make('CartPole-v0')

####################
# NEURAL NET FOR Q #
####################

class Quality_Net(nn.Module):
    def __init__(self):
        n_input_neurons = 5
        n_hidden_neurons = 10
        super().__init__()
        self.input = nn.Linear(n_input_neurons, n_hidden_neurons)
        self.hidden = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.output = nn.Linear(n_hidden_neurons, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

################
# SALIENCE NET #
################

class Salience_Net(nn.Module):
    def __init__(self):
        n_input_neurons = 5
        n_hidden_neurons = 10
        super().__init__()
        self.input = nn.Linear(n_input_neurons, n_hidden_neurons)
        self.hidden = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.output = nn.Linear(n_hidden_neurons, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = F.sigmoid(self.output(x))
        return x
