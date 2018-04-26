import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

############
# SETTINGS #
############

env = gym.make('Pong-ram-v0')

n_games = 3000
learning_rate = 0.1
weight_decay = 0.01
gamma = 0.95
temperature = 10

####################
# NEURAL NET FOR Q #
####################

class Quality_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(129, 20)
        self.hidden = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

class Quality():
    def __init__(self):
        self.net = Quality_Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lossFunction = nn.MSELoss()
    def eval(self, state, action):
        return self.net(Variable(torch.FloatTensor(state)), Variable(torch.FloatTensor([action])))
    def train(self, state, action, quality):
        self.net.train()
        self.optimizer.zero_grad()
        predicted_quality = self.eval(state, action)
        #quality = quality.detach()
        quality = Variable(torch.FloatTensor([quality]))
        quality.requires_grad = False
        loss = self.lossFunction(predicted_quality, quality)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]
    
quality_function = Quality()
def Q(state, action):
    return quality_function.eval(state,action).data[0]

##########
# POLICY #
##########

def qualities_from_state(state): # list of (action, quality) for all actions
    qualities = [(action, Q(state, action)) for action in range(env.action_space.n)]
    qualities.sort()
    # print(qualities)
    return qualities

def V(state):
    return qualities_from_state(state)[0][1]

def policy(state): # return softmax (action, quality)
    qualities = qualities_from_state(state)
    probs = [np.exp(quality[1] / temperature) for quality in qualities]
    cumulative_prob = np.cumsum(probs)
    throw = np.random.rand()*cumulative_prob[-1]
    action_choice = np.searchsorted(cumulative_prob, throw)
    return qualities[action_choice]

######################
# TRAIN VIA TD-GAMMA #
######################

history = []
max_history = 50

for i in range(n_games):
    state = env.reset()
    total_reward = 0
    while True:
        # animate
        env.render()
        # take a step and add to history
        (action, quality) = policy(state)
        history = [(state, action, quality)] + history
        if len(history) > max_history:
            history.pop(-1)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        # print(reward)
        # update Q function
        quality_difference = reward + gamma * V(state) - quality
        for t_back, history_point in enumerate(history):
            (h_state, h_action, h_quality) = history_point
            quality_function.train(h_state, h_action, pow(gamma, t_back) * quality_difference + h_quality)
        if (done): break
    if i%10 == 0:
        print("Iteration", i, "Reward", total_reward)
