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

env = gym.make('CartPole-v0')

n_games = 10000
learning_rate = 0.01
weight_decay = 0.01
gamma = 0.95
max_t = 100
temperature = 100

####################
# NEURAL NET FOR Q #
####################

class Quality_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(5, 5)
        self.output = nn.Linear(5, 1)
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = F.relu(self.input(x))
        x = F.softmax(self.output(x))
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
def quality(state, action):
    return quality_function.eval(state,action).data[0]

##########
# POLICY #
##########

def qualities_from_state(state): # list of (action, quality) for all actions
    qualities = [(action, quality(state, action)) for action in range(env.action_space.n)]
    return qualities.sort()

def policy(state): # return softmax (action, quality)
    qualities = qualities_from_state(state)
    probs = [exp(quality[1] / temperature) for quality in qualities]
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

    for t in range(max_t):

        action = policy(state)

        history = [(state, action)] + history
        if len(history) > max_history:
            history.pop(-1)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        history_quality = s
        for history_point in history:
            (history_state, history_action) = history_point
            quality_function.train(old_state, action, reward + value(state))

        if (done): break

    if i%100 == 0:
        print("Iteration", i, "Reward", total_reward)
