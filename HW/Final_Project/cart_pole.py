import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

env = gym.make('CartPole-v0')

n_games = 10000
learning_rate = 0.01
weight_decay = 0.01
gamma = 0.95
epsilon = 0.01
max_t = 100

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

def policy(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        best_action = 0
        best_quality = quality_function.eval(state, 0).data[0]
        for action in range(env.action_space.n):
            quality = quality_function.eval(state, action)
            if quality.data[0] > best_quality:
                best_quality = quality.data[0]
                best_action = action
        return best_action

def value(state):
    best_action = 0
    best_quality = quality_function.eval(state, 0).data[0]
    for action in range(env.action_space.n):
        quality = quality_function.eval(state, action)
        if quality.data[0] > best_quality:
            best_quality = quality.data[0]
            best_action = action
    return best_quality

for i in range(n_games):
    state = env.reset()
    total_reward = 0
    for t in range(max_t):
        action = policy(state)
        old_state = state
        state, reward, done, _ = env.step(action)
        total_reward += reward
        #quality_function.train(old_state, action, (1-lr) * quality_function.eval(old_state, action) + lr * (reward + value(state)))
        quality_function.train(old_state, action, reward + value(state))
        if (done): break
    if i%100 == 0:
        print("Iteration", i, "Reward", total_reward)
