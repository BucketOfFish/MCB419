import torch
import numpy as np
from collections import OrderedDict
from RL_agent import RLAgent
from threading import Thread
import sys, os

###############
# ENVIRONMENT #
###############

from cart_pole import *

################
# SET UP AGENT #
################

learning_rate = 0.001
weight_decay = 0.01
animate = True
use_salience_net = False

class Agent(RLAgent):
    def init_model(self):
        self.env = env
        self.animate = animate
        # quality
        self.quality_function = Quality_Net()
        self.quality_optimizer = optim.Adam(self.quality_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.quality_lossFunction = nn.MSELoss()
        # salience
        self.salience_function = Salience_Net()

agent = Agent()
agent.learn(use_salience_net)
