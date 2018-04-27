import gym
from RL_agent import RLAgent

###############
# ENVIRONMENT #
###############

from cart_pole import *

############
# TRAINING #
############

learning_rate = 0.001
weight_decay = 0.01
use_salience_net = False

class Agent(RLAgent):
    def init_model(self):
        self.env = env
        # quality
        self.quality_function = Quality_Net()
        self.quality_optimizer = optim.Adam(self.quality_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.quality_lossFunction = nn.MSELoss()
        # salience
        self.salience_function = Salience_Net()
        self.salience_optimizer = optim.Adam(self.salience_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.salience_lossFunction = nn.MSELoss()
    
Learner = Agent()
Learner.learn(use_salience_net)
