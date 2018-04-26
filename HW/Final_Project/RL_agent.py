from abc import abstractmethod
import numpy as np
from torch import FloatTensor
from torch.autograd import Variable
from collections import deque

############
# SETTINGS #
############

n_games = 3000
alpha = 1
gamma = 0.8
temperature = 10
max_history = 50

############
# RL Agent #
############

class RLAgent:

    def __init__(self):
        self.init_model()

    @abstractmethod
    def init_model(self):
        pass

    def eval(self, state, action):
        return self.quality_function(Variable(FloatTensor(state)), Variable(FloatTensor([action])))

    def Q(self, state, action):
        return self.eval(state,action).data[0]

    def train(self, state, action, quality):
        self.quality_function.train()
        self.quality_optimizer.zero_grad()
        predicted_quality = self.eval(state, action)
        quality = Variable(FloatTensor([quality]))
        quality.requires_grad = False
        loss = self.lossFunction(predicted_quality, quality)
        loss.backward()
        self.quality_optimizer.step()
        return loss.data[0]

    ##########
    # POLICY #
    ##########

    def qualities_from_state(self, state): # list of (action, quality) for all actions
        qualities = [(action, self.Q(state, action)) for action in range(self.env.action_space.n)]
        qualities.sort()
        # print(qualities)
        return qualities

    def V(self, state):
        return self.qualities_from_state(state)[0][1]

    def policy(self, state): # return softmax (action, quality)
        qualities = self.qualities_from_state(state)
        probs = [np.exp(quality[1] / temperature) for quality in qualities]
        cumulative_prob = np.cumsum(probs)
        throw = np.random.rand()*cumulative_prob[-1]
        action_choice = np.searchsorted(cumulative_prob, throw)
        return qualities[action_choice]

    ######################
    # TRAIN VIA TD-GAMMA #
    ######################

    def learn(self):

        history = deque(maxlen=max_history)

        for i in range(n_games):
            state = self.env.reset()
            total_reward = 0
            while True:
                # animate
                self.env.render()
                # take a step and add to history
                (action, quality) = self.policy(state)
                history.appendleft((state, action, quality))
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # update Q function
                if done: reward = -500
                quality_difference = reward + gamma * self.V(state) - quality
                for t_back, history_point in enumerate(history):
                    (h_state, h_action, h_quality) = history_point
                    # print(pow(gamma, t_back) * quality_difference + h_quality)
                    self.train(h_state, h_action, alpha * pow(gamma, t_back+1) * quality_difference + h_quality)
                if (done): break
            if i%10 == 0:
                print("Iteration", i, "Reward", total_reward)
