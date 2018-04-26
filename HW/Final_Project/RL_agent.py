from abc import abstractmethod
import numpy as np
import random
from torch import FloatTensor
from torch.autograd import Variable
from collections import deque

############
# SETTINGS #
############

n_games = 3000
gamma = 0.95
temperature = 1
max_memory = 1000
memory_batch_size = 50

############
# RL Agent #
############

class RLAgent:

    def __init__(self):
        self.init_model()

    @abstractmethod
    def init_model(self):
        pass

    def eval_quality(self, state, action):
        return self.quality_function(Variable(FloatTensor(state)), Variable(FloatTensor([action])))

    def Q(self, state, action):
        return self.eval_quality(state,action).data[0]

    def train_quality(self, state, action, quality):
        self.quality_function.train()
        self.quality_optimizer.zero_grad()
        predicted_quality = self.eval_quality(state, action)
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

        memory = deque(maxlen=max_memory)

        for i in range(n_games):
            state = self.env.reset()
            total_reward = 0
            while True:
                # animate
                self.env.render()
                # take a step and add to memory
                (action, quality) = self.policy(state)
                new_state, reward, done, _ = self.env.step(action)
                memory.appendleft((state, action, reward, new_state, done))
                total_reward += reward
                state = new_state
                # update Q function
                if len(memory) >= memory_batch_size:
                    minibatch = random.sample(memory, memory_batch_size)
                    for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                        target = m_reward
                        if not m_done:
                            target = (m_reward + gamma * self.policy(m_next_state)[1]) # SARSA
                        self.train_quality(m_state, m_action, target)
                if (done): break
            if i%10 == 0:
                print("Iteration", i, "Reward", total_reward)
