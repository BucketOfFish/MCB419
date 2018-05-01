from abc import abstractmethod
import numpy as np
import random
from torch import FloatTensor
from torch.autograd import Variable
from collections import deque
import time

############
# SETTINGS #
############

gamma = 0.9
temperature = 1
max_memory = 1000
memory_batch_size = 50
salience_decay = 0.9

############
# RL Agent #
############

class RLAgent:

    def __init__(self):
        self.init_model()
        self.salience_threshold = 0 # how surprising something has to be
    @abstractmethod
    def init_model(self):
        pass

    # quality
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
        loss = self.quality_lossFunction(predicted_quality, quality)
        loss.backward()
        self.quality_optimizer.step()
        return loss.data[0]

    # salience
    def eval_salience(self, state, action):
        return self.salience_function(Variable(FloatTensor(state)), Variable(FloatTensor([action])))
    def S(self, state, action):
        return self.eval_salience(state, action).data[0]
    def train_salience(self, state, action, salience):
        self.salience_function.train()
        self.salience_optimizer.zero_grad()
        predicted_salience = self.eval_salience(state, action)
        salience = Variable(FloatTensor([salience]))
        salience.requires_grad = False
        loss = self.salience_lossFunction(predicted_salience, salience)
        loss.backward()
        self.salience_optimizer.step()
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

    def run(self, use_salience_net, train=False, n_games=500):
        memory = deque(maxlen=max_memory)
        reward_history = []
        for i in range(n_games):
            state = self.env.reset()
            total_reward = 0
            while True:
                # animate
                if self.animate:
                    self.env.render()
                # take a step
                (action, quality) = self.policy(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    self.env.reset() # has to be here because else the program complains during threading
                # update Q function
                if train:
                    # add to memory
                    if use_salience_net:
                        # update salience net - if deltaQ for this (state, action) was large, mark it as important in the net
                        old_Q = quality
                        if done:
                            new_Q = reward
                        else:
                            new_Q = (reward + gamma * self.policy(new_state)[1]) # SARSA
                        delta_Q = new_Q - old_Q
                        if delta_Q > self.salience_threshold:
                            self.salience_threshold = delta_Q
                        self.salience_threshold *= salience_decay
                        self.train_salience(state, action, delta_Q > self.salience_threshold)
                        # # print whether or not a state was important
                        # if delta_Q > self.salience_threshold:
                            # # print("Important state-action", state, action)
                            # print("Important state-action")
                        # else:
                            # # print("Unimportant state-action", state, action)
                            # print("Unimportant state-action")
                        # # time.sleep(0.5)
                        # see if net says this (state, action) is worth remembering
                        if self.S(state, action) > 0.5 or len(memory) < max_memory:
                            memory.appendleft((state, action, reward, new_state, done))
                        pass
                    else:
                        memory.appendleft((state, action, reward, new_state, done))
                    # update Q function
                    if len(memory) >= memory_batch_size:
                        minibatch = random.sample(memory, memory_batch_size)
                        for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                            target = m_reward
                            if not m_done:
                                target = (m_reward + gamma * self.policy(m_next_state)[1]) # SARSA
                            self.train_quality(m_state, m_action, target)
                # total reward
                total_reward += reward
                state = new_state
                if (done): break
            # if i%10 == 0:
                # print("Iteration", i, "Reward", total_reward)
            reward_history.append(total_reward)
        return np.mean(reward_history), np.std(reward_history)

    def learn(self, use_salience_net, n_history_points):
        performance_history = []
        performance_error_history = []
        for i in range(n_history_points):
            self.run(use_salience_net, train=True, n_games=5)
            rewards = self.run(use_salience_net, train=False, n_games=50)
            # print ("After", (i+1)*5, "games - Reward", rewards[0], "+/-", rewards[1])
            performance_history.append(rewards[0])
            performance_error_history.append(rewards[1])
        return(performance_history, performance_error_history)
