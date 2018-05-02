from RL_agent import RLAgent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###############
# ENVIRONMENT #
###############

from cart_pole import *

################
# SET UP AGENT #
################

learning_rate = 0.001
weight_decay = 0.01
animate = False # Matplotlib will crash if it tries to draw a plot after the OpenAI environment has been animating
use_salience_net = True

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
        self.salience_optimizer = optim.Adam(self.salience_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.salience_lossFunction = nn.MSELoss()

n_history_points = 20
n_training_rounds = 25
performance_history = []
performance_error_history = []

for _ in range(n_training_rounds):
    agent = Agent()
    history = agent.learn(use_salience_net, n_history_points)
    print(history)
    performance_history.append(history[0])
    performance_error_history.append(history[1])
    agent.env.close()

performance_history = np.mean(performance_history, axis=0)
performance_error_history = np.sqrt(np.mean(np.square(performance_error_history), axis=0)) # approximation of std using n instead of (n-1)
print(performance_history)
print(performance_error_history)

x = np.arange(1, n_history_points+1, 1)*5
sns.set_style("whitegrid")
plt.errorbar(x, performance_history, performance_error_history, fmt='o-', markersize=8)
plt.xlabel("Training Games")
plt.ylabel("Reward (Evaluated 500 Times)")
plt.title("Average Reward vs. Training Time (Evaluated by Training 25 Times)")
xmax = max(x) + 5
ymax = max([i+j for (i, j) in zip(performance_history, performance_error_history)]) + 5
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.show()
