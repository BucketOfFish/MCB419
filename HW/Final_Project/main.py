import torch
import numpy as np
from collections import OrderedDict
from RL_agent import RLAgent

###############
# ENVIRONMENT #
###############

from cart_pole import *

################
# SET UP AGENT #
################

learning_rate = 0.001
weight_decay = 0.01
use_salience_net = True

class Agent(RLAgent):
    def init_model(self):
        self.env = env
        self.animate = False
        # quality
        self.quality_function = Quality_Net()
        self.quality_optimizer = optim.Adam(self.quality_function.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.quality_lossFunction = nn.MSELoss()
        # salience
        self.salience_function = Salience_Net()

#####################
# GENETIC ALGORITHM #
#####################

generations = 100
population_size = 20
keep_top_n = 5
population = []
fitnesses = []
population_temperature = 1

if not use_salience_net:
    agent = Agent()
    agent.learn(use_salience_net = False)
else:
    # set up control agent
    control_agent = Agent()
    torch.save(control_agent.quality_function.state_dict(), 'Saved/initial_Q_net.pt')
    # initialize population with different S parameters but identical Q parameters
    initial_Q_net_parameters = torch.load('Saved/initial_Q_net.pt')
    for individual in range(population_size):
        agent = Agent()
        agent.quality_function.load_state_dict(initial_Q_net_parameters)
        population.append(agent)
    # evolve
    for generation in range(generations):
        print("Entering generation", generation)
        # evaluate population
        fitnesses = []
        for individual in range(population_size):
            print(".", end='', flush=True)
            agent = population[individual]
            fitnesses.append(agent.learn(use_salience_net = True))
        fitnesses, population = zip(*sorted(zip(fitnesses, population)))
        print("Most fit individual has a fitness of", fitnesses[-1])
        # mating
        mate_probs = [np.exp(fitness / population_temperature) for fitness in fitnesses]
        cumulative_prob = np.cumsum(mate_probs)
        new_population = population[:keep_top_n]
        for individual in range(population_size - keep_top_n):
            # find mates
            throw = np.random.rand()*cumulative_prob[-1]
            guy = np.searchsorted(cumulative_prob, throw)
            while True:
                throw = np.random.rand()*cumulative_prob[-1]
                girl = np.searchsorted(cumulative_prob, throw)
                if guy != girl: break
            # mate
            dad_DNA = guy.salience_function.state_dict()
            mom_DNA = girl.salience_function.state_dict()
            child_DNA = OrderedDict()
            for key in dad_DNA:
                dad_gene = dad_DNA[key]
                mom_gene = mom_DNA[key]
                gene_shape = dad_gene.shape
                dad_gene = dad_gene.view(-1)
                mom_gene = mom_gene.view(-1)
                if (gene_shape[0] > 1):
                    while True:
                        cut_point = np.random.randint(1, gene_shape[0])
                        temp_gene = torch.cat([dad_gene[:cut_point], mom_gene[cut_point:]])
                        dad_gene = torch.cat([mom_gene[:cut_point], dad_gene[cut_point:]])
                        mom_gene = temp_gene
                        if np.random.rand() > 0.4:
                            break
                child_gene = dad_gene
                child_gene = child_gene.view(gene_shape)
                child_DNA[key] = child_gene
            child = Agent()
            child.quality_function.load_state_dict(child_DNA)
            # mutate
            population.append(child)
        population = new_population
