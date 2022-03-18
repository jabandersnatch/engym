import numpy as np
from ..DDPG.Agent.agent import Agent
from RL.Environment.custom_environment import StackedBarsEnv
from .utils import plotLearning

class TrainDDPGAgent():
    def __init__(self, name_run = 'def_run', n_episodes=1000, show_every=100,  alpha = 0.00005, beta = 0.0005, tau=0.001, 
                batch_size = 64, layer1_size = 800, layer2_size = 800, gamma = 0.99):
        
        self.name = name_run
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        
        self.env = StackedBarsEnv()

        self.input_dims = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.show_every = show_every
        self.agent = Agent(alpha = self.alpha, name = 'Agent'+ name_run, beta = self.beta, gamma = self.gamma, up_bound=self.env.action_space.high, 
                            low_bound=self.env.action_space.low, input_dims=self.input_dims, tau=self.tau, env=self.env, 
                            batch_size=self.batch_size, layer1_size=self.layer1_size, layer2_size=self.layer2_size, 
                            n_actions=self.n_actions)
    
        self.score_history = []
        self.train()

    def train(self):
        for i in range(self.n_episodes+1):
            done = False
            score = 0
            prev_state = self.env.reset()
            while not done:
                action = self.agent.choose_action(prev_state)
                new_state, reward, done, info = self.env.step(action[0])
                self.agent.remember(prev_state, action, reward, new_state, int(done))
                self.agent.learn()
                score += reward
                prev_state = new_state

            self.score_history.append(score)
            if i % self.show_every == 0:
                print('episode: ', i, 'score: ', score)
                self.env.render(n_episode = i ,n_run = self.name)
        filename = './out/graphs/'+self.name+'DDPG_' + str(self.alpha) + '_' + str(self.beta) + '_' + str(self.tau) + '_' + str(self.batch_size) + '_' + str(self.layer1_size) + '_' + str(self.layer2_size) + '.png'
        
        plotLearning(self.score_history, filename, window=100)