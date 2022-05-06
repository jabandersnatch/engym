import logging
import random
import tensorflow as tf
import numpy as np
import gym

from tqdm import trange
from RL_Algorithms.DDPG.Agent.agent import Agent
from Environment.custom_environment import StackedBarsEnv
from .utils import plotLearning

CHKPT_PATH = './checkpoints/DDPG'
class TrainDDPGAgent():
    def __init__(self, name = 'def_run_h', n_episodes=100, show_every=100,  lr_critic = 0.002, 
                 lr_actor = 0.001, rho=0.005, batch_size = 64, buffer_size=50000, 
                 fc1_dims = 256, fc2_dims = 128, fc3_dims = 64, gamma = 0.99, warm_up=1, 
                 eps_greedy = 1, use_noise = True, learn = True, unbalance_p = 0.8, 
                 save_weights = True, load_weights = True, buffer_unbalance_gap = 0.5):
        
        self.name = name
        self.n_episodes = n_episodes
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.warm_up = warm_up
        self.eps_greedy = eps_greedy
        self.use_noise = use_noise
        self.unbalance_p = unbalance_p
        self.save_weights = save_weights
        self.load_weights = load_weights
        self.learn = learn
        self.rho = rho
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        
        self.env = gym.make('Pendulum-v1')

        self.num_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.show_every = show_every
        self.agent = Agent(name = 'Agent'+ name, lr_critic=lr_critic, lr_actor=lr_actor, 
                           gamma = gamma, action_high=self.env.action_space.high,
                           action_low=self.env.action_space.low, num_states=self.num_states, 
                           rho=rho, batch_size=batch_size, buffer_size=buffer_size, 
                           buffer_unbalance_gap = buffer_unbalance_gap,
                           fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims,
                           num_actions=self.n_actions)
    
        self.score_history = []
        self.acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
        self.actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
        self.Q_loss = tf.keras.metrics.Mean('Q_loss',dtype=tf.float32)
        self.A_loss = tf.keras.metrics.Mean('A_loss',dtype=tf.float32)

        self.ep_reward_list = []
        self.avg_reward_list = []

        self.train()

    def train(self):
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

        weight_path = CHKPT_PATH +'_'+ self.name
        logging.info('Loading weights from %s*, make sure the folder exists', weight_path)
        if self.load_weights:
            self.agent.load_weights(weight_path)

        with trange(self.n_episodes) as t:
            for ep in t:
                prev_state = self.env.reset()
                self.acc_reward.reset_states()
                self.actions_squared.reset_states()
                self.Q_loss.reset_state()
                self.A_loss.reset_states()
                self.agent.noise.reset()

                while True:
                    self.env.render()
                    cur_act = self.agent.act(tf.expand_dims(prev_state, 0), _notrandom=(ep >= self.warm_up) and
                                        (random.random() < self.eps_greedy+(1-self.eps_greedy)*ep/self.n_episodes),
                                        noise = self.use_noise)
                    state, reward, done, _ = self.env.step(cur_act)
                    self.agent.remember(prev_state, reward, state, int(done))

                    if self.learn:
                        c, a = self.agent.learn(self.agent.buffer.get_batch(unbalance_p=self.unbalance_p))
                        self.Q_loss(c)
                        self.A_loss(a)

                    self.acc_reward(reward)
                    self.actions_squared(np.square(cur_act/self.env.action_space.high)) 
                    prev_state = state

                    if done:
                        break

                self.ep_reward_list.append(self.acc_reward.result().numpy())

                self.avg_reward = np.mean(self.ep_reward_list[-40:])

                self.avg_reward_list.append(self.avg_reward)

                t.set_postfix(r=self.avg_reward)
                
                if ep % self.show_every == 0:
                    #self.env.render(n_run = self.name, n_episode = ep)
                    if self.save_weights:
                        self.agent.save_weights(weight_path)

        plotLearning(scores=self.ep_reward_list, filename='./out/graphs/'+self.name)
        self.env.close()
        self.agent.save_weights(weight_path)

        logging.info('Training done...')
