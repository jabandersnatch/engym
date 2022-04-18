from RL.RL_Algorithms.DDPG.Agent.agent import Agent
from RL.Environment.custom_environment import StackedBarsEnv
from .utils import plotLearning

class TrainDDPGAgent():
    def __init__(self, name_run = 'def_run', n_episodes=1000, show_every=100,  lr_critic = 0.00005, lr_actor = 0.0005, rho=0.001, 
                batch_size = 64, buffer_size=1000000, fc1_dims = 600, fc2_dims = 300, fc3_dims = 150, gamma = 0.99):
        
        self.name = name_run
        self.n_episodes = n_episodes
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.rho = rho
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        
        self.env = StackedBarsEnv()

        self.num_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.show_every = show_every
        self.agent = Agent(name = 'Agent'+ name_run, lr_critic=lr_critic, lr_actor=lr_actor, gamma = gamma, action_high=self.env.action_space.high, 
                            action_low=self.env.action_space.low, num_states=self.num_states, rho=rho, 
                            batch_size=batch_size, buffer_size=buffer_size, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims,
                            num_actions=self.n_actions)
    
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
        filename = './out/graphs/'+self.name+'DDPG_' + str(self.lr_critic) + '_' + str(self.lr_actor) + '_' + str(self.tau) + '_' + str(self.batch_size) + '_' + str(self.fc1_dims) + '_' + str(self.fc2_dims) + '.png'
        
        plotLearning(self.score_history, filename, window=100)
