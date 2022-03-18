"""
    In this file, we will make a custom environment for training our DQN agent. This environment consist of stacking bars on top of each other.
    The agent will be able to decide the dimensions of the stacked bars. The bars must hold a given force that has being given to them as well it's own weight.
"""

import numpy as np
from gym import Env
from gym.spaces import Box

from RL.Render.render import Bar_Builder


class StackedBarsEnv(Env):
    """
    This is a custom environment for stacking bars on top of each other.
    The agent will be able to decide the dimensions of the stacked bars.
    """
    def __init__(self, goal_dist=1, down_force=113100, E=200e9, s_y=250e6, rho=7800, g=9.81, r_min=10e-3, r_max=100e-3, min_h=0.01, max_h=0.1, dist_lim= 0.05):
        """

        ## Parameters
        * **goal_dist** (float): The distance between the two bars.
        * **down_force** (float): The force that is being given to the bars.
        * **E** (float): The Young's modulus of the bars.
        * **s_y** (float): The yield stress of the bars.
        * **rho** (float): The density of the bars.
        * **g** (float): The gravity.
        """

        super(StackedBarsEnv, self).__init__()

        # Set the action space

        self.action_space = Box(low=np.array([r_min, min_h*goal_dist]), high=np.array([r_max, max_h*goal_dist]), dtype=np.float64)

        # Set the observation space

        self.observation_space = Box(low=np.array([0, 0, 0, 0]), high=np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float64)

        # Set the goal distance 
        self.goal_dist = goal_dist
        # Set the down force
        self.down_force = down_force
        # Set the Young's modulus
        self.E = E
        # Set the yield stress
        self.s_y = s_y
        # Set the density
        self.rho = rho
        # Set the gravity
        self.g = g
        # Set the state
        self.total_weight = 0
        # Set min radius
        self.r_min = r_min
        # Set max radius
        self.r_max = r_max
        # Set min height
        self.min_h = min_h
        # Set max height
        self.max_h = max_h
        # Set the state
        self.state = None
        # self set the distance limit
        self.dist_lim = dist_lim
        # Set the min mass
        self.min_mass = r_min ** 2 * np.pi * min_h * goal_dist * rho

        # create a empty numpy array 1x3 to store the position, mass, total deformation and deformation of the bar
        self.list_render = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

        self.temp_position = 0

    def step(self, action):
        """
        This function is called every time the agent takes an action.
        The action the creation of a bar.
        """
        error_msg = f"{action!r} was not a valid action. Actions must be a numpy array of shape (2,)"
        assert action.shape == (2,), error_msg        
        r_bar = action[0]
        h_bar = action[1]
        position, t_mass, total_def, d_def = self.state

        current_mass = r_bar ** 2 * np.pi * h_bar * self.rho
        # Calculate mass of the bar
        t_mass = t_mass + current_mass
        # calculate the weight of the bar
        t_w_bar = t_mass * self.g

        t_force = self.down_force + self.total_weight

        # Calculate the normal stress of the section

        n_stress = (t_force) / (r_bar**2 * np.pi)

        # Calculate the deformation of the section

        d_def = n_stress / self.E * h_bar
        
        # Calculate the total deformation of the bar

        total_def += d_def


        # Calculate the current position of the bar

        position = self.goal_dist - self.temp_position

        self.temp_position += h_bar
        
        # Calculate the total weight of the bar
        self.total_weight = t_w_bar

        # Save the current state in the array

        self.state = np.array([position, t_mass, total_def, d_def], dtype=np.float64)

        # Calculate the reward

        # create a atan function that goes from 0 to 1 with the distance 

        # Check if the bar has reached the goal
        done = (position <= 0)


        reward = 1/(t_mass)*(self.goal_dist-position)*10

 
        if n_stress > self.s_y:
            reward -= t_mass*position**2*100

        state_action_holder= np.concatenate((self.state, action), axis=None)
        self.list_render = np.vstack((self.list_render, state_action_holder)) 
            
            
        # Store the state

        return self.state, reward, done, {}
    
    def reset(self):
        """
        This function is called every time the environment is reset.
        """
        self.state = np.array([0, 0, 0, 0], dtype=np.float64)
        self.total_weight = 0
        self.temp_position = 0
        self.list_render = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        return self.state
    
    def render(self, mode='human', n_episode=None, n_run=None):
        """
        This function is called every time the environment is rendered.
        """
        if mode == 'human':
            
            bar_builder = Bar_Builder(max_dist_y=self.goal_dist, n_episode=n_episode, n_run=n_run)
            bar_builder.run_render(self.list_render)