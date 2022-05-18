"""
    In this file, we will make a custom environment for training our DQN agent. This environment consist of stacking bars on top of each other.
    The agent will be able to decide the dimensions of the stacked bars. The bars must hold a given force that has being given to them as well it's own weight.
"""

import numpy as np
from gym import Env
from gym.spaces import Box

from Render.render import Bar_Builder


class StackedBarsEnv(Env):
    """
    This is a custom environment for stacking bars on top of each other.
    The agent will be able to decide the dimensions of the stacked bars.
    """
    def __init__(self, goal_dist:int =1000, down_force: int=200000, E: float=200e9, s_y: float=250e6, 
                 rho: int=7800, g: float=9.81, res: int = 100) -> None:
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

        r_min = np.sqrt(down_force/(np.pi*s_y))

        r_max = np.sqrt(down_force/(np.pi*(s_y-rho*goal_dist*g)))

        self.action_space = Box(low=np.array([r_min]), high=np.array([r_max]), dtype=np.float64)

        # Calculate the height for the bar segment
        self.h = goal_dist/res
        # Calculate the min mass
        self.min_mass = r_min ** 2 * np.pi * goal_dist * rho
        # Calculate the min deformation
        self.min_def = ((down_force+r_min**2*np.pi*goal_dist*rho*g)/(r_min**2 * np.pi))/E * goal_dist

        # Calculate the max mass
        self.max_mass = r_max ** 2 * np.pi * goal_dist * rho
        # Calculate the max deformation
        self.max_def = ((down_force+r_max**2*np.pi*goal_dist*rho*g)/(r_max**2 * np.pi))/E * goal_dist

        # Set the observation space

        self.observation_space = Box(
            low=np.array([self.min_mass, self.min_def, 0]), 
            high=np.array([self.max_mass, self.max_def, goal_dist]), dtype=np.float64)

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
        # Set min radius
        self.r_min = r_min
        # Set max radius
        self.r_max = r_max
        # create a empty numpy array 1x3 to store the position, mass, total deformation and deformation of the bar
        self.list_render = np.array([0, 0, goal_dist, 0, 0], dtype=np.float64)

        print('The minimun radius is: {}'.format(self.r_min))
        print('The maximun radius is: {}'.format(self.r_max))

    def step(self, action: np.ndarray) -> tuple:
        """
        This function is called every time the agent takes an action.
        The action the creation of a bar.
        ##Parameters
            self 
            action = np.array()
        """
        r_bar = action[0]
        t_mass, total_def, position = self.state

        current_mass = r_bar ** 2 * np.pi * self.h * self.rho
        # Calculate mass of the bar
        t_mass += current_mass
        # calculate the weight of the bar
        t_w_bar = t_mass * self.g

        t_force = self.down_force + t_w_bar

        # Calculate the normal stress of the section

        n_stress = (t_force) / (r_bar ** 2 * np.pi)

        # Calculate the deformation of the section

        d_def = n_stress / self.E * self.h
        
        # Calculate the total deformation of the bar

        total_def += d_def

        position -= self.h

        # Save the current state in the array

        self.state = np.array([t_mass, total_def, position], dtype=np.float64)
        # Check if the bar has reached the goal
        done = (position == 0)

        reward = 1/np.abs(n_stress-self.s_y+0.000001) * position/self.goal_dist


        state_action_holder= np.concatenate((self.state, action, self.h), axis=None)
        self.list_render = np.vstack((self.list_render, state_action_holder)) 
            
            
        # Store the state

        return self.state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        """
        This function is called every time the environment is reset.
        """
        self.state = np.array([0, 0, self.goal_dist], dtype=np.float64)
        self.total_weight = 0
        self.list_render = np.array([0, 0, self.goal_dist, 0, 0], dtype=np.float64)

        return self.state
    
    def render(self, mode='human', n_episode=None, n_run=None):
        """
        This function is called every time the environment is rendered.
        """
        if mode == 'human':
            
            bar_builder = Bar_Builder(max_dist_y=self.goal_dist, max_dist_x=self.r_max, n_episode=n_episode, n_run=n_run)
            bar_builder.run_render(self.list_render)
