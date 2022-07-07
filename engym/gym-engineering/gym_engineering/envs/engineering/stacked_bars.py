"""
    In this file, we will make a custom environment for training our RL agent. This environment consist of stacking bars on top of each other.
    The agent will be able to decide the dimensions of the stacked bars. The bars must hold a given force that has being given to them as well it's own weight.
"""

from turtle import position
import numpy as np
import gym
from gym import logger
from pygame import gfxdraw
import pygame
from gym.spaces import Box


class StackedBarsEnv(gym.Env):

    metadata = {'render.modes': ['human']}

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
        * **g** (float): The gravity acceleration.
        """

        # Set the action space

        r_min = np.sqrt(down_force/(np.pi*s_y))

        r_max = np.sqrt(down_force/(np.pi*(s_y-rho*goal_dist*g)))

        self.action_space = Box(low=np.array([r_min]), high=np.array([r_max]), dtype=np.float64)

        self.res = res
        # Calculate the height for the bar segment
        self.h = goal_dist/res
        # Calculate the min mass
        self.min_mass = r_min ** 2 * np.pi * goal_dist * rho
        # Calculate the min deformation
        self.max_def = ((down_force+r_min**2*np.pi*goal_dist*rho*g)/(r_min**2 * np.pi))/E * goal_dist

        # Calculate the max mass
        self.max_mass = r_max ** 2 * np.pi * goal_dist * rho
        # Calculate the max deformation
        self.min_def = ((down_force+r_max**2*np.pi*goal_dist*rho*g)/(r_max**2 * np.pi))/E * goal_dist

        # Set the observation space

        self.observation_space = Box(
            low=np.array([self.min_mass, self.max_def, 0, 0]), 
            high=np.array([self.max_mass, self.min_def, goal_dist, r_max**2*np.pi]), dtype=np.float64)

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
        self.screen = None

        self.clock = None

    def step(self, action: np.ndarray) -> tuple:
        """
        This function is called every time the agent takes an action.
        The action the creation of a bar.
        ##Parameters
            self 
            action = np.array()
        """
        r_bar = action[0]
        t_mass, total_def, position, area = self.state

        area = r_bar**2 * np.pi

        current_mass = area * self.h * self.rho
        # Calculate mass of the bar
        t_mass += current_mass
        # calculate the weight of the bar
        t_w_bar = t_mass * self.g

        t_force = self.down_force + t_w_bar

        # Calculate the normal stress of the section

        n_stress = (t_force) / (area)

        # Calculate the deformation of the section

        d_def = n_stress / self.E * self.h
        # Calculate the total deformation of the bar

        total_def += d_def

        position -= self.h

        # Save the current state in the array

        self.state = np.array([t_mass, total_def, position, area], dtype=np.float64)
        # Check if the bar has reached the goal
        done = bool(
            n_stress > self.s_y 
            or position == 0 
            )

        reward = 1/self.res
        if done == True:
            reward = total_def/self.min_def

        # Store the state

        return self.state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        """
        This function is called every time the environment is reset.
        """
        self.state = np.array([0, 0, self.goal_dist, 0], dtype=np.float64)
        self.total_weight = 0
        self.screen = None
        self.clock = None
        return self.state
    
    def render(self, mode='human'):
        """
        This function is called every time the environment is rendered.
        """
        screen_width = 200
        screen_height = 800

        world_width = self.goal_dist
        scale_y = screen_height / world_width
        scale_x = screen_width / (self.r_max * 2)

        if self.state is None:
            return None

        x = self.state
        r_bar = np.sqrt(x[3] / (np.pi))
        position = x[2]
        total_mass = x[0]
        total_deff = x[1]

            

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width+20, screen_height+20))
            self.font = pygame.font.SysFont("stencil", 20)
            self.font_small = pygame.font.SysFont("stencil", 15)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width+20, screen_height+20))
        self.surf.fill((0, 0, 0))

        # Draw the goal
        gfxdraw.filled_circle(self.screen, int(screen_width/2), 800, 5, (0, 255, 255))

        a1,b1 = int(screen_width/2 - r_bar * scale_x), int(position * scale_y)
        a2,b2 = int(screen_width/2 + r_bar * scale_x), int(position * scale_y)
        a3,b3 = int(screen_width/2 + r_bar * scale_x), int(position * scale_y + self.h * scale_y)
        a4,b4 = int(screen_width/2 - r_bar * scale_x), int(position * scale_y + self.h * scale_y)

        gfxdraw.filled_polygon(self.screen, ([a1,b1],[a2,b2],[a3,b3],[a4,b4]), (255, 255, 255))


        self.surf = pygame.transform.flip(self.surf, False, True)
        
        if position == 0:
            text_t_mass = self.font.render('Total mass: {} [kg]'.format(total_mass), True, (255, 255, 255))
            self.screen.blit(text_t_mass, (3, 5))
            text_t_deff = self.font.render('Total deformation: {} [µm]'.format(total_deff), True, (255, 255, 255))
            self.screen.blit(text_t_deff, (3, 25))
            pygame.time.delay(500)
            pygame.display.update()
        pygame.display.update()
    

        if mode == 'human':
            # Draw the bars
            pygame.event.pump()
            self.clock.tick(15)
            pygame.display.flip()
        
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes = (1,0,2)
            )



    def close(self)->None:
        pass

