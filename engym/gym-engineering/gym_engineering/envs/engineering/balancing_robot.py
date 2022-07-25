"""
    In this file, we will make a custom environment for training our RL agent. This environment
    is a control environment for a balancing robot. The robot is equipped with a motor that can
    accelerate to a certain speed. The goal is to keep the robot balanced while it is moving.
"""

import numpy as np
import gym
import pygame
from gym import logger
from pygame import gfxdraw
from gym.spaces import Discrete, Box

class BlancingRobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    This is the environment for the balancing robot. The robot is equipped with a motor that can
    accelerate to a certain speed. The goal is to keep the robot balanced while it is moving.
    """
    def __init__(self, L: float = 1, g: float = 9.81) -> None:
        """
        Initialize the environment.
        :param L: The length of the robot.
        :param g: The acceleration of gravity.
        """
        self.L = L
        self.g = g
        self.action_space = Discrete(2)
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf]),
                                                 high=np.array([np.inf, np.inf, np.inf]), dtype=np.float32)
        self.reset()

    def step(self, action: int) -> tuple:
        """
        Perform one step of the environment.
        :param action: The action to perform.
        :return: A tuple containing: (observation, reward, done, info).
        """
        pass

    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        :return: The initial observation.
        """
        pass

    def render(self, mode='human') -> None:
        """
        Render the environment.
        :param mode: The mode to render the environment in.
        :param close: Whether to close the window.
        """
        pass

    def close(self) -> None:
        """
        Close the environment.
        """
        pass
