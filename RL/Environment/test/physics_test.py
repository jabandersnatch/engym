"""
In this file we are going to test the environment. To see if the phisics are working correctly.
We are going to make a set of cases in which the environment should be in a certain state.
"""

from unittest import TestCase
import numpy as np
# import custom environment as env
from ..custom_environment import StackedBarsEnv
class Test_Environment(TestCase):
    """
    This is a test class for the environment.
    """
    def __init__(self):
        super(Test_Environment, self).__init__()
        self.setUp()
        self.check_mass()
        self.check_total_deformation()


    def setUp(self):
        """
        In this function we are going to set up the environment.
        Also we are going to initialize some variables that we will use in the tests.
        """
        self.env = StackedBarsEnv()
        self.env.reset()
        self.test_radius = np.ones(10) * 0.01
        self.test_height = np.ones(10) * 0.1
        self.radius_size = len(self.test_radius)

    def test_reset(self):
        self.env.reset()
        self.assertEqual(self.env.state, np.array([0, 0, 0 ,0]))

    def check_mass(self):
        """
        This function is going to check if the total mass fo the system is correct.
        """
        # Initliaze the total mass

        calc_total_mass = 0
        for i in range (self.radius_size):
            state, reward, done, info =self.env.step(np.array([self.test_radius[i], self.test_height[i]]))
            env_total_mass = state[1]
            calc_total_mass = calc_total_mass + self.test_radius[i] ** 2 * np.pi * self.test_height[i] * self.env.rho
            self.assertEqual(env_total_mass, calc_total_mass, "The total mass is not correct at position {}, ".format(state[0]) + "with the radius {}".format(self.test_radius[i]))
        self.env.reset()

    def check_total_deformation(self):
        """
        This function is going to check if the total deformation of the system is correct.
        """
        # Initialize the total deformation
        calc_total_deformation = 0
        env_t_weight = 0
        for i in range (self.radius_size):
            state, reward, done, info =self.env.step(np.array([self.test_radius[i], self.test_height[i]]))
            env_total_deformation = state[2]
            # Calculate the stress of the section
            total_force = (self.env.down_force + env_t_weight)
            n_stress = total_force / (self.test_radius[i]**2 * np.pi)
            # Calculate the deformation of the section
            d_def = n_stress / self.env.E * self.test_height[i]
            # Calculate the total deformation of the bar
            calc_total_deformation += d_def
            # update the weight of the bar
            env_t_weight = state[1] * self.env.g

            self.assertEqual(np.round(env_total_deformation,8), np.round(calc_total_deformation,8)), "The total deformation is not correct at position {}, ".format(state[0]) + "with the radius {}".format(self.test_radius[i])
        self.env.reset()
    
    def check_position(self):
        """
        This function is going to check if the position of the system is correct.
        """
        # Initialize the position
        calc_position = 0
        for i in range (self.radius_size):
            state, reward, done, info =self.env.step(np.array([self.test_radius[i], self.test_height[i]]))
            env_position = state[0]
            calc_position += self.test_height[i]
            self.assertEqual(env_position, calc_position, "The position is not correct at position {}, ".format(state[0]) + "with the radius {}".format(self.test_radius[i]))
        self.env.reset()
        
