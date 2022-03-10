"""
    In this file we are going to initialize the project.

"""

from Environment.test.physics_test import Test_Environment
from RL_Algorithms.Train.train import TrainDDPGAgent

import os
import numpy as np
import argparse
import unittest as ut


"""
    In this function we are going to initialize the arguments.

"""

parser = argparse.ArgumentParser(description="This action is going to run the tests for the environment.")
parser.add_argument("-t","--test_env", help="This is a test for the environment.", action="store_true")
args = parser.parse_args()

def main(action):
    """
    This function is going to run the tests for the environment.
    """
    if action.test_env:
        test = Test_Environment()
    else:
        TrainDDPGAgent()

if __name__ == "__main__":
    main(args)

