"""
    In this file we are going to initialize the project.

"""

from Environment.test.physics_test import Test_Environment
from RL_Algorithms.Train.train import TrainDDPGAgent
from RL_Algorithms.Train.auto_train import ddpg_auto_train


import argparse


"""
    In this function we are going to initialize the arguments.

"""

parser = argparse.ArgumentParser(description="This action is going to run the tests for the environment.")
parser.add_argument("-t","--test_env", help="This is a test for the environment.", action="store_true")
parser.add_argument("-a_ddpg","--auto_train_ddpg", help="This is a test for the environment.", action="store_true")
args = parser.parse_args()

def main(action):
    """
    This function is going to run the tests for the environment.
    """
    if action.test_env:
        test = Test_Environment()
    
    if action.auto_train_ddpg:
        ddpg_auto_train()
    else:
        TrainDDPGAgent()

if __name__ == "__main__":
    main(args)

