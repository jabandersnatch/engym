import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Environment.test.physics_test import Test_Environment
from Train.train import TrainDDPGAgent
from Train.auto_train import ddpg_auto_train
import argparse

parser = argparse.ArgumentParser(description='The main file can recieve a number of arguments')

parser.add_argument('-tenv','--test_env', help='test the physics of the enviroments')
parser.add_argument('-a_ddpg','--auto_train_ddpg', help='This is a test for the environment.', action='store_true')
args = parser.parse_args()

def main(action):
    '''
    The main function for the repository
    '''
    if action.test_env:
        Test_Environment()
    if action.auto_train_ddpg:
        ddpg_auto_train()
    else:
        TrainDDPGAgent()

if __name__ == "__main__":
    main(args)

