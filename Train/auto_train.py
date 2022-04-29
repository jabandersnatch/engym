import json

from .train import TrainDDPGAgent

CONFIG_FILE = './config/ddpg_config.json'

def ddpg_auto_train(config_file = CONFIG_FILE):
    with open(config_file) as f:
        config = json.load(f)
    # get all the runs from the config file
    runs = config['run']
    for run in runs:
        # get the parameters for the run
        n_run = run['run']
        n_episodes = run['n_episodes']
        show_every = run['show_every']
        alpha = run['alpha']
        beta = run['beta']
        gamma = run['gamma']
        tau = run['tau']
        batch_size = run['batch_size']
        layer1_size = run['layer1_size']
        layer2_size = run['layer2_size']
        print('Exploration runs: ', n_run)
        print('run: ', n_run, 'alpha: ', alpha, 'beta: ', beta, 'gamma: ', gamma, 'tau: ', tau, 'batch_size: ', batch_size, 'layer1_size: ', layer1_size, 'layer2_size: ', layer2_size)

        TrainDDPGAgent(name_run = n_run, n_episodes = n_episodes, show_every = show_every, alpha = alpha, beta = beta, gamma = gamma, tau = tau, batch_size = batch_size, layer1_size = layer1_size, layer2_size = layer2_size)
