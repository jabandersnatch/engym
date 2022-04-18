import logging
import os

import numpy as np
import tensorflow as tf
from ..Network.networks import ActorNetwork, CriticNetwork
from ..Noise.ou_action_noise import OUActionNoise
from ..Buffer.buffer import ReplayBuffer

def update_target(model_target, model_ref, rho=0):
    '''
    Update target's weights with the given model reference

    Args:
        model_target: target model to be changed
        model_ref: the reference model
        rho: the ratio of the new and old weights
    '''
    model_target.set_weigths([rho * ref_weight + (1 - rho) * target_weight
                             for (target_weight, ref_weight) in
                             list(zip(model_target.get_weights(), model_ref.get_weights()))])

class Agent:
    '''
    The Agent that contains all the models
    '''

    def __init__(self, name, num_states, num_actions, action_high, fc1_dims = 600, fc2_dims = 300, 
    fc3_dims = 150, action_low = None, gamma=0.99, rho=0.001, std_dev=0.2, buffer_size = 1000000, 
    batch_size=64, lr_critic=0.001, lr_actor=0.001):
        # initialize everything
        self.actor_network = ActorNetwork(name = name, num_states = num_states, num_actions = num_actions, action_high=action_high, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_network = CriticNetwork(name = 'Critic'+name, num_states = num_states, num_actions = num_actions, action_high=action_high, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)
        self.actor_target = ActorNetwork(name = 'Actor target'+name, num_states = num_states, num_actions = num_actions, action_high=action_high, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_target = CriticNetwork(name = 'Critic target'+name, num_states = num_states, num_actions = num_actions, action_high=action_high, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)
    
        # Making the weights equal initially

        self.actor_target.set_weigths(self.actor_network.get_weights())
        self.critic_target.set_weigths(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)
        self.gamma = tf.constant(gamma)
        self.rho = rho
        self.action_high = action_high
        self.action_low = action_low
        self.num_states = num_states
        self.num_actions = num_actions
        self.noise = OUActionNoise(mean=np.zeros(1), std_dev=float(std_dev) * np.ones(1))
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adaman(lr_actor, amsgrad=True)
        
        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, num_states], dtype=tf.float32),
                tf.TensorSpec(shape=[None, num_actions], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, num_states], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            ]
        )
        def update_weights(states, actions, rewards, next_states, dones):
            '''
            Function to update weights with optimimzer
            '''
            with tf.GradientTape() as tape:
                # define target
                y = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, self.actor_target(next_states))
                # define the delta Q
                critic_loss = tf.reduce_mean(tf.square(y - self.critic_network(states, actions)))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network(states, self.actor_network(states)))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_network.trainable_variables))

            return critic_loss, actor_loss
        
        self.update_weights = update_weights

    def act(self, state, _notrandom=True, noise=True):
        '''
            Run action by the actor network
            Args:
                state: the current state
                _notrandom: if true, the action is not random
                noise: if true, the action is noisy
        '''
        self.cur_action = (self.actor_network(state)[0].numpy()
                           if _notrandom 
                           else (np.ramdom.uniform(self.action_low, self.action_high, self.num_actions)) + (self.noise() if noise else 0)
                           )
        self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)

        return self.cur_action

    def remember(self, prev_state, action, reward, state, done):
        '''
            Remember the previous state, action, reward, next state, done
        '''
        self.buffer.add(prev_state, action, reward, state, done)
    
    def learn(self, entry):
        '''
            Learn from the entry
        '''
        s, a, r, sn, d = zip(*entry)

        c_1, a_1 = self.update_weights(
            tf.convert_to_tensor(s, dtype=tf.float32),
            tf.convert_to_tensor(a, dtype=tf.float32),
            tf.convert_to_tensor(r, dtype=tf.float32),
            tf.convert_to_tensor(sn, dtype=tf.float32),
            tf.convert_to_tensor(d, dtype=tf.float32)
        )

        update_target(self.actor_target, self.actor_network, self.rho)
        update_target(self.critic_target, self.critic_network, self.rho)

        return c_1, a_1

    def save_weights(self, path):
        '''
            Save the weights
        '''
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.actor_network.save_weights(path + 'an.h5')
        self.critic_network.save_weights(path + 'cn.h5')
        self.actor_target.save_weights(path + 'at.h5')
        self.critic_target.save_weights(path + 'ct.h5')

    def load_weights(self, path):
        '''
            Load the weights
        '''
        try:
            self.actor_network.load_weights(path + 'an.h5')
            self.critic_network.load_weights(path + 'cn.h5')
            self.actor_target.load_weights(path + 'at.h5')
            self.critic_target.load_weights(path + 'ct.h5')
        except OSError as err:
            logging.warning('Weights not found, %s', err)
