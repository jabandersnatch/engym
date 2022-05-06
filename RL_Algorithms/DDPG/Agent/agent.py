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
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                             for (target_weight, ref_weight) in
                             list(zip(model_target.get_weights(), model_ref.get_weights()))])

class Agent:
    '''
    The Agent that contains all the models
    '''

    def __init__(self, name, num_states, num_actions, action_high, action_low, fc1_dims = 600, fc2_dims = 300, 
    fc3_dims = 150, gamma=0.99, rho=0.001, std_dev=0.2, buffer_size = 1000000, 
    batch_size=64, buffer_unbalance_gap=0.5, lr_critic=0.001, lr_actor=0.001):
        # initialize everything
        self.actor_network = ActorNetwork(
                                            name = name, num_states = num_states, 
                                            num_actions = num_actions, action_high=action_high,
                                            fc1_dims=fc1_dims, fc2_dims=fc2_dims
                                            ).get_model()
        self.critic_network = CriticNetwork(
                                            name = 'Critic'+name, num_states = num_states, 
                                            num_actions = num_actions, action_high=action_high,
                                            fc1_dims=fc1_dims, fc2_dims=fc2_dims, 
                                            fc3_dims=fc3_dims
                                            ).get_model()
        self.actor_target = ActorNetwork(
                                            name = 'Actor target'+name, num_states = num_states,
                                            num_actions = num_actions, action_high=action_high, 
                                            fc1_dims=fc1_dims, fc2_dims=fc2_dims
                                            ).get_model()
        self.critic_target = CriticNetwork(
                                            name = 'Critic target'+name, num_states = num_states,
                                            num_actions = num_actions, action_high=action_high, 
                                            fc1_dims=fc1_dims, fc2_dims=fc2_dims, 
                                            fc3_dims=fc3_dims
                                            ).get_model()
    
        # Making the weights equal initially

        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, buffer_unbalance_gap=buffer_unbalance_gap)
        self.gamma = tf.constant(gamma)
        self.rho = rho
        self.action_high = action_high
        self.action_low = action_low
        self.num_states = num_states
        self.num_actions = num_actions
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor, amsgrad=True)
        
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
        def update_weights(s, a, r, sn, d) -> tuple:
            '''
            Function to update weights with optimimzer
            '''
            with tf.GradientTape() as tape:
                y = r + self.gamma * (1-d) * self.critic_target([sn, self.actor_target(sn)])
                critic_loss = tf.math.reduce_mean(tf.math.abs(y-self.critic_network([s, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables)
            )

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables)
            )
            return critic_loss, actor_loss
        
        self.update_weights = update_weights

    def act(self, state, _notrandom=True, noise=True):
        '''
            Run action by the actor network
            Args:
                state: the current state
                _notrandom: if true, the action is not random
                noise: if true, the action is noisy
            Returns:
                the resulting action
        '''
        self.cur_action = (self.actor_network(state)[0].numpy()
                           if _notrandom 
                           else (np.random.uniform(self.action_low, self.action_high, 
                                self.num_actions)) + 
                           (self.noise() if noise else 0)
                           )
        self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)

        return self.cur_action

    def remember(self, prev_state, reward, state, done):
        '''
            Remember the previous state, action, reward, next state, done
        '''
        self.buffer.append(prev_state, self.cur_action, reward, state, done)
    
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

        update_target(model_target=self.actor_target, model_ref=self.actor_network, rho=self.rho)
        update_target(model_target=self.critic_target, model_ref=self.critic_network, rho=self.rho)

        return c_1, a_1

    def save_weights(self, path):
        '''
            Save the weights
        '''
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.actor_network.save_weights(path + '_an.h5')
        self.critic_network.save_weights(path + '_cn.h5')
        self.actor_target.save_weights(path + '_at.h5')
        self.critic_target.save_weights(path + '_ct.h5')

    def load_weights(self, path):
        '''
            Load the weights
        '''
        try:
            self.actor_network.load_weights(path + '_an.h5')
            self.critic_network.load_weights(path + '_cn.h5')
            self.actor_target.load_weights(path + '_at.h5')
            self.critic_target.load_weights(path + '_ct.h5')
        except OSError as err:
            logging.warning('Weights not found, %s', err)
