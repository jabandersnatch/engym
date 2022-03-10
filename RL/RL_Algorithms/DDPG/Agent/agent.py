import numpy as np
import tensorflow as tf
from ..Network.networks import ActorNetwork, CriticNetwork
from ..Noise.ou_action_noise import OUActionNoise
from ..Buffer.buffer import Buffer

class Agent(object):

    def __init__(self, alpha, beta, up_bound, low_bound, input_dims, tau, env, gamma = 0.99, 
        n_actions=2, buffer_capacity = 1000000, layer1_size = 400, layer2_size = 300,
        batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(buffer_capacity = buffer_capacity, n_states = input_dims, n_actions = n_actions)
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()
        self.actor = ActorNetwork(alpha, n_actions, 'Actor', input_dims, self.sess,
                            layer1_size, layer2_size, env.action_space.high)
        self.critic = CriticNetwork(beta, n_actions, 'Critic', input_dims,self.sess,
                                layer1_size, layer2_size, action_bound=up_bound)

        self.target_actor = ActorNetwork(alpha, n_actions, 'TargetActor',
                                    input_dims, self.sess, layer1_size,
                                    layer2_size, env.action_space.high)
        self.target_critic = CriticNetwork(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_critic = \
        [self.target_critic.params[i].assign(
                        tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
            for i in range(len(self.target_critic.params))]

        self.update_actor = \
        [self.target_actor.params[i].assign(
                        tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
            for i in range(len(self.target_actor.params))]

        self.upper_bound = up_bound
        
        self.lower_bound = low_bound
        
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        obs_tuple = (state, action, reward, new_state, done)
        self.memory.record(obs_tuple)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state) # returns list of list
        noise = self.noise()
        mu_prime = mu + noise
        
        legal_actions = np.clip(mu_prime, self.lower_bound, self.upper_bound)
        

        return [np.squeeze(legal_actions)]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer()

        critic_value_ = self.target_critic.predict(new_state,
                                           self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])

        self.update_network_parameters()
    