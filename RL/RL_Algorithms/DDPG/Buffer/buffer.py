import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self, buffer_capacity=50000, batch_size=64, n_states=None, n_actions=None):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        if n_states is None:
            # Throws error if n_states is not specified
            raise ValueError("n_states must be specified")
        self.n_states = n_states
        if n_actions is None:
            # Throws error if n_actions is not specified
            raise ValueError("n_actions must be specified")
        self.n_actions = n_actions

        self.state_buffer = np.zeros((self.buffer_capacity, self.n_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.n_actions))
        self.reward_buffer = np.zeros(self.buffer_capacity)
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.n_states))
        self.terminal_buffer = np.zeros(self.buffer_capacity, dtype=np.float64)
        self.mem_cntr = 0
    def record(self, obs_tuple):
        
        index = self.mem_cntr % self.buffer_capacity

        state, action, reward, obs_tp1, done = obs_tuple

        self.state_buffer[index] = state
        self.action_buffer[index] = action[0]
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = obs_tp1
        self.terminal_buffer[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.buffer_capacity)

        batch = np.random.choice(max_mem, self.batch_size)

        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        next_states = self.next_state_buffer[batch]
        terminal = self.terminal_buffer[batch]

        return states, actions, rewards, next_states, terminal
        