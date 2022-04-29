"""
Buffer system for the RL
"""

import numpy as np

import random
from collections import deque


class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size, buffer_unbalance_gap = 0.5):
        """
        Initialize the attributes.
        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [buffer_unbalance_gap/2]

        self.buffer_unbalance_gap = buffer_unbalance_gap

    def append(self, state, action, r, sn, d):
        """
        Append to the Buffer
        Args:
            state: the state
            action: the action
            r: the reward
            sn: the next state
            d: done (whether one loop is done or not)
        """
        self.buffer.append([state, action, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])

    def get_batch(self, unbalance_p=True):
        """
        Get the batch randomly from the buffer
        Args:
            unbalance_p: If true, unbalance probability of taking the batch from buffer with
            recent event being more prioritized
        Returns:
            the resulting batch
        """
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)
                                  * self.buffer_unbalance_gap + self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer
