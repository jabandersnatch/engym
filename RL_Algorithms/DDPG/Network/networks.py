import tensorflow as tf

from tensorflow import keras
from keras.initializers.initializers_v2 import GlorotNormal

KERNEL_INITIALIZERS = GlorotNormal()

class ActorNetwork():
    def __init__(self, name, num_actions, num_states, action_high, fc1_dims, fc2_dims) -> None:
        self.name = name 
        self.num_actions = num_actions
        self.num_states = num_states
        self.action_high = action_high
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.build_network()

    def build_network(self) -> None:
        '''
        Get Actor Network with the given parameters.

        Args:
            fc1_dims: number of neurons of the first hidden layer
            fc2_dims: number of neurons of the second hidden layer
            num_actions: number of actions in the nn
            num_states: number of states in the nn
            action_high: the top value from the action

        Returns:
            the Keras Model
        '''
        last_init = tf.random_normal_initializer(stddev=0.0005)

        inputs = keras.layers.Input(shape=(self.num_states,), dtype=tf.float32)
        out = keras.layers.Dense(self.fc1_dims, activation=tf.nn.leaky_relu,
                                 kernel_initializer=KERNEL_INITIALIZERS)(inputs)
        out = keras.layers.Dense(self.fc2_dims, activation=tf.nn.leaky_relu,
                                 kernel_initializer=KERNEL_INITIALIZERS)(out)
        outputs = keras.layers.Dense(self.num_actions, activation='tanh',
                                     kernel_initializer=last_init)(out) * self.action_high
        self.model = tf.keras.Model(inputs, outputs)

    def get_model(self):
        return self.model

class CriticNetwork():
    def __init__(self, name, num_actions, num_states, action_high, fc1_dims, fc2_dims, fc3_dims) -> None:
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.action_high = action_high
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.build_network()

    def build_network(self) -> None:
        '''
        Get the CriticNetwork with the given parameters

        Args:
            num_actions: number of actions in the nn
            num_states: number of states in the nn
            action_high: the top value from the action
        '''
        last_init = tf.random_normal_initializer(stddev=0.0005)

        state_input = keras.layers.Input(shape=(self.num_states), dtype=tf.float32)
        state_out = keras.layers.Dense(self.fc1_dims, activation=tf.nn.leaky_relu,
                                       kernel_initializer=KERNEL_INITIALIZERS)(state_input)
        state_out = keras.layers.BatchNormalization()(state_out)
        state_out = keras.layers.Dense(self.fc2_dims, activation=tf.nn.leaky_relu,
                                       kernel_initializer=KERNEL_INITIALIZERS)(state_out)

        action_input = keras.layers.Input(shape=(self.num_actions), dtype=tf.float32)
        action_out = keras.layers.Dense(self.fc2_dims, activation=tf.nn.leaky_relu,
                                        kernel_initializer=KERNEL_INITIALIZERS)(action_input/self.action_high)
        
        added = keras.layers.Add()([state_out, action_out])

        # make aded layers batchnorm
        added = keras.layers.BatchNormalization()(added)

        outs = keras.layers.Dense(self.fc3_dims, activation=tf.nn.leaky_relu,
                                     kernel_initializer=KERNEL_INITIALIZERS)(added)
        outs = keras.layers.BatchNormalization()(outs)
        outputs = keras.layers.Dense(1, kernel_initializer=last_init)(outs)

        self.model = tf.keras.Model([state_input, action_input], outputs)

    def get_model(self):
        return self.model
