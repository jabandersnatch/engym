import os
import numpy as np
import tensorboard
import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout


# Set up gpu options


class CriticNetwork(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 batch_size=64, chkpt_dir='tmp/ddpg', action_bound=1):
        tf.compat.v1.disable_eager_execution()
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, self.input_dims],
                                        name='inputs')

            self.actions = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.compat.v1.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')

            f1 = 1. /np.sqrt(self.fc1_dims)
            # dense1 = tf.compat.v1.layers.dense(self.input, units=self.fc1_dims,
            #                          kernel_initializer=tf.random_uniform_initializer(-f1, f1),
            #                          bias_initializer=tf.random_uniform_initializer(-f1, f1))
            dense1 = Dense(self.fc1_dims, activation='relu', 
                            kernel_initializer=tf.random_uniform_initializer(-f1, f1), 
                            bias_initializer=tf.random_uniform_initializer(-f1, f1))(self.input)
            # batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            batch1 = BatchNormalization()(dense1)
            layer1_activation = tf.nn.relu(batch1)
            
            f2 = 1. / np.sqrt(self.fc2_dims)
            
            # dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
            #                          kernel_initializer=tf.random_uniform_initializer(-f2, f2),
            #                          bias_initializer=tf.random_uniform_initializer(-f2, f2))
            dense2 = Dense(self.fc2_dims, activation='relu',
                            kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                            bias_initializer=tf.random_uniform_initializer(-f2, f2))(layer1_activation)

            #batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            batch2 = BatchNormalization()(dense2)

            # action_in = tf.compat.v1.layers.dense(self.actions, units=self.fc2_dims,
            #                             activation='relu')
            action_in = Dense(self.fc2_dims, activation='relu')(self.actions)


            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = 0.004
            # self.q = tf.compat.v1.layers.dense(state_actions, units=1,
            #                    kernel_initializer=tf.random_uniform_initializer(-f3, f3),
            #                    bias_initializer=tf.random_uniform_initializer(-f3, f3))
            self.q = Dense(1, activation='linear',
                            kernel_initializer=tf.random_uniform_initializer(-f3, f3),
                            bias_initializer=tf.random_uniform_initializer(-f3, f3))(state_actions)

            self.loss = tf.compat.v1.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.actions: actions,
                                 self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class ActorNetwork(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        tf.compat.v1.disable_eager_execution()
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.compat.v1.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, self.input_dims],
                                        name='inputs')

            self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='gradients')

            f1 = 1. / np.sqrt(self.fc1_dims)
            # dense1 = tf.compat.v1.layers.dense(self.input, units=self.fc1_dims,
            #                          kernel_initializer=tf.random_uniform_initializer(-f1, f1),
            #                          bias_initializer=tf.random_uniform_initializer(-f1, f1))
            dense1 = Dense(self.fc1_dims, activation='relu',
                            kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                            bias_initializer=tf.random_uniform_initializer(-f1, f1))(self.input)
            
            # batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            
            batch1= BatchNormalization()(dense1)

            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            
            # dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
            #                         kernel_initializer=tf.random_uniform_initializer(-f2, f2),
            #                         bias_initializer=tf.random_uniform_initializer(-f2, f2))
            dense2 = Dense(self.fc2_dims, activation='relu',
                            kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                            bias_initializer=tf.random_uniform_initializer(-f2, f2))(layer1_activation)

            #batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            
            batch2= BatchNormalization()(dense2)

            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.004
            # mu = tf.compat.v1.layers.dense(layer2_activation, units=self.n_actions,
            #                 activation='tanh',
            #                 kernel_initializer= tf.random_uniform_initializer(-f3, f3),
            #                 bias_initializer=tf.random_uniform_initializer(-f3, f3))
            mu = Dense(self.n_actions, activation='tanh',
                            kernel_initializer= tf.random_uniform_initializer(-f3, f3),
                            bias_initializer=tf.random_uniform_initializer(-f3, f3))(layer2_activation)
                            
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.action_gradient: gradients})

