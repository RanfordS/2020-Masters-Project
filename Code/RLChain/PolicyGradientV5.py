# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx ('float64')

# Environment
class Chain:
    def __init__ (self):
        self.state = 0
        self.dings = 0
    def reset (self):
        self.state = 0
        self.dings = 0
        return 0
    def step (self, action):
        if action == 1:
            self.state = 0
            return 1
        if self.state == 4:
            self.dings += 1
            return 10
        self.state += 1
        return 0
env = Chain ()

def one_hot (index, size):
    z = np.zeros (size)
    z[index] = 1.0
    return z

gamma = 0.9
def do_discount (episode_rewards):
    global gamma
    discounted = np.zeros_like (episode_rewards, dtype = 'float')
    acc = 0
    for i in reversed (range (len (episode_rewards))):
        acc = acc*gamma + episode_rewards[i]
        discounted[i] = acc
    return discounted

def do_normalize (data):
    return (data - np.mean (data)) / np.std (data)

#model = tf.keras.models.Sequential ()
#model.add (tf.keras.layers.Dense (2, activation = 'softmax', input_shape = (5,)))
#model.compile (loss = 'categorical_crossentropy',
#               optimizer = 'adam')

class PGModel (tf.keras.Model):
    def __init__ (self):
        super (PGModel, self).__init__ ()
        self.layer = tf.keras.layers.Dense (2, activation = 'softmax', input_shape = (5,))
    def call (self, inputs, training = False):
        return self.layer (inputs)

    def r_get_loss (self, p, a, d):
        k = tf.keras.backend
        p = k.log (k.sum (p*a, axis = 1))
        return k.mean (-p*d)

    @tf.function
    def r_train_step (self, inputs, logits):
        with tf.GradientTape () as tape:
            d = self (inputs, training = True)


for episode in range (200):
    state0s = []
    actions = []
    rewards = []
    for step in range (50):
        state0 = one_hot (env.state, 5)
        action = get_action (state0)
        reward = env.step (action)
        state0s.append (state0)
        actions.append (action)
        rewards.append (reward)
