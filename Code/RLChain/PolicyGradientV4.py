# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
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

model = tf.keras.models.Sequential ()
model.add (tf.keras.layers.Dense (2, activation = 'softmax', input_shape = (5,)))
model.compile (loss = 'categorical_crossentropy',
               optimizer = 'adam')

def get_action (state):
    global model
    distribution = model (np.array ([state]))[0]
    action = np.random.choice (2, p = distribution)
    return action

def do_update (rewards, state0s):
    global model
    discounted = do_normalize (do_discount (rewards))
    return model.train_on_batch (state0s, discounted)

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
    print ("score: {0:3d}".format (sum (rewards)))
    state0s = np.array (state0s)
    actions = np.array (actions)
    rewards = np.array (rewards)
    loss = do_update (rewards, state0s)
