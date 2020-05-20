
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

def discount (episode_rewards):
    global gamma
    discounted = np.zeros_like (episode_rewards, dtype = 'float')
    acc = 0
    for i in reversed (range (len (episode_rewards))):
        acc = acc*gamma + episode_rewards[i]
        discounted[i] = acc
    return discounted

def make_grad_buffer (model):
    grad_buff = model.trainable_variables
    for i, v in enumerate (grad_buff):
        grad_buff[i] = v*0
    return grad_buff

model = tf.keras.models.Sequential ()
model.add (tf.keras.layers.Dense (2, activation = 'softmax', input_shape = (5,)))
model.build ()
optimizer = tf.keras.optimizers.Adam (learning_rate = 0.001)
func_loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits = True)

gamma = 0.9

for episode in range (200):
    env.reset ()
    memory = []
    for step in range (50):
        with tf.GradientTape () as tape:
            state0 = one_hot (env.state, 5)
            logits = model (np.array ([state0]))
            action = np.random.choice (range (2), p = logits.numpy ()[0])
            loss = func_loss (action, logits)
            grad = tape.gradient (loss, model.trainable_variables)
            reward = env.step (action)
            memory.append ((grad, action, reward))
    grads, actions, rewards = zip (*memory)
    print ("score {0:3d}, dings {1:2d}".format (sum (rewards), env.dings))
    rewards = discount (rewards)
    grad_buff = make_grad_buffer (model)
    for i in range (10):
        for j in range (2):
            gr = tf.multiply (grads[i][j], rewards[i])
            grad_buff[j] = tf.add (grad_buff[j], gr)
    optimizer.apply_gradients (zip (grad_buff, model.trainable_variables))
