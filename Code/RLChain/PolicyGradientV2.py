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

# Hyperparameters
num_episodes = 200
num_testepis = 1
num_steps = 50

gamma = 0.7

data_filename = "DataPolicyGradient.csv"

# State array
def one_hot (index, size):
    z = np.zeros (size)
    z[index] = 1.0
    return z

def discount (episode_rewards):
    discounted = np.zeros_like (episode_rewards)
    acc = 0
    for i in reversed (range (len (episode_rewards))):
        acc = acc*gamma + episode_rewards[i]
        discounted[i] = acc
    return discounted

class Model:
    
    def __init__ (self):
        l = tf.keras.layers
        self.layer = l.Dense (2, activation = 'softmax')
        self.optimizer = tf.keras.optimizers.Adam (learning_rate = 0.05)
    
    def layer_list (self):
        return [self.layer]
    
    def var_list (self):
        res = []
        for layer in self.layer_list ():
            res.append (layer.variables[0])
            res.append (layer.variables[1])
        return res
    
    def run (self, x):
        return self.layer (x)

    #def get_loss (self, x, y):
    #    return tf.math.square (self.run (x) - y)
    def get_loss (self, probability, action_taken, discounted):
        k = tf.keras.backend
        print (probability)
        print (action_taken)
        probability = k.log (k.sum (probability*action_taken, axis = 1))
        print (probability)
        print (discounted)
        return k.mean (-probability*discounted)

    #def get_grad (self, x, y):
    #    with tf.GradientTape () as tape:
    #        for layer in self.layer_list ():
    #            tape.watch (layer.variables)
    #        loss = self.get_loss (x, y)
    #        print ("loss =")
    #        print (loss)
    #        g = tape.gradient (loss, self.var_list ())
    #        return g
    def get_grad (self, probability, action_taken, discounted):
        with tf.GradientTape () as tape:
            for layer in self.layer_list ():
                tape.watch (layer.variables)
            loss = self.get_loss (probability, action_taken, discounted)
            grad = tape.gradient (loss, self.var_list ())
            return grad
    
    #def fit_step (self, x, y):
    #    g = self.get_grad (x, y)
    #    self.optimizer.apply_gradients (zip (g, self.var_list ()))
    def fit_step (self, probability, action_taken, discounted):
        grad = self.get_grad (probability, action_taken, discounted)
        self.optimizer.apply_gradients (zip (grad, self.var_list ()))

model = Model ()

state0s = []
distrbs = []
actions = []
rewards = []
for i in range (50):
    state0 = one_hot (env.state, 5)
    distrb = model.run (np.array ([state0]))[0]
    action = np.random.choice (range (2), p = distrb)
    reward = env.step (action)
    state0s.append (state0)
    distrbs.append (distrb)
    actions.append (one_hot (action, 2))
    rewards.append (reward)
discounted = discount (rewards)
model.fit_step (np.array (distrbs),
                np.array (actions),
                discounted)
#model.fit_step (np.array ([state0]), np.array ([1,0]))
