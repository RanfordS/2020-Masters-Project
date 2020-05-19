# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

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

def state_to_array (index):
    z = np.zeros (5)
    z[index] = 1.0
    return z

# Hyperparameters
num_episodes = 25
num_testepis = 1
num_steps = 50
tau = 10

size_batch = 100
size_memory = 500

lr = 0.8
gamma = 0.7

max_eps = 1.0
min_eps = 0.01

use_double = False
data_filename = "DataAllDeepQLearning.csv"

# Memory
class SumTree:

    def __init__ (self, capacity):
        self.capacity = capacity
        self.index = 0
        self.data = np.zeros (capacity, dtype=object)
        self.tree = np.zeros (2*capacity - 1)
        #       0
        #   ,---'---,
        #   1       2
        # ,-'-,   ,-'-,
        # 3   4   5   6

    def add (self, priority, data):
        tree_index = self.index + self.capacity - 1
        self.data[self.index] = data
        self.update (tree_index, priority)
        self.index = (self.index + 1) % self.capacity

    def update (self, index, priority):
        delta = priority - self.tree[index]
        self.tree[index] = priority
        # cascade up the tree
        while index != 0:
            index = (index - 1)//2
            self.tree[index] += delta

    def get_leaf (self, val):
        parent = 0
        while True:
            left  = 2*parent + 1
            right = 2*parent + 2
            
            if left >= len (self.tree):
                # if the left child node would exceed the bounds of the tree
                # then the current node is a leaf
                break

            if val <= self.tree[left]:
                parent = left
            else:
                val -= self.tree[left]
                parent = right

        data = parent - self.capacity + 1
        return parent, self.tree[parent], self.data[data]
    
    @property
    def max_priority (self):
        return np.max (self.tree[-self.capacity:])
    
    @property
    def min_priority (self):
        return np.min (self.tree[-self.capacity:])
    
    @property
    def total_priority (self):
        return self.tree[0]

class Memory:
    
    def __init__ (self, capacity):
        self.e = 0.01
        self.a = 0.6
        self.b = 0.4
        self.clip = 1.0
        self.tree = SumTree (capacity)

    def append (self, experience):
        max_priority = self.tree.max_priority
        if max_priority == 0:
            max_priority = self.clip
        self.tree.add (max_priority, experience)

    def sample (self, num):
        batch = []
        b_index  = np.empty ((num,),   dtype = np.int32)
        b_weight = np.empty ((num,), dtype = np.float32)

        priority_segment = self.tree.total_priority / num
        max_weight = num*self.tree.min_priority / self.tree.total_priority
        max_weight = max_weight**(-self.b)

        for i in range (num):
            a, b = priority_segment*i, priority_segment*(i + 1)
            value = np.random.uniform (a, b)
            index, priority, data = self.tree.get_leaf (value)
            sampling_probability = priority / self.tree.total_priority
            
            b_weight[i] = (num*sampling_probability)**(-self.b) / max_weight
            b_index[i] = index
            batch.append (data)
        b_weight = np.array (b_weight)

        return b_index, batch, b_weight

    def batch_update (self, index, err):
        err += self.e
        err = np.minimum (err, self.clip)
        err = np.power (err, self.a)
        for i, e in zip (index, err):
            self.tree.update (i, e)

class Model (tf.keras.Model):

    def __init__ (self):
        super (Model, self).__init__ ()
        l = tf.keras.layers
        self.action_layer = l.Dense (2, input_dim = 5, activation = 'relu')
        self.value_layer = l.Dense (1, input_dim = 5, activation = 'relu')
        self.average_layer = l.Lambda (lambda x: x - tf.reduce_mean (x))
        self.q_layer = l.Add ()

        self.compile (optimizer = 'adam',
                      loss = 'mean_squared_error',
                      metrics = [])

    def call (self, x_in):
        x_a = self.action_layer (x_in)
        x_a = self.average_layer (x_a)
        x_v = self.value_layer (x_in)
        return self.q_layer ([x_v, x_a])

policy_a = Model ()
target_a = Model ()
target_a.set_weights (policy_a.get_weights ())
policy_b = Model ()
target_b = Model ()
target_b.set_weights (policy_b.get_weights ())

memory = Memory (size_memory)

# Populate Memory

for i in range (size_memory):
    state0 = state_to_array (env.state)
    action = random.randrange (2)
    reward = env.step (action)
    state1 = state_to_array (env.state)
    memory.append ((state0, action, reward, state1))

# Training

scores = []
for episode in range (num_episodes):
    eps = max_eps*(min_eps/max_eps)**(episode/num_episodes)

    state0 = state_to_array (env.reset ())
    score = 0
    for step in range (num_steps):

        # choose action
        action = None
        if random.random () > eps:
            q_vals = policy_a.predict (np.array ([state0]))
            if use_double:
                q_vals = q_vals + policy_b.predict (np.array ([state0]))
            action = np.argmax (q_vals)
        else:
            action = random.randrange (2)

        # perform
        reward = env.step (action)
        score += reward
        state1 = state_to_array (env.state)
        memory.append ((state0, action, reward, state1))
        state0 = state1

        # create training batch
        b_index, batch, b_weight = memory.sample (size_batch)
        batch = list (zip (*batch))
        b_state0 = np.array (batch[0])
        b_action = np.array (batch[1])
        b_reward = np.array (batch[2])
        b_state1 = np.array (batch[3])

        # predict Q-values
        policy = None
        qf_val = None
        a_star = np.empty (size_batch)

        if use_double:
            target = None
            othert = None
            if random.random () < 0.5:
                policy = policy_a
                target = target_a
                othert = target_b
            else:
                policy = policy_b
                target = target_b
                othert = target_a

            qp_val = target.predict (b_state1)
            qf_val = othert.predict (b_state1)
            a_star = np.argmax (qp_val, axis = 1)
        else:
            policy = policy_a
            target = target_a

            qf_val = target.predict (b_state1)
            a_star = np.argmax (qf_val, axis = 1)

        b_target = policy.predict (b_state0)
        for i in range (size_batch):
            a = b_action[i]
            a_s = a_star[i]
            r = b_reward[i]
            b_target[i,a] += lr*(r + gamma*qf_val[i,a_s] - b_target[i,a])

        # fit values
        policy.fit (b_state0,
                    b_target,
                    epochs = 10,
                    batch_size = size_batch,
                    verbose = 0)
        
        # update fixed q-values
        if (step + 1) % tau == 0:
            target_a.set_weights (policy_a.get_weights ())
            target_b.set_weights (policy_b.get_weights ())

    # display episode results
    print ("episode {0:3d}, score {1:3d}, dings {2:2d}, eps {3:6f}"
            .format (episode, score, env.dings, eps))
    scores.append (score)

# write results to file
if data_filename:
    with open (data_filename, 'w') as f:
        for i in range (num_episodes):
            f.write ("{0:d},{1:d}\n".format (i, scores[i]))

plt.plot (scores)
plt.show ()

# Play
for episode in range (num_testepis):
    state0 = state_to_array (env.reset ())
    score = 0
    for step in range (num_steps):
        q_vals = policy_a.predict (np.array ([state0]))
        if use_double:
            q_vals = q_vals + policy_b.predict (np.array ([state0]))
        action = np.argmax (q_vals)
        reward = env.step (action)
        score += reward
        state = state_to_array (env.state)

    print ("score", score)

