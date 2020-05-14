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
num_episodes = 30
num_testepis = 1
num_steps = 50

size_batch = 100
size_memory = 500

lr = 0.8
gamma = 0.7

max_eps = 1.0
min_eps = 0.01

data_filename = "DataPERDeepQLearning.csv"

#
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
        while index != 0:
            index = (index - 1)//2
            self.tree[index] += delta
    def get_leaf (self, val):
        parent = 0
        while True:
            left  = 2*parent + 1
            right = 2*parent + 2
            if left >= len (self.tree):
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
        self.db = 0.001
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
        priority_segment = self.tree.total_priority/num
        max_weight = (num*self.tree.min_priority/self.tree.total_priority)**(-self.b)
        for i in range (num):
            a, b = priority_segment*i, priority_segment*(i + 1)
            value = np.random.uniform (a, b)
            index, priority, data = self.tree.get_leaf (value)
            sampling_probabilities = priority / self.tree.total_priority
            b_weight[i] = (num*sampling_probabilities)**(-self.b) / max_weight
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

memory = Memory (size_memory)

model = tf.keras.models.Sequential ()
l = tf.keras.layers
model.add (l.Dense (2, input_dim = 5, activation = 'relu'))
model.compile (optimizer = 'adam', loss = 'mean_squared_error', metrics = [])

# Prep
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
        memory.b = (episode*num_steps + step)/(num_episodes*num_steps)
        # choose action
        if random.random () > eps:
            q_vals = model.predict (np.array ([state0]))
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
        curr_q = model.predict (b_state0)
        next_q = model.predict (b_state1)
        b_target = curr_q
        errors = []
        for i in range (size_batch):
            TD = b_reward[i] + gamma*max (next_q[i]) - b_target[i,b_action[i]]
            errors.append (abs (TD))
            b_target[i,b_action[i]] += lr*TD
        model.fit (b_state0, b_target, epochs = 10, batch_size = size_batch, sample_weight = b_weight, verbose = 0)
        memory.batch_update (b_index, np.array (errors))
    print ("episode {0:3d}, score {1:3d}, dings {2:2d}, eps {3:6f}"
            .format (episode, score, env.dings, eps))
    scores.append (score)

if data_filename:
    with open (data_filename, 'w') as f:
        for i in range (num_episodes):
            f.write ("{0:d},{1:d}\n".format (i, scores[i]))

plt.plot (scores)
plt.show ()

# Play
for episode in range (num_testepis):
    state0 = state_to_array (env.reset ())
    done = False
    score = 0

    for step in range (num_steps):
        q_vals = model.predict (np.array ([state0]))
        action = np.argmax (q_vals)
        reward = env.step (action)
        score += reward
        state = state_to_array (env.state)

    print ("score", score)

