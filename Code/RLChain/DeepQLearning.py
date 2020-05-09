# Imports
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
        self.action_space = 2
        self.observation_space = 5
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
num_episodes = 50
num_testepis = 1
num_steps = 50

size_batch = 100
size_memory = 500

lr = 0.8
gamma = 0.7

max_eps = 1.0
min_eps = 0.01

data_filename = "DataDeepQLearning.csv"

#
memory = deque (maxlen = size_memory)

model = tf.keras.models.Sequential ()
l = tf.keras.layers
model.add (l.Dense (2, input_dim = 5, activation = 'relu'))
model.compile (optimizer = 'adam', loss = 'mean_squared_error', metrics = [])

# Prep
for i in range (size_batch):
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
        batch = random.sample (memory, size_batch)
        batch = list (zip (*batch))
        b_state0 = np.array (batch[0])
        b_action = np.array (batch[1])
        b_reward = np.array (batch[2])
        b_state1 = np.array (batch[3])
        curr_q = model.predict (b_state0)
        next_q = model.predict (b_state1)
        b_target = curr_q
        for i in range (size_batch):
            b_target[i,b_action[i]] += lr*(b_reward[i] + gamma*max (next_q[i]) - b_target[i,b_action[i]])
        model.fit (b_state0, b_target, epochs = 10, batch_size = size_batch, verbose = 0)
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

