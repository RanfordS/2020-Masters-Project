# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

import gym

# Environment
env = gym.make ('CartPole-v1')

# Hyperparameters
num_episodes = 50
num_testepis = 5
num_steps = 400

size_batch = 1000
size_memory = 6000

lr = 0.8
gamma = 0.9

max_eps = 1.0
min_eps = 0.1

data_filename = "DataDeepQLearning.csv"

#
memory = deque (maxlen = size_memory)

model = tf.keras.models.Sequential ()
l = tf.keras.layers
model.add (
        l.Dense (16,
            input_dim = env.observation_space.shape[0],
            activation = 'tanh'))
model.add (l.Dense (12, activation = 'tanh'))
model.add (l.Dense (8, activation = 'tanh'))
model.add (l.Dense (env.action_space.n, activation = 'linear'))
model.compile (
        optimizer = 'adam',
        loss = 'mean_squared_error',
        metrics = [])

# Prep
state0 = env.reset ()
for i in range (size_batch):
    action = random.randrange (env.action_space.n)
    state1, reward, isdone, info = env.step (action)
    memory.append ((state0, action, reward, state1, isdone))
    if isdone:
        state0 = env.reset ()
    else:
        state0 = state1

# Training
scores = []
for episode in range (num_episodes):
    eps = max_eps*(min_eps/max_eps)**(episode/num_episodes)

    state0 = env.reset ()
    score = 0
    for step in range (num_steps):
        #env.render ()
        # choose action
        if random.random () > eps:
            q_vals = model.predict (np.array ([state0]))
            action = np.argmax (q_vals)
        else:
            action = random.randrange (env.action_space.n)
        # perform
        state1, reward, isdone, info = env.step (action)
        if isdone:
            reward = 0
        score += reward
        memory.append ((state0, action, reward, state1, isdone))
        state0 = state1
        # create training batch
        batch = random.sample (memory, size_batch)
        batch = list (zip (*batch))
        b_state0 = np.array (batch[0])
        b_action = np.array (batch[1])
        b_reward = np.array (batch[2])
        b_state1 = np.array (batch[3])
        b_isdone = np.array (batch[4])
        curr_q = model.predict (b_state0)
        next_q = model.predict (b_state1)
        b_target = curr_q
        for i in range (size_batch):
            b_target[i,b_action[i]] += lr*(b_reward[i] + gamma*max (next_q[i]) \
                    - b_target[i,b_action[i]])
        model.fit (
                b_state0,
                b_target,
                epochs = 10,
                batch_size = size_batch,
                verbose = 0)
        if isdone:
            break
    print ("episode {0:3d}, score {1:d}, eps {2:6f}"
            .format (episode, int (score), eps))
    scores.append (score)

if data_filename:
    with open (data_filename, 'w') as f:
        for i in range (num_episodes):
            f.write ("{0:d},{1:d}\n".format (i, int (scores[i])))

plt.plot (scores)
plt.show ()

# Play
for episode in range (num_testepis):
    state0 = env.reset ()
    score = 0

    for step in range (num_steps):
        env.render ()
        q_vals = model.predict (np.array ([state0]))
        action = np.argmax (q_vals)
        state0, reward, isdone, info = env.step (action)
        score += reward
        if isdone:
            break

    print ("score", score)

