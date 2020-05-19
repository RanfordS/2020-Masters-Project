# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# Environment
class Chain:
    def __init__ (self):
        self.state = 0
    def reset (self):
        self.state = 0
        return 0
    def step (self, action):
        if action == 1:
            self.state = 0
            return 1
        if self.state == 4:
            return 10
        self.state += 1
        return 0
env = Chain ()

# Hyperparameters
num_episodes = 200
num_testepis = 1
num_steps = 50

lr = 0.8
gamma = 0.7

#eps = 1.0
#max_eps = 1.0
#min_eps = 0.01

data_filename = "DataQLearning.csv"

# State array
def state_to_array (index):
    z = np.zeros (5)
    z[index] = 1.0
    return z

# Discount
def discount (episode_rewards):
    discounted = np.zeros_like (episode_rewards)
    acc = 0
    for i in reversed (range (len (episode_rewards))):
        acc = cumulative*gamma + episode_rewards[i]
        discounted[i] = acc
# Normalize
def normalize (data):
    return (data - np.mean (data)) / np.std (data)

# Network
model = tf.keras.models.Sequential ()
l = tf.keras.layers
model.add (l.Dense (2, input_dim = 5, activation = 'softmax'))
model.compile (optimizer = 'adam', loss = 'mean_squared_error', metrics = [])

# Training
episode_scores = []
episode_rewards = []
for episode in range (num_episodes):
    state0 = state_to_array (env.reset ())

    state0s = []
    actions = []
    rewards = []
    for step in range (num_steps):
        state0 = state_to_array (env.state)
        state0s.append (state0)
        # choose action
        distribution = model.predict (np.array ([state0]))
        action = np.random.choice (range (2), p = distribution)
        actions.append (action)
        # perform
        reward = env.step (action)
        rewards.append (reward)
    episode_scores.append (sum (rewards))
    rewards = normalize (discount (rewards))
    episode_rewards.append (rewards)
    model.fit (b_state0, b_target, epochs = 10, batch_size = size_batch, verbose = 0)

    print ("episode {0:3d}, score {1:3d}, dings {2:2d}"
            .format (episode, episode_scores[i], env.dings))
    #eps = max_eps*(min_eps/max_eps)**(episode/num_episodes)

if data_filename:
    with open (data_filename, 'w') as f:
        for i in range (num_episodes):
            f.write ("{0:d},{1:d}\n".format (i, episode_scores[i]))

plt.plot (scores)
plt.show ()

# Play
for episode in range (num_testepis):
    state = env.reset ()
    done = False
    score = 0

    for step in range (num_steps):
        action = np.argmax (qtable[state,:])
        reward = env.step (action)
        score += reward
        state = new_state

    print ("score", score)

print (qtable)
