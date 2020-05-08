# Imports
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque

# Hyperparameters
memory_size = 20
pretrain_length = 20

num_episodes = 200
num_steps = 50

max_eps = 1.0
min_eps = 0.01
eps = max_eps

lr = 0.8
gamma = 0.7

# Environment
class Chain:
    def __init__ (self):
        self.state = 0
        self.action_space = 2
        self.observation_space = 5
    def reset (self):
        self.state = 0
        return 0
    def step (self, action):
        if action == 1:
            self.state = 0
            return 1
        if state == 4:
            return 10
        self.state += 1
        return 0
env = Chain ()

def state_to_array (index):
    z = np.array.zeros (5)
    z[index] = 1.0
    return z

class DQNetwork:
    def __init__ (self):
        global lr
        with tf.variable_scope ("DQNetwork"):
            self.inputs = tf.placeholder (tf.float32, [None, 5], name = "inputs")
            self.actions = tf.placeholder (tf.float32, [None, 2], name = "actions")
            self.target_Q = tf.placeholder (tf.float32, [None], name = "target")
            self.output = tf.layers.dense (inputs = self.inputs, units = 2, activation = tf.nn.softmax)
            self.Q = tf.reduce_sum (tf.multiply (self.output, self.actions))
            self.loss = tf.reduce_mean (tf.square (self.target_Q - self.Q))
            self.optimizer = tf.train.AdamOptimizer (lr).minimize (self.loss)
#tf.reset_default_graph ()
network = DQNetwork ()

class Memory:
    def __init__ (self, max_size):
        self.buffer = deque (maxlen = max_size)
    def add (self, experience):
        self.buffer.append (experience)
    def sample (self, batch_size):
        return random.sample (self.buffer, batch_size)
memory = Memory (memory_size)

env.reset ()
for i in range (pretrain_length):
    state_0 = state_to_array (env.state)
    action  = random.randrange (2)
    reward  = env.step (action)
    state_1 = state_to_array (env.state)
    memory.add ((state_0, action, reward, state_1))

def predict_action (state):
    if random.random () < eps:
        return randon.randrange (2)
    Qs = sess.run (network.output, feed_dict = {network.inputs: state.reshape ((1, 5))})
    return np.argmax (Qs)

with tf.Session () as sess:
    sess.run (tf.global_variables_initializer ())
    for episode in range (num_episodes):
        env.reset ()
        score = 0
        for step in range (num_steps):
            state_0 = state_to_array (env.state)
            action  = predict_action (2)
            reward  = env.step (action)
            state_1 = state_to_array (env.state)
            memory.add ((state_0, action, reward, state_1))
            score += reward
        batch = memory.sample (batch_size)
        batch_state_0 = np.array ([ex[0] for ex in batch])
        batch_actions = np.array ([ex[1] for ex in batch])
        batch_rewards = np.array ([ex[2] for ex in batch])
        batch_state_1 = np.array ([ex[3] for ex in batch])
        target_Qs_batch = []
        Qs_next_state = sess.run (network.output, feed_dict = {network.inputs: batch_state_1.reshape ((1, 5))})
        for i in range (batch_size):
            target = batch_rewards[i] + gamma*np.max (Qs_next_state[i])
            target_Qs_batch.append (target)
        batch_targets = np.array (target_Qs_batch)
        loss, _ = sess.run ([network.loss, network.optimizer],
                feed_dict = {
                    network.inputs: batch_state_0,
                    network.target_Q: batch_targets,
                    network.actions: batch_actions})
        print (score)
        eps = max_eps*(min_eps/max_eps)**(episode/num_episodes)


# Q-Table
act_size = env.action_space
state_size = env.observation_space

qtable = np.zeros ((state_size, act_size))
print (qtable)

# Hyperparameters
num_episodes = 200
num_testepis = 1
num_steps = 50



data_filename = False#"DataDeepQLearning.csv"

# Training
scores = []
for episode in range (num_episodes):
    state = env.reset ()
    done = False

    score = 0
    for step in range (num_steps):
        if random.random () > eps:
            action = np.argmax (qtable[state,:])
        else:
            action = random.randrange (act_size)

        reward = env.step (action)
        score += reward
        new_state = env.state

        qtable[state,action] = qtable[state,action] + lr*(reward + gamma*np.max (qtable[new_state,:]) - qtable[state,action])
        state = new_state
    scores.append (score)
    #eps = min_eps + (max_eps - min_eps)*np.exp (-dr*episode)
    eps = max_eps*(min_eps/max_eps)**(episode/num_episodes)

if data_filename:
    with open (data_filename, 'w') as f:
        for i in range (num_episodes):
            f.write ("{0:d},{1:d}\n".format (i, scores[i]))

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
