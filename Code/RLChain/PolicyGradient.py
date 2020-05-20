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

lr = 0.8
gamma = 0.7

#eps = 1.0
#max_eps = 1.0
#min_eps = 0.01

data_filename = "DataPolicyGradient.csv"

# State array
def index_to_array (index, size):
    z = np.zeros (size)
    z[index] = 1.0
    return z

# Discount
def discount (episode_rewards):
    discounted = np.zeros_like (episode_rewards)
    acc = 0
    for i in reversed (range (len (episode_rewards))):
        acc = acc*gamma + episode_rewards[i]
        discounted[i] = acc
    return discounted
# Normalize
def normalize (data):
    return (data - np.mean (data)) / np.std (data)

# Network
#model = tf.keras.models.Sequential ()
#l = tf.keras.layers
#model.add (l.Dense (2, input_dim = 5, activation = 'softmax'))
#model.compile (optimizer = 'adam', loss = 'mean_squared_error', metrics = [])
class Agent (object):
    def __init__ (self):
        # model
        l = tf.keras.layers
        self.input = l.Input (shape = (5,))
        self.layer = l.Dense (2, input_dim = 5, activation = 'softmax')(self.input)
        self.model = tf.keras.models.Model (inputs = self.input, outputs = self.layer)
        # training function
        k = tf.keras.backend
        action_taken = k.placeholder (shape = (None, 2), name = "action_taken")
        discount = k.placeholder (shape = (None,), name = "discounted_reward")
        action_prob = k.log (k.sum (self.layer*action_taken, axis = 1))
        loss = k.mean (-action_prob*discount)
        adam = tf.keras.optimizers.Adam ()
        update = adam.get_updates (params = self.model.trainable_weights,
                                   #constraints = [],
                                   loss = loss)
        self.train_f = k.function (inputs = [self.model.input,
                                             action_taken,
                                             discount],
                                   outputs = [self.layer],
                                   updates = update)
    def act (self, state):
        dist = self.model.predict (state)[0]
        return np.random.choice (range (2), p = dist)
    def fit (self, state0s, actions, rewards):
        rewards = normalize (discount (rewards))
        print ("state0s", state0s.shape)
        print ("actions", actions.shape)
        print ("rewards", rewards.shape)
        self.train_f ([state0s, actions, rewards])
# end
agent = Agent ()

# Training
episode_scores = []
episode_rewards = []
for episode in range (num_episodes):
    #state0 = state_to_array (env.reset ())
    env.reset ()

    state0s = []
    actions = []
    rewards = []
    for step in range (num_steps):
        state0 = index_to_array (env.state, 5)
        state0s.append (state0)
        # choose action
        #distribution = model.predict (np.array ([state0]))[0]
        #action = np.random.choice (range (2), p = distribution)
        action = agent.act (np.array ([state0]))
        actions.append (index_to_array (action, 2))
        # perform
        reward = env.step (action)
        rewards.append (reward)
    episode_scores.append (sum (rewards))
    #rewards = normalize (discount (rewards))
    #episode_rewards.append (rewards)
    
    state0s = np.array (state0s)
    actions = np.array (actions)
    rewards = np.array (rewards)
    #model.fit (state0s, rewards, epochs = 10, batch_size = num_steps, verbose = 0)
    agent.fit (state0s, actions, rewards)

    print ("episode {0:3d}, score {1:3d}, dings {2:2d}"
            .format (episode, episode_scores[episode], env.dings))
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
