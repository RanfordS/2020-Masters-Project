# Imports
import numpy as np
import random
import matplotlib.pyplot as plt

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
        if self.state == 4:
            return 10
        self.state += 1
        return 0
env = Chain ()

# Q-Table
act_size = env.action_space
state_size = env.observation_space

qtable = np.zeros ((state_size, act_size))
print (qtable)

# Hyperparameters
num_episodes = 200
num_testepis = 1
num_steps = 50

lr = 0.8
gamma = 0.7

eps = 1.0
max_eps = 1.0
min_eps = 0.01

data_filename = "DataQLearning.csv"

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
