import random
import numpy as np
import tensorflow as tf

class Chain:

    def __init__ (self):
        self.state = 0

    def step (self, action):
        if action == 1:
            print ("retr")
            self.state = 0
            return 1
        elif self.state == 4:
            print ("ding")
            return 10
        else:
            print ("prog")
            self.state += 1
            return 0
    # returns: reward

    def reset (self):
        self.state = 0

class Agent:

    def __init__ (self):
        self.replay = []
        self.epsilon = 0.2
        self.gamma = 0.8
        self.policy_net = self._build_model ()
        self.target_net = self._build_model ()
        self.policy_to_target ()

    def store (self, state_0, action, reward, state_1):
        self.replay.append ([state_0, action, reward, state_1])

    def _build_model (self):
        model = tf.keras.Sequential ()
        l = tf.keras.layers
        model.add (l.Dense (2, activation = 'linear', input_shape = (5,)))
        model.compile (loss = 'mse', optimizer = 'adam')
        return model

    def policy_to_target (self):
        self.target_net.set_weights (self.policy_net.get_weights ())

    def act (self, state):
        if random.random () <= self.epsilon:
            return random.randrange (2)
        q_values = self.policy_net.predict (state)
        return np.argmax (q_values[0])

    def train (self, batch_size):
        print ("Doing batch fit")
        batch = np.array (random.sample (self.replay, batch_size))
        fit_x = []
        fit_y = []
        for state_0, action, reward, state_1 in batch:
            target = self.policy_net.predict (np.array ([state_0]))[0]
            t = self.target_net.predict (np.array ([state_1]))
            target[action] = reward + self.gamma*np.amax (t)
            fit_x.append (state_0)
            fit_y.append (target) 
        fit_x = np.array (fit_x)
        fit_y = np.array (fit_y)
        self.policy_net.fit (fit_x, fit_y, epochs = 1, verbose = 1)


environment = Chain ()
agent = Agent ()
agent.policy_net.summary ()
batch_size = 200

for e in range (10):
    environment.reset ()
    score = 0
    for t in range (100):
        state_0 = np.zeros (5)
        state_0[environment.state] = 1.0

        action = agent.act (np.array ([state_0]))
        reward = environment.step (action)
        score += reward

        state_1 = np.zeros (5)
        state_1[environment.state] = 1.0
        agent.store (state_0, action, reward, state_1)

        if len (agent.replay) >= batch_size:
            agent.train (batch_size)
    print (score)
