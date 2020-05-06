import numpy as np
import tensorflow as tf

class Chain:
    def __init__ (self):
        self.state = 0

    def reset (self):
        self.state = 0
    
    def step (self, action):
        # if action, return to start with minor reward
        if action == 1:
            self.state = 0
            return 1
        # if at end, stay with major reward
        elif self.state == 4:
            return 5
        # advance with no reward
        else:
            self.state += 1
            return 0

class Distribution (tf.keras.Model):
    def call (self, logits, **kwargs):
        return tf.squeeze (tf.random.categorical (logits, 1), axis = -1)

class Model (tf.keras.Model):
    def __init__ (self):
        super ().__init__ ('mlp_policy')
        l = tf.keras.layers
        self.hidden = l.Dense (4, activation = 'tanh')
        # probability of action 0/1
        self.action = l.Dense (2, activation = 'softmax')
        self.distri = Distribution ()

    def call (self, inputs, **kwargs):
        x = tf.convert_to_tensor (inputs)
        hidden = self.hidden (x)
        action = self.action (hidden)
        return action, action

    def get_action (self, observations):
        logits, value = self.predict_on_batch (observations)
        action = self.distri.predict_on_batch (logits)
        result = np.squeeze (action, axis = -1)
        return result

class Agent:
    def __init__ (self, model):
        self.model = model
        model.compile (optimizer = 'RMSprop',
                       loss = [self._logits_loss, self._value_loss])

    def train (self, environment):
        for epoch in range (100):
            for step in range (25):
                self.model.get_action (environment.state)
