import numpy as np
import tensorflow as tf
import gym

GAMMA = 0.95
env = gym.make ('CartPole-v0')
state_size = 4
num_actions = env.action_space.n

model = tf.keras.Sequential ()
model.add (tf.keras.layers.Dense (30,
    activation = 'relu',
    kernel_initializer = tf.keras.initializers.he_normal ()))
model.add (tf.keras.layers.Dense (30,
    activation = 'relu',
    kernel_initializer = tf.keras.initializers.he_normal ()))
model.add (tf.keras.layers.Dense (num_actions,
    activation = 'softmax'))
model.compile (loss='categorical_crossentropy',
               optimizer = tf.keras.optimizers.Adam ())

def get_action (state):
    global model, num_actions
    distrib = model (state.reshape ((1,-1)))
    action = np.random.choice (num_actions, p = distrib.numpy()[0])
    return action

def update_model (rewards, states, actions):
    global model, num_actions
    acc = 0
    discounted = np.empty_like (rewards)
    for i in reversed (range (len (rewards))):
        acc = GAMMA*acc + rewards[i]
        discounted[i] = acc
    discounted = np.array (discounted)
    discounted = (discounted - np.mean (discounted)) / np.std (discounted)
    states = np.vstack (states)
    return model.train_on_batch (states, discounted)

for episode in range (50):
    state = env.reset ()
    rewards = []
    states  = []
    actions = []
    while True:
        action = get_action (state)
        new_state, reward, done, info = env.step (action)
        states.append (state)
        rewards.append (reward)
        actions.append (action)
        state = new_state
        if done: break
    loss = update_model (rewards, states, actions)
    print ("reward = {0:3d}".format (sum (rewards)))
