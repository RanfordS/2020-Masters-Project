import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# Create the environment
env = gym.make('BipedalWalker-v2')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# Define the model
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_bipedal_tensorboard/")

# Train the agent
model.learn(total_timesteps=25000)

# After training, watch our agent walk
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
