import gym
import gridenv
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# multiprocess environment
env = DummyVecEnv([lambda: gym.make('no_noise-v2')])
env_noisy = DummyVecEnv([lambda: gym.make('noise-v2')])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo2")

# del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2")

# Enjoy trained agent
env = env_noisy
obs = env.reset()
count = 0
while count <10000:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    count = count + 1
