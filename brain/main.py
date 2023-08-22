"""
"""
import os

import gymnasium as gym
import numpy as np

import twolegged

from stable_baselines3 import PPO, A2C, DQN, TD3, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


vec_env = make_vec_env("TwoLegged-v0", n_envs=1, env_kwargs=dict(render_mode=None, render_fps=30))

n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Initialize the model
model = TD3(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
)

while True:
    if os.path.exists("test_save.zip"): model.load("test_save", env=vec_env)
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("test_save")

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

env.close()
