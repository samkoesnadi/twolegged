"""
"""
import os

import gymnasium as gym
import numpy as np

import twolegged

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, A2C, DQN, TD3, HerReplayBuffer, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


IS_TRAIN = False

vec_env = make_vec_env("TwoLegged-v0", n_envs=1 if IS_TRAIN else 1, env_kwargs=dict(render_mode=None if IS_TRAIN else "human", render_fps=30))
vec_env = VecNormalize(
    vec_env, norm_obs=True, norm_reward=True,
    clip_obs=15.
)

n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Initialize the model
# model = SAC(
#     "MlpPolicy",
#     vec_env,
#     verbose=1,
# )

model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    verbose=1,
    use_sde=True
)

if os.path.exists("test_save.zip"):
            model = model.load("test_save", env=vec_env)
            print("model loaded")

if IS_TRAIN:
    while True:
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save("test_save")
else:
    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

env.close()
