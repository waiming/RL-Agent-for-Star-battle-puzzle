# --- File: train/train_agent.py ---
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.star_battle_env import StarBattleEnv

with open("puzzles/puzzles_6x6.json") as f:
    region_layout = np.array(json.load(f))

def make_env():
    return StarBattleEnv(region_layout)

vec_env = make_vec_env(make_env, n_envs=4)

# 2 out of 1 choices
# new model
model = PPO("MlpPolicy", vec_env, learning_rate=1e-4, verbose=1)
# Load trained model
# model = PPO.load("models/ppo_star_battle", env=vec_env)

model.learn(total_timesteps=500_000)
model.save("models/ppo_star_battle")

