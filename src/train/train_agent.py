# --- File: train/train_agent.py ---
import json
import numpy as np
import random
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.star_battle_env import StarBattleEnv

BOARD_SIZE = 6
NUM_STARS = 1
MODEL_PATH = "../models/ppo_star_battle"

# Load multiple puzzles
with open("../puzzles/puzzles_6x6.json") as f:
    all_layouts = [np.array(p) for p in json.load(f)]

def make_env():
    return StarBattleEnv(region_layout=random.choice(all_layouts))

# Create training environments
vec_env = make_vec_env(make_env, n_envs=4)

# Load or create model
if Path(MODEL_PATH + ".zip").exists():
    print("🔄 Loading existing model...")
    model = PPO.load(MODEL_PATH, env=vec_env)
else:
    print("🆕 Creating new model...")
    model = PPO("MlpPolicy", vec_env, learning_rate=1e-4, verbose=1)

# Train and save model
model.learn(total_timesteps=500_000)
model.save(MODEL_PATH)