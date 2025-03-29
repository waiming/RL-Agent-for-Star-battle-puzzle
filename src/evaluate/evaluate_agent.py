# --- File: evaluate/evaluate_agent.py ---
import json
import numpy as np
import random
from stable_baselines3 import PPO
import torch

from env.star_battle_env import StarBattleEnv
from env.star_battle_utils import generate_mask

# Load multiple region layouts
with open("../puzzles/puzzles_6x6_1000.json") as f:
    region_layouts = [np.array(p) for p in json.load(f)]

# Load trained PPO model
model = PPO.load("../models/ppo_star_battle")

# Evaluate across N episodes
n_eval_episodes = 5000
wins = 0

for _ in range(n_eval_episodes):
    region_layout = random.choice(region_layouts)
    env = StarBattleEnv(region_layout)
    obs, _ = env.reset()
    done = False

    while not done:
        mask = generate_mask(obs, env.region_layout, env.num_stars)
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        logits = model.policy.forward(obs_tensor)[0].detach().cpu().numpy().flatten()
        masked_logits = np.where(mask == 1, logits, -1e8)
        action = np.argmax(masked_logits)

        obs, reward, done, truncated, info = env.step(action)

    if reward > 0:
        wins += 1

print(f"Win rate over {n_eval_episodes} episodes: {wins / n_eval_episodes:.2%}")