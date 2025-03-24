# --- File: evaluate/evaluate_agent.py ---
import json
import numpy as np
from stable_baselines3 import PPO
import torch
from torch.nn.functional import softmax

from env.star_battle_env import StarBattleEnv

with open("puzzles/puzzles_6x6.json") as f:
    region_layout = np.array(json.load(f))

env = StarBattleEnv(region_layout)
model = PPO.load("models/ppo_star_battle")

n_eval_episodes = 5000
wins = 0

for _ in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        mask = env.get_action_mask()
        # Get model's raw logits
        obs_tensor = model.policy.obs_to_tensor(obs)[0]  # Convert obs to tensor
        logits = model.policy.forward(obs_tensor)[0]     # Get raw logits (before softmax)
        logits = logits.detach().cpu().numpy().flatten()

        # Apply action mask
        mask = env.get_action_mask()
        masked_logits = np.where(mask == 1, logits, -1e8)  # Very negative number for invalid actions

        # Apply softmax to get probabilities (optional)
        probs = softmax(torch.tensor(masked_logits), dim=0).numpy()

        # Sample or take max (greedy)
        action = int(np.argmax(probs))  # use np.random.choice(...) if sampling
        obs, reward, done, _, _ = env.step(action)
    if reward == 1:
        wins += 1

print(f"Win rate over {n_eval_episodes} episodes: {wins / n_eval_episodes * 10000:.2f}% wins: {wins}")

