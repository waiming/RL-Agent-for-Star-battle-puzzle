# --- File: evaluate/evaluate_one_episode.py ---
import json
import numpy as np
import torch
from torch.nn.functional import softmax
from stable_baselines3 import PPO
from env.star_battle_env import StarBattleEnv

with open("puzzles/puzzles_6x6.json") as f:
    region_layout = np.array(json.load(f))

env = StarBattleEnv(region_layout)
model = PPO.load("models/ppo_star_battle")

obs, _ = env.reset()
done = False
step_count = 0

print("Initial board:")
env.render()

while not done:
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    logits = model.policy.forward(obs_tensor)[0].detach().cpu().numpy().flatten()

    mask = env.get_action_mask()
    masked_logits = np.where(mask == 1, logits, -1e8)
    probs = softmax(torch.tensor(masked_logits), dim=0).numpy()

    action = int(np.argmax(probs))
    obs, reward, done, _, _ = env.step(action)

    step_count += 1
    print(f"Step {step_count}: placed star at {divmod(action, env.board_size)}")
    env.render()

if reward == 1:
    print("✅ Puzzle solved!")
else:
    print("❌ Failed to solve the puzzle.")
