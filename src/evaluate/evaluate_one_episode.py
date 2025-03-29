# --- File: evaluate/evaluate_one_episode.py ---
import json
import numpy as np
import torch
from torch.nn.functional import softmax
from stable_baselines3 import PPO

from env.star_battle_env import StarBattleEnv
from env.star_battle_utils import generate_mask

# Load puzzles
with open("../puzzles/puzzles_6x6_100.json") as f:
    region_layouts = [np.array(p) for p in json.load(f)]

# Choose one random puzzle
region_layout = region_layouts[np.random.randint(len(region_layouts))]
env = StarBattleEnv(region_layout)

# Load trained model
model = PPO.load("../models/ppo_star_battle")

obs, _ = env.reset()
done = False
step_count = 0

print("Initial board:")
env.render()

while not done:
    mask = generate_mask(obs, env.region_layout, env.num_stars)
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    logits = model.policy.forward(obs_tensor)[0].detach().cpu().numpy().flatten()
    masked_logits = np.where(mask == 1, logits, -1e8)
    probs = softmax(torch.tensor(masked_logits), dim=0).numpy()

    action = int(np.argmax(probs))
    obs, reward, done, _, info = env.step(action)

    step_count += 1
    print(f"Step {step_count}: placed star at {divmod(action, env.board_size)}")
    env.render()

if reward > 0:
    print("✅ Puzzle solved!")
else:
    print("❌ Puzzle failed.")