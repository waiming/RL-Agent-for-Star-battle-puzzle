import numpy as np
import torch
from star_battle_env import StarBattleEnv
from region_presets import custom_regions_6x6
from stable_baselines3 import PPO

env = StarBattleEnv(grid_size=6, stars_per_row=2, regions=custom_regions_6x6)

# ✅ Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = PPO.load("ppo_star_battle", device=device)

obs, _ = env.reset()
env.render()
done = False

while not done:
    mask = env.get_action_mask()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model.policy.forward(obs_tensor)[0].cpu().numpy().squeeze()

    logits[mask == 0] = -np.inf
    action = np.argmax(logits)

    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    env.render()