import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from curriculum import curriculum_env

# 🧠 Choose curriculum level
level = 1

# 🧱 Build vectorized environment
vec_env = make_vec_env(lambda: curriculum_env(level), n_envs=4)

# ✅ Check if MPS is available
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# 🎯 Train the model on MPS (if available)
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard/star_battle",
    device=device
)

model.learn(total_timesteps=100_000)
model.save("ppo_star_battle")