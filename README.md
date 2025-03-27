# RL-Agent-for-Star-battle-puzzle
Reinforcement Learning Agent for Star battle puzzle (in progress)

## Introduction
This project trains a reinforcement learning agent using PPO to solve the 6x6, 1-star-per-row Puzzle Star Battle.

## Structure
- `env/`: Custom environment and logic
- `train/`: PPO training script
- `evaluate/`: Evaluate win rate across episodes
- `puzzles/`: 6x6 region preset

## Train
```bash
PYTHONPATH=. python train/train_agent.py
```

## Evaluate (all episodes)
```bash
PYTHONPATH=. python evaluate/evaluate_agent.py
```

## Evaluate (one episode with step-by-step output)
```bash
PYTHONPATH=. python evaluate/evaluate_one_episode.py
```
