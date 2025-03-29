# RL-Agent-for-Star-battle-puzzle
Reinforcement Learning Agent for Star battle puzzle

## Introduction
This project trains a reinforcement learning agent using PPO to solve the 6x6, 1-star-per-row Puzzle Star Battle.

## Structure
- `models/`: trained models
- `puzzles/`: 6x6 region preset
- `src/`: main source code
- `src/env/`: Custom environment and logic
- `src/evaluate/`: Evaluate win rate across episodes
- `src/train/`: PPO training script

## Train
```bash
cd src
python train/train_agent.py
```

## Evaluate (all episodes)
```bash
cd src
python evaluate/evaluate_agent.py
```

## Evaluate (one episode with step-by-step output)
```bash
cd src
python evaluate/evaluate_one_episode.py
```
