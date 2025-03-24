import json
import numpy as np
from torch.utils.data import Dataset
import torch

class StarBattleImitationDataset(Dataset):
    def __init__(self, json_path):
        self.states = []
        self.actions = []

        with open(json_path, "r") as f:
            puzzles = json.load(f)

        for puzzle in puzzles:
            board = np.zeros((6, 6), dtype=np.int64)
            region_map = np.array(puzzle["region_map"], dtype=np.int64)

            for step in puzzle["steps"]:
                if step["action"] == "place_star":
                    state = np.stack([region_map, board.copy()])
                    self.states.append(torch.tensor(state, dtype=torch.long))
                    row, col = step["cell"]
                    action_index = row * 6 + col
                    self.actions.append(torch.tensor(action_index, dtype=torch.long))
                    board[row, col] = 1  # Update board after saving state
                elif step["action"] == "remove_star":
                    row, col = step["cell"]
                    board[row, col] = 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# test_loader.py
# from imitation_dataset_loader import StarBattleImitationDataset

# dataset = StarBattleImitationDataset("./solver/solved_6x6_puzzles_with_steps.json")
# print("Number of steps in dataset:", len(dataset))

# # Visualize a sample
# state, action = dataset[0]
# print("State shape:", state.shape)  # Should be (2, 6, 6)
# print("Action (flattened index):", action)