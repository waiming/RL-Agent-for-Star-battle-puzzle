# imitation_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np

# --- Dataset ---
from imitation_dataset_loader import StarBattleImitationDataset

BOARD_SIZE = 6


# --- Model ---

class StarBattleImitationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_region = nn.Embedding(6, 8)  # region_id ∈ [0-5]
        self.embed_star = nn.Embedding(2, 4)    # star ∈ [0,1]
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 36)  # logits for 6x6 cells
        )

    def forward(self, x):
        x = x.long()  # Ensure integer type
        region = x[:, 0]  # (B, 6, 6)
        star = x[:, 1]    # (B, 6, 6)

        r = self.embed_region(region)  # (B, 6, 6, 8)
        s = self.embed_star(star)      # (B, 6, 6, 4)

        x = torch.cat([r, s], dim=-1).permute(0, 3, 1, 2)  # (B, 12, 6, 6)
        x = self.conv(x)
        x = self.fc(x)
        return x

# --- Training ---
def train():
    dataset = StarBattleImitationDataset("/Users/wmwong/Documents/GitHub/RL-Agent-for-Star-battle-puzzle/src/solver/solved_6x6_puzzles_with_steps.json")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = StarBattleImitationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        total_loss = 0
        for boards, actions in loader:
            logits = model(boards)
            loss = loss_fn(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    
    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "imitation_model_6x6.pth")
    