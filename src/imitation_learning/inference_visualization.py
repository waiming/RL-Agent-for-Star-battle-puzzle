import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from imitation_model import StarBattleImitationModel
from imitation_dataset_loader import StarBattleImitationDataset

def predict_next_action(model, board_tensor):
    model.eval()
    board_tensor = board_tensor.unsqueeze(0)  # (1, 2, 6, 6)
    with torch.no_grad():
        logits = model(board_tensor)         # (1, 36)
        probs = F.softmax(logits, dim=-1)    # (1, 36)
        action = torch.argmax(probs, dim=-1) # (1,)
    return action.item(), probs.squeeze().view(6, 6)  # flat index, 6x6 grid

def visualize_prediction(board_tensor, true_action, pred_action, prob_grid, sample_id):
    fig, ax = plt.subplots()
    true_row, true_col = divmod(true_action.item(), 6)
    pred_row, pred_col = divmod(pred_action, 6)

    ax.set_title(f"Sample {sample_id} | True: ({true_row}, {true_col}) | Pred: ({pred_row}, {pred_col})")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.imshow(prob_grid.numpy(), cmap='Blues', vmin=0, vmax=1)

    ax.scatter(true_col, true_row, c='green', marker='o', s=150, label="True")
    ax.scatter(pred_col, pred_row, c='red', marker='x', s=150, label="Predicted")
    ax.legend(loc='upper right')
    plt.show()

def main():
    print("Loading model and dataset...")
    model = StarBattleImitationModel()
    model.load_state_dict(torch.load("/Users/wmwong/Documents/GitHub/RL-Agent-for-Star-battle-puzzle/src/imitation_learning/imitation_model_6x6.pth"))
    model.eval()

    dataset = StarBattleImitationDataset("/Users/wmwong/Documents/GitHub/RL-Agent-for-Star-battle-puzzle/src/solver/solved_6x6_puzzles_with_steps.json")
    print(f"Dataset loaded with {len(dataset)} samples.")

    num_samples = 5  # change this to visualize more
    sample_indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(sample_indices):
        board_tensor, true_action = dataset[idx]
        pred_action, prob_grid = predict_next_action(model, board_tensor)
        visualize_prediction(board_tensor, true_action, pred_action, prob_grid, idx)

if __name__ == "__main__":
    main()