# --- File: env/star_battle_utils.py ---
import numpy as np
from collections import Counter

def is_valid_action(board, row, col, regions):
    if board[row, col] == 1:
        return False

    # Only one star per row
    if np.sum(board[row, :]) >= 1:
        return False

    # Only one star per column
    if np.sum(board[:, col]) >= 1:
        return False

    # Only one star per region
    region_id = regions[row][col]
    if np.sum(board[regions == region_id]) >= 1:
        return False

    # No adjacent stars
    for i in range(max(0, row-1), min(board.shape[0], row+2)):
        for j in range(max(0, col-1), min(board.shape[1], col+2)):
            if board[i, j] == 1:
                return False

    return True

def apply_action(board, row, col):
    new_board = board.copy()
    new_board[row, col] = 1
    return new_board

def check_win_condition(board, regions):
    n = board.shape[0]
    # ✅ Check 1: Each row must have exactly one star
    if not all(np.sum(board, axis=1) == 1):
        return False

    # ✅ Check 2: Each column must have at most one star
    # (This should be == 1 if strictly enforcing one star per column)
    if not all(np.sum(board, axis=0) == 1):
        return False

    # ✅ Check 3: Each region must have exactly one star
    region_ids = np.unique(regions)
    for r in region_ids:
        if np.sum(board[regions == r]) != 1:
            return False

    # ✅ Rule 4: No adjacent stars (including diagonals)
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # skip self
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < n:
                            if board[ni, nj] == 1:
                                return False
                            
    # All conditions passed
    return True

def generate_mask(board, regions):
    mask = np.zeros(board.size, dtype=np.int8)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if is_valid_action(board, i, j, regions):
                mask[i * board.shape[1] + j] = 1
    return mask


