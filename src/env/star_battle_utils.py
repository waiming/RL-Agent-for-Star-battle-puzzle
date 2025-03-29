# --- File: env/star_battle_utils.py ---
import numpy as np
from collections import Counter

def is_valid_action(board, row, col, regions, num_stars):
    if board[row, col] == 1:
        return False

    if np.sum(board[row, :]) >= num_stars:
        return False
    if np.sum(board[:, col]) >= num_stars:
        return False

    region_id = regions[row][col]
    if np.sum(board[regions == region_id]) >= num_stars:
        return False

    for i in range(max(0, row-1), min(board.shape[0], row+2)):
        for j in range(max(0, col-1), min(board.shape[1], col+2)):
            if board[i, j] == 1:
                return False

    return True

def apply_action(board, row, col):
    new_board = board.copy()
    new_board[row, col] = 1
    return new_board

def check_win_condition(board, regions, num_stars):
    n = board.shape[0]
    if not all(np.sum(board, axis=1) == num_stars):
        return False
    if not all(np.sum(board, axis=0) == num_stars):
        return False
    region_ids = np.unique(regions)
    for r in region_ids:
        if np.sum(board[regions == r]) != num_stars:
            return False
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < n:
                            if board[ni, nj] == 1:
                                return False
    return True

def generate_mask(board, regions, num_stars):
    mask = np.zeros(board.size, dtype=np.int8)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if is_valid_action(board, i, j, regions, num_stars):
                mask[i * board.shape[1] + j] = 1
    return mask