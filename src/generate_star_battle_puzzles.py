import numpy as np
import random
import json
from itertools import product

BOARD_SIZE = 6
NUM_STARS = 1
MAX_TRIES = 100000

def generate_random_regions(board_size):
    n_cells = board_size * board_size
    region_ids = list(range(board_size)) * board_size
    random.shuffle(region_ids)
    layout = np.array(region_ids).reshape((board_size, board_size))
    return layout

def is_valid(board, row, col, regions, num_stars):
    if board[row, col] == 1:
        return False
    if np.sum(board[row, :]) >= num_stars:
        return False
    if np.sum(board[:, col]) >= num_stars:
        return False
    region_id = regions[row][col]
    if np.sum(board[regions == region_id]) >= num_stars:
        return False
    for i in range(max(0, row - 1), min(BOARD_SIZE, row + 2)):
        for j in range(max(0, col - 1), min(BOARD_SIZE, col + 2)):
            if board[i, j] == 1:
                return False
    return True

def solve(board, regions, row=0):
    if row == BOARD_SIZE:
        return True
    for col in range(BOARD_SIZE):
        if is_valid(board, row, col, regions, NUM_STARS):
            board[row, col] = 1
            if solve(board, regions, row + 1):
                return True
            board[row, col] = 0
    return False

def generate_valid_puzzle():
    for _ in range(100):
        regions = generate_random_regions(BOARD_SIZE)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        if solve(board, regions):
            return regions
    return None

def generate_multiple_puzzles(n=1000):
    puzzles = []
    attempts = 0
    while len(puzzles) < n and attempts < MAX_TRIES:
        regions = generate_valid_puzzle()
        if regions is not None:
            puzzles.append(regions.tolist())
        attempts += 1
    return puzzles

if __name__ == "__main__":
    puzzles = generate_multiple_puzzles(100)
    with open("puzzles_6x6_100_test.json", "w") as f:
        json.dump(puzzles, f)
    print(f"Generated {len(puzzles)} puzzles and saved to puzzles_6x6_1000.json")
