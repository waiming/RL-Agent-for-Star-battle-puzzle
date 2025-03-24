import json
from typing import List
import copy

Grid = List[List[str]]


def print_board(board: Grid):
    for row in board:
        print(" ".join(row))
    print()


def is_valid(board: Grid, row: int, col: int, region_map: Grid, stars: int) -> bool:
    size = len(board)
    region_id = region_map[row][col]

    # Count stars in row
    if sum(1 for c in board[row] if c == '*') >= stars:
        return False

    # Count stars in column
    if sum(1 for r in range(size) if board[r][col] == '*') >= stars:
        return False

    # Count stars in region
    if sum(1 for r in range(size) for c in range(size) if region_map[r][c] == region_id and board[r][c] == '*') >= stars:
        return False

    # Check adjacent cells
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == '*':
                return False

    return True


def solve_star_battle(region_map: Grid, stars: int = 1) -> Grid:
    size = len(region_map)
    board = [['.' for _ in range(size)] for _ in range(size)]

    def backtrack(r=0, c=0):
        if r == size:
            return is_complete(board, region_map, stars)

        nr, nc = (r, c + 1) if c + 1 < size else (r + 1, 0)

        # Try placing a star
        if is_valid(board, r, c, region_map, stars):
            board[r][c] = '*'
            if backtrack(nr, nc):
                return True
            board[r][c] = '.'

        # Try without placing a star
        if backtrack(nr, nc):
            return True

        return False

    solved = backtrack()
    return board if solved else None


def is_complete(board: Grid, region_map: Grid, stars: int) -> bool:
    size = len(board)

    # Check each row, column, and region
    for i in range(size):
        if sum(1 for j in range(size) if board[i][j] == '*') != stars:
            return False
        if sum(1 for j in range(size) if board[j][i] == '*') != stars:
            return False

    regions = set(cell for row in region_map for cell in row)
    for region_id in regions:
        if sum(1 for r in range(size) for c in range(size) if region_map[r][c] == region_id and board[r][c] == '*') != stars:
            return False

    return True


# Load region maps
with open('./puzzles/puzzles_6x6_100.json', 'r') as f:
    region_maps = json.load(f)

print(f"Solving {len(region_maps)} puzzles...")

for idx, region_map in enumerate(region_maps):
    print(f"\nPuzzle {idx + 1}")
    solution = solve_star_battle(region_map, stars=1)
    if solution:
        print_board(solution)
    else:
        print("No solution found.")