# --- File: env/star_battle_env.py ---
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary
from env.star_battle_utils import is_valid_action, check_win_condition, apply_action, generate_mask

BOARD_SIZE = 6
NUM_STARS = 1

class StarBattleEnv(gym.Env):
    def __init__(self, region_layout):
        super().__init__()
        self.board_size = BOARD_SIZE
        self.num_stars = NUM_STARS
        self.region_layout = region_layout
        self.action_space = Discrete(self.board_size * self.board_size)
        self.observation_space = MultiBinary((self.board_size, self.board_size))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.placed_stars = 0
        return self.board.copy(), {}

    def step(self, action):
        row, col = divmod(action, self.board_size)

        if not is_valid_action(self.board, row, col, self.region_layout, self.num_stars):
            reward = -1.0  # Strong penalty
            done = True    # End episode immediately
            return self.board.copy(), reward, done, False, {"invalid_action": True}

        self.board[row, col] = 1
        self.placed_stars += 1

        reward = 0.2  # Reward for a valid placement
        done = False

        if check_win_condition(self.board, self.region_layout, self.num_stars):
            reward += 10.0  # Bonus for winning
            done = True

        if self.placed_stars >= self.board_size * self.num_stars:
            done = True  # Safety net to avoid overlong episodes

        return self.board.copy(), reward, done, False, {}
    
    def render(self):
        print("\nCurrent Board State:\t\tRegion Layout:")
        for i in range(self.board_size):
            board_row = ""
            region_row = ""
            for j in range(self.board_size):
                cell = self.board[i, j]
                board_row += "★ " if cell == 1 else ". "
                region_row += f"{self.region_layout[i][j]} "
            print(f"{board_row}\t\t{region_row}")
        print("-" * 40)