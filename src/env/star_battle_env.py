# Folder: star_battle_rl/

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
        done = False

        if not is_valid_action(self.board, row, col, self.region_layout):
            return self.board.copy(), 0, True, False, {}  # No reward for invalid move, ends episode

        self.board = apply_action(self.board, row, col)
        self.placed_stars += 1

        if self.placed_stars == self.board_size:
            done = True
            if check_win_condition(self.board, self.region_layout):
                reward = 1.0  # Only reward for a valid full solution
                print("Win!✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
            else:
                reward = -1.0  # No win, penalize or set to 0
        else:
            reward = 0.0  # No reward during episode

        return self.board.copy(), reward, done, False, {}

    def get_action_mask(self):
        return generate_mask(self.board, self.region_layout)

    def render(self):
        display = np.full_like(self.region_layout, fill_value=".", dtype=object)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    display[i, j] = "*"
                else:
                    display[i, j] = str(self.region_layout[i, j])
        print("\n".join(" ".join(row) for row in display))
        print("\n" + "="*20 + "\n")
