import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StarBattleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=6, stars_per_row=2, regions=None):
        super().__init__()
        self.grid_size = grid_size
        self.stars_per_row = stars_per_row
        self.total_stars = stars_per_row * grid_size

        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int8
        )

        if regions is None:
            self.regions = self._generate_regions()
        else:
            self.regions = regions

        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.done = False
        return self.state.copy(), {}

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, self.done, False, {}

        row = action // self.grid_size
        col = action % self.grid_size

        reward = 0
        info = {}

        if self._is_valid_move(row, col):
            self.state[row, col] = 1
            reward += 1
            if self._check_win():
                reward += 100
                self.done = True
        else:
            reward -= 1

        return self.state.copy(), reward, self.done, False, info

    def render(self):
        print("\nCurrent Board:")
        for row in self.state:
            print(" ".join(['*' if x == 1 else '.' for x in row]))

    def _generate_regions(self):
        region_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        region_id = 0
        region_size = self.stars_per_row
        for r in range(0, self.grid_size, region_size):
            for c in range(0, self.grid_size, region_size):
                for i in range(region_size):
                    for j in range(region_size):
                        if r+i < self.grid_size and c+j < self.grid_size:
                            region_grid[r+i, c+j] = region_id
                region_id += 1
        return region_grid

    def _is_valid_move(self, row, col):
        if self.state[row, col] == 1:
            return False
        for i in range(-1, 2):
            for j in range(-1, 2):
                r, c = row + i, col + j
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    if self.state[r, c] == 1:
                        return False
        if np.sum(self.state[row, :]) >= self.stars_per_row:
            return False
        if np.sum(self.state[:, col]) >= self.stars_per_row:
            return False
        region_id = self.regions[row, col]
        if np.sum(self.state[self.regions == region_id]) >= self.stars_per_row:
            return False
        return True

    def _check_win(self):
        for i in range(self.grid_size):
            if np.sum(self.state[i, :]) != self.stars_per_row:
                return False
            if np.sum(self.state[:, i]) != self.stars_per_row:
                return False
        for region_id in np.unique(self.regions):
            if np.sum(self.state[self.regions == region_id]) != self.stars_per_row:
                return False
        return True

    def get_action_mask(self):
        mask = np.zeros(self.grid_size * self.grid_size, dtype=np.int8)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                idx = row * self.grid_size + col
                if self._is_valid_move(row, col):
                    mask[idx] = 1
        return mask