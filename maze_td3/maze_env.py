import os
def load_custom_points(filename="selected_points.txt"):
    if not os.path.exists(filename):
        return None, None
    with open(filename, "r") as f:
        lines = f.read().strip().split("\n")
        if len(lines) >= 2:
            start = tuple(map(float, lines[0].split(",")))
            goal = tuple(map(float, lines[1].split(",")))
            return start, goal
    return None, None
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MazeEnv(gym.Env):
    """10x10 grid maze environment for continuous pathfinding."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, start_pos=None, goal_pos=None):
        super().__init__()
        self.grid_size = 10
        self.x_max = self.grid_size
        self.y_max = self.grid_size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=self.grid_size, shape=(4 + self.grid_size**2,), dtype=np.float32)
        self.render_mode = render_mode
        self.max_steps = 400
        self.success_tolerance = 0.5
        self._generate_obstacles()
        self._custom_start_pos = start_pos
        self._custom_goal_pos = goal_pos
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._custom_start_pos is not None:
            self.agent_pos = np.array(self._custom_start_pos, dtype=np.float32)
        else:
            self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        if self._custom_goal_pos is not None:
            self.goal_pos = np.array(self._custom_goal_pos, dtype=np.float32)
        else:
            self.goal_pos = np.array([self.x_max - 1, self.y_max - 1], dtype=np.float32)
        self.steps = 0
        self.done = False
        obs = self._get_obs()
        info = {}
        return obs, info

    
    def step(self, action):
        action = np.clip(action, -1, 1)
        move = action * 0.7  # scale movement
        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        # Check for collision with obstacles
        grid_x, grid_y = int(round(new_pos[0])), int(round(new_pos[1]))
        stagnation_penalty = 0.0
        if np.allclose(new_pos, self.agent_pos):
            stagnation_penalty = -2.0  # penalize no movement
        if self.obstacles[grid_x, grid_y] == 1:
            reward = -5.0 + stagnation_penalty
            new_pos = self.agent_pos  # stay in place
        else:
            reward = -0.01 + stagnation_penalty
            self.agent_pos = new_pos
        self.steps += 1
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        if dist_to_goal < self.success_tolerance:
            reward += 100.0
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True
        obs = self._get_obs()
        info = {'distance_to_goal': dist_to_goal}
        return obs, reward, self.done, False, info

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = '.'
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x, y] == 1:
                    grid[x, y] = '#'
        ax, ay = int(round(self.agent_pos[0])), int(round(self.agent_pos[1]))
        gx, gy = int(round(self.goal_pos[0])), int(round(self.goal_pos[1]))
        grid[ax, ay] = 'A'
        grid[gx, gy] = 'G'
        print('\n'.join(' '.join(row) for row in grid))

    def _generate_obstacles(self):
        # Randomly place obstacles, avoiding start and goal
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        num_obstacles = np.random.randint(6, 12)  # random number of obstacles
        forbidden = {(0, 0), (self.grid_size - 1, self.grid_size - 1)}
        placed = 0
        while placed < num_obstacles:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in forbidden and self.obstacles[x, y] == 0:
                self.obstacles[x, y] = 1
                placed += 1

    def _get_obs(self):
        # Flattened obstacles
        return np.concatenate([
            self.agent_pos,
            self.goal_pos,
            self.obstacles.flatten()
        ]).astype(np.float32)

  ####  def _generate_obstacles(self):
        # Fixed obstacle layout
    #    self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
      #  fixed_obstacles = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 6), (5, 7), (6, 7)]
       # for (x, y) in fixed_obstacles:
       #     self.obstacles[x, y] = 1

    

    def close(self):
        pass
