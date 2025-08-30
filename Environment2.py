import numpy as np

class Env:
	def __init__(self, x_max=10, y_max=10, default_start=(1.0,1.0), default_goal=(9.0,9.0)):
		self.X_max = float(x_max)
		self.Y_max = float(y_max)
		self.default_start = np.array(default_start, dtype=np.float32)
		self.default_goal = np.array(default_goal, dtype=np.float32)
		self.vector_agentState = self.default_start.copy()
		self.Terminal = self.default_goal.copy()
		self.steps_counter = 0
		self.index = 0
		self.dic = {}
		self.final_path = {}
		self.firstsuc = True
		self.shortest = int(1e9)
		self.longest = 0
		self.Is_Terminal = False
		self.doneType = 0
		# default obstacles (can be replaced by randomize_obstacles)
		self.obstacles = []
		self.success_tol = 1.5  # Default tolerance for success

	def reset(self, start=None, goal=None, randomize_obstacles=False, n_obstacles=6, max_size=20, seed=None):
		
		if seed is not None:
			np.random.seed(seed)
		if start is None:
			self.vector_agentState = self.default_start.copy()
		else:
			self.vector_agentState = np.array(start, dtype=np.float32)
		if goal is None:
			self.Terminal = self.default_goal.copy()
		else:
			self.Terminal = np.array(goal, dtype=np.float32)

		# reset bookkeeping
		self.steps_counter = 0
		self.index = 0
		self.dic = {}
		self.Is_Terminal = False
		self.doneType = 0

		if randomize_obstacles:
			self.randomize_obstacles(n_obstacles=n_obstacles, max_size=max_size)

		return self._get_obs()

	def _get_obs(self):
		# observation is [agent_x, agent_y, goal_x, goal_y]
		obs = np.concatenate([self.vector_agentState.copy(), self.Terminal.copy()])
		return obs.astype(np.float32)

	def randomize_obstacles(self, n_obstacles=6, max_size=None, min_size=0.5, clearance=2.0):
		"""
		Randomly generate rectangular obstacles, avoiding start and goal positions.
		Each obstacle: [x, y, width, height]
		"""
		self.obstacles = []
		tries = 0
		while len(self.obstacles) < n_obstacles and tries < n_obstacles * 10:
			width = np.random.uniform(min_size, max_size) if max_size else min_size
			height = np.random.uniform(min_size, max_size) if max_size else min_size
			x = np.random.uniform(0, self.X_max - width)
			y = np.random.uniform(0, self.Y_max - height)
			obs_rect = [x, y, width, height]
			# Check for overlap with start or goal
			sx, sy = self.default_start
			gx, gy = self.default_goal
			def is_clear(px, py):
				return not (x - clearance < px < x + width + clearance and y - clearance < py < y + height + clearance)
			if is_clear(sx, sy) and is_clear(gx, gy):
				self.obstacles.append(obs_rect)
			tries += 1

	def step(self, action):
			prev_pos = self.vector_agentState.copy()
			# Expect action in same units as environment (see suggestion 2 below).
			self.vector_agentState += np.array(action, dtype=np.float32)
			self.vector_agentState = np.clip(self.vector_agentState, [0,0], [self.X_max, self.Y_max])

			# --- collision detection: check whether new pos intersects any obstacle rect ---
			collided = False
			ax, ay = self.vector_agentState
			for (ox, oy, w, h) in self.obstacles:
				# rectangle contains point check
				if (ox <= ax <= ox + w) and (oy <= ay <= oy + h):
					collided = True
					break

			if collided:
				# Reject the move (simple) and penalize
				self.vector_agentState = prev_pos
				collision_penalty = -5.0  # tune this
			else:
				collision_penalty = 0.0

			self.steps_counter += 1

			# Check for goal
			dist_to_goal = np.linalg.norm(self.vector_agentState - self.Terminal)
			flag = dist_to_goal < self.success_tol if hasattr(self, 'success_tol') else dist_to_goal < 1.5
			done = flag or self.steps_counter >= 200

			# Make reward consistent and include collision penalty
			if flag:
				reward = 100.0
			else:
				reward = -dist_to_goal*0.1 - 1.0 + collision_penalty

			info = {'distance_to_goal': dist_to_goal, 'steps': self.steps_counter, 'collided': collided}
			next_obs = self._get_obs()
			return next_obs, flag, reward, done, info
	

if __name__ == "__main__":
	env = Env()
	env.reset(randomize_obstacles=True)
	print("Random Obstacles:", env.obstacles)
