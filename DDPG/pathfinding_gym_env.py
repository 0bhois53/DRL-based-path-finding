from time import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PathfindingGymEnv(gym.Env):
    """Standalone Gymnasium environment for pathfinding with obstacle avoidance"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, 
                 x_max=10.0, 
                 y_max=10.0, 
                 default_start=(1.0, 1.0), 
                 default_goal=(9.0, 9.0),
                 max_episode_steps=300,
                 success_tolerance=1.5,
                 render_mode=None):
        
        super().__init__()
        
        # Environment parameters
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.default_start = np.array(default_start, dtype=np.float32)
        self.default_goal = np.array(default_goal, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.success_tolerance = success_tolerance
        self.render_mode = render_mode
        
        # Action space: continuous 2D movement
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0.0, high=max(x_max, y_max), shape=(4,), dtype=np.float32
        )
        
        # State variables
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.goal_position = np.zeros(2, dtype=np.float32)
        self.obstacles = []
        self.step_count = 0
        self.prev_distance_to_goal = 0.0
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Set start and goal positions
        if options and 'start' in options:
            self.agent_position = np.array(options['start'], dtype=np.float32)
        else:
            self.agent_position = self.default_start.copy()
            
        if options and 'goal' in options:
            self.goal_position = np.array(options['goal'], dtype=np.float32)
        else:
            self.goal_position = self.default_goal.copy()
        
        # Reset counters
        self.step_count = 0
        self.prev_distance_to_goal = np.linalg.norm(self.agent_position - self.goal_position)
        
        # Generate random obstacles
        n_obstacles = options.get('n_obstacles', 7) if options else 7
        max_obstacle_size = options.get('max_obstacle_size', 1.0) if options else 1.0
        self._generate_obstacles(n_obstacles, max_obstacle_size)
        
        observation = self._get_observation()
        info = self._get_info()

       

        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Scale action from [-1, 1] to actual movement
        

        max_movement = 0.5
        movement = np.array(action, dtype=np.float32) * max_movement
        
        # Store previous position
        prev_position = self.agent_position.copy()
        
        # Calculate new position
        new_position = self.agent_position + movement
        new_position = np.clip(new_position, [0, 0], [self.x_max, self.y_max])
        
        # Check for collisions along the path
        collided = self._check_path_collision(prev_position, new_position)
        
        if collided:
            # Reject move and apply collision penalty
            collision_penalty = -50.0
            terminated = True  # End episode on collision
        else:
            # Accept move
            self.agent_position = new_position
            collision_penalty = 0.0
            terminated = False
        
        self.step_count += 1
        
        # Calculate distances and goal achievement
        distance_to_goal = np.linalg.norm(self.agent_position - self.goal_position)
        goal_reached = distance_to_goal < self.success_tolerance
        
        if goal_reached:
            terminated = True
        
        # Check if episode should be truncated
        truncated = self.step_count >= self.max_episode_steps and not terminated
        
        # Calculate reward
        reward = self._calculate_reward(
            distance_to_goal, goal_reached, collided, collision_penalty
        )
        
        # Update previous distance
        self.prev_distance_to_goal = distance_to_goal
        
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'goal_reached': goal_reached,
            'collision': collided,
            'distance_to_goal': distance_to_goal,
            'step_count': self.step_count
        })
        
        return observation, reward, terminated, truncated, info
    
    def _generate_obstacles(self, n_obstacles=10, max_size=4.0, min_size=3.5, clearance=2.0):
        """Generate random rectangular obstacles"""
        self.obstacles = []
        tries = 0
        max_tries = n_obstacles * 10
        
        while len(self.obstacles) < n_obstacles and tries < max_tries:
            # Random obstacle dimensions and position
            width = np.random.uniform(min_size, max_size)
            height = np.random.uniform(min_size, max_size)
            x = np.random.uniform(0, self.x_max - width)
            y = np.random.uniform(0, self.y_max - height)
            
            obstacle = [x, y, width, height]
            
            # Check if obstacle overlaps with start or goal (with clearance)
            if self._is_position_clear(self.agent_position, obstacle, clearance) and \
               self._is_position_clear(self.goal_position, obstacle, clearance):
                self.obstacles.append(obstacle)
            
            tries += 1
    
    def _is_position_clear(self, position, obstacle, clearance):
        """Check if a position is clear of an obstacle with clearance"""
        x, y, width, height = obstacle
        px, py = position
        
        return not (x - clearance < px < x + width + clearance and 
                   y - clearance < py < y + height + clearance)
    
    def _check_path_collision(self, start_pos, end_pos):
        """Check if the path from start_pos to end_pos intersects any obstacle"""
        for obstacle in self.obstacles:
            if self._line_intersects_rectangle(start_pos, end_pos, obstacle):
                return True
        return False
    
    def _line_intersects_rectangle(self, p1, p2, rect):
        """Check if line segment intersects rectangle"""
        x, y, w, h = rect
        
        # Rectangle corners
        rect_corners = [
            (x, y), (x + w, y), (x + w, y + h), (x, y + h)
        ]
        
        # Rectangle edges
        rect_edges = [
            (rect_corners[0], rect_corners[1]),
            (rect_corners[1], rect_corners[2]),
            (rect_corners[2], rect_corners[3]),
            (rect_corners[3], rect_corners[0])
        ]
        
        # Check line-line intersection with each edge
        for edge in rect_edges:
            if self._lines_intersect(p1, p2, edge[0], edge[1]):
                return True
        
        # Check if either endpoint is inside rectangle
        if self._point_in_rectangle(p1, rect) or self._point_in_rectangle(p2, rect):
            return True
        
        return False
    
    def _lines_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and 
                ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    def _point_in_rectangle(self, point, rect):
        """Check if point is inside rectangle"""
        x, y, w, h = rect
        px, py = point
        return x <= px <= x + w and y <= py <= y + h
    
    def _calculate_reward(self, distance_to_goal, goal_reached, collided, collision_penalty):
        """Calculate reward based on current state"""
        if goal_reached:
            return 50.0
        
        if collided:
            return collision_penalty
        
        # Base distance penalty
        distance_penalty = -distance_to_goal * 0.1
        
        # Step penalty to encourage efficiency
        step_penalty = -0.1

        # Progress reward
        progress_reward = 0.0
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - distance_to_goal
            progress_reward = progress * 5.0  # Encourage progress toward goal
        
        # Obstacle proximity penalty
        proximity_penalty = self._calculate_obstacle_proximity_penalty()
        
        # Safe progress bonus
        safe_progress_bonus = 0.0
        min_obstacle_distance = self._get_min_obstacle_distance()
        if progress_reward > 0 and min_obstacle_distance > 1.0:
            safe_progress_bonus = progress_reward * 1.0
        
        total_reward = (distance_penalty + step_penalty + progress_reward + 
                       proximity_penalty + safe_progress_bonus)
        
        return total_reward
    
    def _calculate_obstacle_proximity_penalty(self):
        """Calculate penalty based on proximity to obstacles"""
        min_distance = self._get_min_obstacle_distance()
        
        if min_distance < 2.0:
            # Exponential penalty for being close to obstacles
            return -5.0 * np.exp(-min_distance)
        
        return 0.0
    
    def _get_min_obstacle_distance(self):
        """Get minimum distance to any obstacle"""
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        ax, ay = self.agent_position
        
        for obstacle in self.obstacles:
            ox, oy, w, h = obstacle
            
            # Distance to rectangle edge
            dx = max(ox - ax, 0, ax - (ox + w))
            dy = max(oy - ay, 0, ay - (oy + h))
            distance = np.hypot(dx, dy)
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _get_observation(self):
        """Get current observation"""
        return np.concatenate([self.agent_position, self.goal_position]).astype(np.float32)
    
    def _get_info(self):
        """Get additional info"""
        return {
            'agent_position': self.agent_position.copy(),
            'goal_position': self.goal_position.copy(),
            'obstacles': self.obstacles.copy(),
            'min_obstacle_distance': self._get_min_obstacle_distance()
        }
    
    def render(self):
        """Render the environment - placeholder for now"""
        if self.render_mode == 'human':
            print(f"Agent at {self.agent_position}, Goal at {self.goal_position}, Step {self.step_count}")
    
    def close(self):
        """Close the environment"""
        pass


# Convenience function to create environment
def make_pathfinding_env(**kwargs):
    """Create a pathfinding environment with default parameters"""
    return PathfindingGymEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    env = PathfindingGymEnv()
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info keys: {list(info.keys())}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            if info.get('goal_reached', False):
                print("Goal reached!")
            elif info.get('collision', False):
                print("Collision occurred!")
            break
    
    env.close()
    print("Environment test completed!")