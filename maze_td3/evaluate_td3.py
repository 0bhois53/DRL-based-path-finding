import gymnasium as gym
from maze_env import MazeEnv, load_custom_points
from stable_baselines3 import TD3
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Load custom start/goal points if available
    start_pos, goal_pos = load_custom_points()
    env = MazeEnv(start_pos=start_pos, goal_pos=goal_pos)
    model = TD3.load("saved_models/td3_maze")
    obs, info = env.reset()
    path = [env.agent_pos.copy()]
    rewards = []
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        path.append(env.agent_pos.copy())
        rewards.append(reward)
        if done:
            break
    path = np.array(path)
    # Original path
    plt.figure(figsize=(7,7))
    plt.imshow(env.obstacles.T, origin='lower', cmap='gray_r', alpha=0.5)
    plt.plot(path[:,0], path[:,1], 'g.-', label='Agent Path')
    plt.scatter(path[0,0], path[0,1], c='blue', s=100, label='Start')
    plt.scatter(env.goal_pos[0], env.goal_pos[1], c='red', s=100, label='Goal')
        # Show custom start/goal if available
    if start_pos is not None and goal_pos is not None:
            plt.scatter(start_pos[0], start_pos[1], c='blue', s=120, label='Custom Start')
            plt.scatter(goal_pos[0], goal_pos[1], c='red', s=120, label='Custom Goal')
    plt.title('TD3 Maze Pathfinding (Original)')
    plt.legend()
    plt.grid(True)
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/td3_maze_path.png", dpi=200)
    plt.show()

    # Moving average smoothened path
    def moving_average_path(path, window=3):
        if len(path) < window:
            return path
        x = np.convolve(path[:,0], np.ones(window)/window, mode='same')
        y = np.convolve(path[:,1], np.ones(window)/window, mode='same')
        return np.stack([x, y], axis=1)

    smoothed_ma = moving_average_path(path)
    plt.figure(figsize=(7,7))
    plt.imshow(env.obstacles.T, origin='lower', cmap='gray_r', alpha=0.5)
    plt.plot(smoothed_ma[:,0], smoothed_ma[:,1], 'b-', label='Moving Average')
    plt.scatter(smoothed_ma[0,0], smoothed_ma[0,1], c='blue', s=100, label='Start')
    plt.scatter(env.goal_pos[0], env.goal_pos[1], c='red', s=100, label='Goal')
    plt.title('TD3 Maze Pathfinding (Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.savefig('visuals/td3_maze_path_moving_average.png', dpi=200)
    plt.show()

    # B-spline smoothened path
    def bspline_path(path, s=2):
        from scipy.interpolate import splprep, splev
        if len(path) < 4:
            return path
        tck, u = splprep([path[:,0], path[:,1]], s=s)
        unew = np.linspace(0, 1, max(100, len(path)*3))
        out = splev(unew, tck)
        return np.array(out).T

    try:
        smoothed_bspline = bspline_path(path)
        plt.figure(figsize=(7,7))
        plt.imshow(env.obstacles.T, origin='lower', cmap='gray_r', alpha=0.5)
        plt.plot(smoothed_bspline[:,0], smoothed_bspline[:,1], 'r-', label='B-spline')
        plt.scatter(smoothed_bspline[0,0], smoothed_bspline[0,1], c='blue', s=100, label='Start')
        plt.scatter(env.goal_pos[0], env.goal_pos[1], c='red', s=100, label='Goal')
        plt.title('TD3 Maze Pathfinding (B-spline)')
        plt.legend()
        plt.grid(True)
        plt.savefig('visuals/td3_maze_path_bspline.png', dpi=200)
        plt.show()
    except Exception as e:
        print(f'B-spline smoothing failed: {e}')
    print(f"Total reward: {sum(rewards):.2f}, Steps: {len(path)-1}")
    print('Path plots saved to visuals/td3_maze_path.png, td3_maze_path_moving_average.png, td3_maze_path_bspline.png')
