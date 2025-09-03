import gymnasium as gym
from maze_env import MazeEnv
from stable_baselines3 import TD3
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    env = MazeEnv()
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
    plt.figure(figsize=(7,7))
    plt.imshow(env.obstacles.T, origin='lower', cmap='gray_r', alpha=0.5)
    plt.plot(path[:,0], path[:,1], 'g.-', label='Agent Path')
    plt.scatter(path[0,0], path[0,1], c='blue', s=100, label='Start')
    plt.scatter(env.goal_pos[0], env.goal_pos[1], c='red', s=100, label='Goal')
    plt.title('TD3 Maze Pathfinding')
    plt.legend()
    plt.grid(True)
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/td3_maze_path.png", dpi=200)
    plt.show()
    print(f"Total reward: {sum(rewards):.2f}, Steps: {len(path)-1}")
