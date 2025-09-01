import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import splprep, splev
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch
import os
from pathfinding_gym_env import PathfindingGymEnv

def smooth_path_moving_average(path_x, path_y, window=5):
    """Smooth path using moving average"""
    if len(path_x) < window:
        return path_x, path_y
    x_smooth = np.convolve(path_x, np.ones(window)/window, mode='valid')
    y_smooth = np.convolve(path_y, np.ones(window)/window, mode='valid')
    # Pad to match original length
    pad = (len(path_x) - len(x_smooth)) // 2
    x_smooth = np.pad(x_smooth, (pad, len(path_x)-len(x_smooth)-pad), mode='edge')
    y_smooth = np.pad(y_smooth, (pad, len(path_y)-len(y_smooth)-pad), mode='edge')
    return x_smooth, y_smooth

def smooth_path_bspline(path_x, path_y, s=2):
    """Smooth path using B-spline"""
    if len(path_x) < 4:
        return path_x, path_y
    try:
        tck, u = splprep([path_x, path_y], s=s)
        unew = np.linspace(0, 1, len(path_x))
        out = splev(unew, tck)
        return out[0], out[1]
    except:
        # Fallback to original path if B-spline fails
        return path_x, path_y

def load_custom_points(file_path="selected_points.txt"):
    """Load custom start and goal points from file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    # Parse start point (first line)
                    start_coords = lines[0].split(',')
                    start_point = (float(start_coords[0]) / 10.0, float(start_coords[1]) / 10.0)
                    
                    # Parse goal point (second line)
                    goal_coords = lines[1].split(',')
                    goal_point = (float(goal_coords[0]) / 10.0, float(goal_coords[1]) / 10.0)
                    
                    print(f"Loaded custom points from {file_path}:")
                    print(f"  Start: {start_point} (original: {start_coords[0]},{start_coords[1]})")
                    print(f"  Goal: {goal_point} (original: {goal_coords[0]},{goal_coords[1]})")
                    
                    return start_point, goal_point
                else:
                    print(f"Warning: {file_path} does not contain enough points. Using defaults.")
                    return None, None
        else:
            print(f"File {file_path} not found. Using default start and goal points.")
            return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Using default points.")
        return None, None

def create_env(use_custom_points=True):
    """Create the pathfinding environment with optional custom points"""
    # Default points
    default_start = (1.0, 1.0)
    default_goal = (9.0, 9.0)
    
    # Try to load custom points
    if use_custom_points:
        custom_start, custom_goal = load_custom_points()
        start_point = custom_start if custom_start is not None else default_start
        goal_point = custom_goal if custom_goal is not None else default_goal
    else:
        start_point = default_start
        goal_point = default_goal
    
    return PathfindingGymEnv(
        x_max=20.0,
        y_max=20.0,
        default_start=start_point,
        default_goal=goal_point,
        max_episode_steps=300,
        success_tolerance=1.5
    )

def train_ddpg(total_timesteps=100000, save_path="./standalone_ddpg_models", use_custom_points=True):
    """Train DDPG agent using the standalone environment"""
    
    # Create environment
    env = make_vec_env(lambda: create_env(use_custom_points), n_envs=1)
    
    # Create action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    
    # Create DDPG model with improved hyperparameters
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=3e-4,  # Increased learning rate
        buffer_size=100000,
        learning_starts=5000,  # Reduced to start learning earlier
        batch_size=64,  # Reduced batch size
        tau=0.005,  # Slower target network updates for stability
        gamma=0.99,  # Reduced gamma for more immediate reward focus
        train_freq=1,
        gradient_steps=1,  # Reduced gradient steps
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network for better learning
        verbose=1,
        device="cuda",  # Force GPU usage
        tensorboard_log="./standalone_ddpg_tensorboard/"
    )
    
   
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix="ddpg_checkpoint"
    )
    
    # Train the model
    print("Starting DDPG training...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback= [checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(save_path, "ddpg_final"))
    print(f"Training completed! Model saved to {save_path}")
    
    return model



def evaluate_agent(model_path, algorithm='ddpg', n_episodes=20, render=False, use_custom_points=True):
    """Evaluate the trained agent"""
    
    # Load the model
    if algorithm.lower() == 'ddpg':
        model = DDPG.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create environment with custom points if requested
    env = create_env(use_custom_points=use_custom_points)
    
    success_count = 0
    collision_count = 0
    episode_rewards = []
    episode_lengths = []
    paths = []
    
    print(f"\nEvaluating {algorithm.upper()} agent...")
    if use_custom_points:
        print(f"Using start: {env.default_start}, goal: {env.default_goal}")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        path_x, path_y = [obs[0]], [obs[1]]
        
        while True:
            # Get action from trained model (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            path_x.append(obs[0])
            path_y.append(obs[1])
            
            if terminated or truncated:
                if info.get('goal_reached', False):
                    success_count += 1
                    print(f"Episode {episode + 1}: SUCCESS in {steps} steps, reward: {episode_reward:.2f}")
                elif info.get('collision', False):
                    collision_count += 1
                    print(f"Episode {episode + 1}: COLLISION in {steps} steps, reward: {episode_reward:.2f}")
                else:
                    print(f"Episode {episode + 1}: TIMEOUT in {steps} steps, reward: {episode_reward:.2f}")
                break
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        paths.append((path_x, path_y))
    
    # Print statistics
    success_rate = success_count / n_episodes
    collision_rate = collision_count / n_episodes
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{algorithm.upper()} Evaluation Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Collision Rate: {collision_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    
    # Visualize best episode (highest reward)
    if paths and not render:
        best_episode_idx = np.argmax(episode_rewards)
        best_path = paths[best_episode_idx]
        best_reward = episode_rewards[best_episode_idx]
        visualize_path(env, best_path, 
                      f"{algorithm.upper()} Best Path (Reward: {best_reward:.2f}, Success Rate: {success_rate:.2%})")
    
    env.close()
    
    # Visualize episode statistics
    visualize_episode_statistics(episode_rewards, episode_lengths, algorithm.upper())
    
    return success_rate, collision_rate, avg_reward, avg_length, paths

def visualize_path(env, path, title="Agent Path"):
    """Visualize the agent's path"""
    path_x, path_y = path
    # Smooth paths
    ma_x, ma_y = smooth_path_moving_average(path_x, path_y, window=5)
    bs_x, bs_y = smooth_path_bspline(path_x, path_y, s=2)

    plt.figure(figsize=(12, 10))

    # Plot obstacles
    for obstacle in env.obstacles:
        x, y, w, h = obstacle
        rect = Rectangle((x, y), w, h, facecolor='blue', alpha=0.7,
                        edgecolor='darkblue', linewidth=2)
        plt.gca().add_patch(rect)

    # Plot original path
    for i in range(len(path_x) - 1):
        alpha = 0.3 + 0.7 * (i / len(path_x))
        plt.plot(path_x[i:i+2], path_y[i:i+2], 'r-', alpha=alpha, linewidth=2, label='Original Path' if i==0 else None)

    # Plot moving average smoothed path
    plt.plot(ma_x, ma_y, 'g--', linewidth=2, label='Moving Average Smoothed')

    # Plot B-spline smoothed path
    plt.plot(bs_x, bs_y, 'b-', linewidth=2, label='B-Spline Smoothed')

    # Plot start and goal
    plt.scatter(path_x[0], path_y[0], c='green', s=300,
               label='Start', edgecolor='darkgreen', linewidth=3, zorder=5)
    plt.scatter(env.goal_position[0], env.goal_position[1], c='red', s=300,
               label='Goal', edgecolor='darkred', linewidth=3, zorder=5)

    # Plot final position
    plt.scatter(path_x[-1], path_y[-1], c='orange', s=200,
               label='Final Position', linewidth=4, zorder=5)

    # Plot success tolerance circle
    circle = Circle(env.goal_position, env.success_tolerance,
                   fill=False, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.gca().add_patch(circle)

    plt.xlim(0, env.x_max)
    plt.ylim(0, env.y_max)
    plt.xlabel('X Position', fontsize=14)
    plt.ylabel('Y Position', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def visualize_episode_statistics(episode_rewards, episode_lengths, algorithm_name="Agent"):
    """Visualize accumulated rewards and steps taken per episode"""
    episodes = range(1, len(episode_rewards) + 1)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards
    ax1.plot(episodes, episode_rewards, 'b-', marker='o', markersize=4, linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Cumulative Reward', fontsize=12)
    ax1.set_title(f'{algorithm_name} - Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Reward')
    
    # Add rolling average for rewards
    if len(episode_rewards) >= 5:
        window = min(5, len(episode_rewards))
        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        rolling_episodes = episodes[window-1:]
        ax1.plot(rolling_episodes, rolling_avg, 'r-', linewidth=3, alpha=0.7, label=f'Rolling Average ({window} episodes)')
    
    ax1.legend()
    
    # Plot episode lengths (steps)
    ax2.plot(episodes, episode_lengths, 'g-', marker='s', markersize=4, linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps Taken', fontsize=12)
    ax2.set_title(f'{algorithm_name} - Steps per Episode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add rolling average for steps
    if len(episode_lengths) >= 5:
        window = min(5, len(episode_lengths))
        rolling_avg_steps = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        rolling_episodes = episodes[window-1:]
        ax2.plot(rolling_episodes, rolling_avg_steps, 'orange', linewidth=3, alpha=0.7, label=f'Rolling Average ({window} episodes)')
    
    ax2.legend()
    
    # Add statistics text
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_lengths)
    best_reward = np.max(episode_rewards)
    worst_reward = np.min(episode_rewards)
    
    stats_text = f'Statistics:\nAvg Reward: {avg_reward:.2f}\nBest Reward: {best_reward:.2f}\nWorst Reward: {worst_reward:.2f}\nAvg Steps: {avg_steps:.1f}'
    
    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

def test_custom_points():
    """Test the environment with custom points to verify they work correctly"""
    print("\n" + "="*50)
    print("TESTING CUSTOM POINTS SYSTEM")
    print("="*50)
    
    # Test with custom points
    print("\n1. Testing with custom points from selected_points.txt:")
    env_custom = create_env(use_custom_points=True)
    obs_custom, info_custom = env_custom.reset()
    print(f"   Start: {env_custom.default_start}")
    print(f"   Goal: {env_custom.default_goal}")
    print(f"   Distance: {np.linalg.norm(np.array(env_custom.default_goal) - np.array(env_custom.default_start)):.2f}")
    env_custom.close()
    
   


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directories
    os.makedirs("./standalone_ddpg_models", exist_ok=True)
    
    # Test custom points system
    test_custom_points()
    
    # Quick test of the environment
    print("Testing environment...")
    print("\n" + "="*30)
    print("TESTING WITH CUSTOM POINTS")
    print("="*30)
    env = create_env(use_custom_points=True)
    obs, info = env.reset()
    print(f"Environment test successful!")
    print(f"Start position: {env.default_start}")
    print(f"Goal position: {env.default_goal}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Obstacles: {len(info['obstacles'])}")
    env.close()
    
    # Ask user if they want to use custom points for training
    print("\n" + "="*50)
    print("TRAINING OPTIONS")
    print("="*50)
    use_custom = input("Use custom start/goal points from selected_points.txt for training? (y/n): ").lower().strip()
    use_custom_points = use_custom in ['y', 'yes', '1', 'true']
    
    if use_custom_points:
        print("Training with custom start/goal points...")
    else:
        print("Training with default points (1,1) -> (9,9)...")
    
    # Train DDPG
    print("\n" + "="*50)
    print("TRAINING DDPG")
    print("="*50)
    model = train_ddpg(total_timesteps=100000, use_custom_points=use_custom_points)
    
    # Evaluate DDPG with both custom and default points
    print("\n" + "="*50)
    print("EVALUATING DDPG WITH CUSTOM POINTS")
    print("="*50)
    evaluate_agent("./standalone_ddpg_models/ddpg_final", "ddpg", n_episodes=10, render=False, use_custom_points=True)
    
    
