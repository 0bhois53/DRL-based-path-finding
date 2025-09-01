import numpy as np
from pathfinding_gym_env import PathfindingGymEnv

def test_standalone_environment():
    """Test the standalone gymnasium environment"""
    
    print("Testing Standalone Pathfinding Environment")
    print("=" * 50)
    
    # Create environment
    env = PathfindingGymEnv(
        x_max=10.0,
        y_max=10.0,
        default_start=(1.0, 1.0),
        default_goal=(9.0, 9.0),
        max_episode_steps=300,
        success_tolerance=0.5
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nReset successful!")
    print(f"Initial observation: {obs}")
    print(f"Start: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"Goal: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"Distance to goal: {np.linalg.norm(obs[:2] - obs[2:]):.2f}")
    print(f"Number of obstacles: {len(info['obstacles'])}")
    print(f"Min obstacle distance: {info['min_obstacle_distance']:.2f}")
    
    # Test a few steps
    total_reward = 0
    print(f"\nTesting steps:")
    
    for step in range(10):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}:")
        print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"  Position: ({obs[0]:.2f}, {obs[1]:.2f})")
        print(f"  Reward: {reward:.2f}")
        print(f"  Distance to goal: {info['distance_to_goal']:.2f}")
        print(f"  Min obstacle dist: {info['min_obstacle_distance']:.2f}")
        
        if terminated:
            if info['goal_reached']:
                print("  >>> GOAL REACHED! <<<")
            elif info['collision']:
                print("  >>> COLLISION! <<<")
            break
        elif truncated:
            print("  >>> EPISODE TRUNCATED! <<<")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Episode ended after {step+1} steps")
    
    # Test multiple episodes
    print(f"\nTesting multiple episodes:")
    successes = 0
    collisions = 0
    timeouts = 0
    
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Limit steps for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                if info.get('goal_reached', False):
                    successes += 1
                    result = "SUCCESS"
                elif info.get('collision', False):
                    collisions += 1
                    result = "COLLISION"
                else:
                    timeouts += 1
                    result = "TIMEOUT"
                
                print(f"Episode {episode+1}: {result} in {step+1} steps, reward: {episode_reward:.2f}")
                break
    
    print(f"\nResults from 5 episodes:")
    print(f"Successes: {successes}")
    print(f"Collisions: {collisions}")
    print(f"Timeouts: {timeouts}")
    
    env.close()
    print("\nEnvironment test completed successfully!")

def test_reward_components():
    """Test different reward components"""
    
    print("\nTesting Reward Components")
    print("=" * 30)
    
    env = PathfindingGymEnv()
    
    # Test goal reaching
    obs, info = env.reset()
    
    # Move close to goal
    env.agent_position = env.goal_position - np.array([0.3, 0.3])
    obs = env._get_observation()
    
    # Move toward goal
    action = np.array([1.0, 1.0])  # Move toward goal
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Moving toward goal:")
    print(f"  Reward: {reward:.2f}")
    print(f"  Terminated: {terminated}")
    print(f"  Goal reached: {info.get('goal_reached', False)}")
    
    env.close()

if __name__ == "__main__":
    test_standalone_environment()
    test_reward_components()
