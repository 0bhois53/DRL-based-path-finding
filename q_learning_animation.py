"""
Q-Learning Animation Module
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from environment import obstacle_width

# Path smoothing functions
def moving_average_path(path, window_size=3):
    if len(path) < window_size:
        return path
    smoothed = []
    for i in range(len(path)):
        start = max(0, i - window_size // 2)
        end = min(len(path), i + window_size // 2 + 1)
        avg_x = np.mean([p[0] for p in path[start:end]])
        avg_y = np.mean([p[1] for p in path[start:end]])
        smoothed.append((avg_x, avg_y))
    return smoothed

def b_spline_path(path, degree=3, num_points=100):
    if len(path) <= degree:
        return path
    from scipy.interpolate import splprep, splev
    x, y = zip(*path)
    tck, u = splprep([x, y], s=0, k=degree)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return list(zip(x_new, y_new))

def shortcut_path(path):
    # Remove unnecessary zig-zags by shortcutting straight segments
    if len(path) < 3:
        return path
    shortcut = [path[0]]
    for i in range(1, len(path)-1):
        prev, curr, nxt = path[i-1], path[i], path[i+1]
        # If curr is in line with prev and nxt, skip curr
        if (nxt[0]-prev[0])*(curr[1]-prev[1]) == (curr[0]-prev[0])*(nxt[1]-prev[1]):
            continue
        shortcut.append(curr)
    shortcut.append(path[-1])
    return shortcut


def animate_evaluation(agent, env, starting_position, target_position, 
                      model_path='saved_models_Q_learning/q_learning_final_model.pkl', 
                      save_gif=True, interval=500, use_randomized_env=False, episode_num=None):
    """Create an animated visualization of the agent moving through the environment"""
    env_type = "randomized" if use_randomized_env else "training"
    print(f"\nCreating animated evaluation on {env_type} environment from {model_path}...")
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using current agent state.")
    except Exception as e:
        print(f"Error loading model: {e}. Using current agent state.")
    
    # Create randomized environment if requested
    if use_randomized_env:
        from environment import Environment
        eval_env = Environment(starting_position, target_position, 100, 100, env.num_actions)
        print(f"Created randomized environment with {len(eval_env.Obstacle_x)} obstacles")
    else:
        eval_env = env
    
    # Run one episode to collect the path
    state = eval_env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    # Store all positions and states for animation
    positions = [(eval_env.vector_agentState[0], eval_env.vector_agentState[1])]
    actions_taken = []
    rewards_received = []
    
    print("Running episode for animation...")
    while not done and steps < eval_env.max_episode_steps:
        # Use greedy action for evaluation
        action = agent.get_action(str(state), 0.0)
        next_state, next_state_flag, reward, done, _ = eval_env.step(action)
        
        positions.append((eval_env.vector_agentState[0], eval_env.vector_agentState[1]))
        actions_taken.append(action)
        rewards_received.append(reward)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    success = next_state_flag == 'goal'
    print(f"Episode completed: {'SUCCESS' if success else 'FAILED'} - Reward: {episode_reward:.2f}, Steps: {steps}")

    # Interactive menu for path smoothing
    print("\nChoose path smoothing method:")
    print("1. No smoothing (raw path)")
    print("2. Moving Average")
    print("3. B-Spline Interpolation")
    print("4. Shortcutting (remove zig-zags)")
    print("5. Exit menu")
    choice = input("Enter choice (1-5): ").strip()
    if choice == '2':
        positions = moving_average_path(positions)
        print("Applied Moving Average smoothing.")
    elif choice == '3':
        try:
            positions = b_spline_path(positions)
            print("Applied B-Spline smoothing.")
        except Exception as e:
            print(f"B-Spline smoothing failed: {e}. Using raw path.")
    elif choice == '4':
        positions = shortcut_path(positions)
        print("Applied Shortcutting smoothing.")
    elif choice == '5':  
        print("Exiting menu")  
    else:
        print("Invalid choice.")
    
    # Create the animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define action names for display
    action_names = ['Up', 'Down', 'Left', 'Right', 'Up-Left', 'Up-Right', 'Down-Left', 'Down-Right']
    
    def animate_frame(frame):
        ax1.clear()
        ax2.clear()
        
        # Main environment plot
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('x (m)', size=12)
        ax1.set_ylabel('y (m)', size=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot obstacles (use eval_env for correct obstacle positions)
        for i in range(len(eval_env.Obstacle_x)):
            rectangle = mpatches.Rectangle((10 * (eval_env.Obstacle_x[i] - 0.5), 10 * (10 - eval_env.Obstacle_y[i] - 0.5)), 
                                            obstacle_width, obstacle_width, fc='blue', ec="blue", alpha=0.8)
            ax1.add_patch(rectangle)
        
        # Plot start and goal
        ax1.scatter(starting_position[0], starting_position[1], c='green', s=150, marker='s', 
                   label="Start", edgecolors='black', linewidth=2)
        ax1.scatter(target_position[0], target_position[1], c='red', s=150, marker='s', 
                   label="Goal", edgecolors='black', linewidth=2)
        
        # Plot path up to current frame
        if frame > 0:
            path_x = [pos[0] for pos in positions[:frame+1]]
            path_y = [pos[1] for pos in positions[:frame+1]]
            ax1.plot(path_x, path_y, 'orange', linewidth=3, alpha=0.7, label='Path')
            
            # Plot previous positions as small dots
            if len(path_x) > 1:
                ax1.scatter(path_x[:-1], path_y[:-1], c='orange', s=30, alpha=0.6)
        
        # Plot current agent position
        current_pos = positions[frame]
        ax1.scatter(current_pos[0], current_pos[1], c='purple', s=200, marker='o', 
                   label='Agent', edgecolors='white', linewidth=3, zorder=10)
        
        # Add direction arrow if not the first frame
        if frame > 0 and frame < len(positions):
            prev_pos = positions[frame-1]
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            if dx != 0 or dy != 0:  # Only draw arrow if there's movement
                ax1.arrow(prev_pos[0], prev_pos[1], dx*0.7, dy*0.7, 
                         head_width=2, head_length=2, fc='purple', ec='purple', alpha=0.8)
        
        ax1.set_title(f'Q-Learning Agent Animation - {env_type.title()} Environment\n'
                     f'Step {frame}/{len(positions)-1}'
                     f'{" - Episode " + str(episode_num) if episode_num is not None else ""}\n'
                     f'{"SUCCESS!" if success and frame == len(positions)-1 else ""}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        
        # Information panel
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Display current information
        info_text = f"Environment: {env_type.title()}\n"
        if use_randomized_env:
            info_text += f"Obstacles: {len(eval_env.Obstacle_x)}\n"
        info_text += f"Step: {frame}/{len(positions)-1}\n\n"
        info_text += f"Current Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})\n\n"
        
        if frame > 0 and frame-1 < len(actions_taken):
            action_idx = actions_taken[frame-1]
            action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
            info_text += f"Last Action: {action_name}\n"
            info_text += f"Last Reward: {rewards_received[frame-1]:.2f}\n\n"
        
        info_text += f"Total Reward: {sum(rewards_received[:frame]):.2f}\n"
        info_text += f"Episode Status: {'Complete' if frame == len(positions)-1 else 'In Progress'}\n"
        
        if frame == len(positions)-1:
            info_text += f"Final Result: {'SUCCESS' if success else 'FAILED'}\n"
            
        # Calculate distance to goal
        goal_dist = np.sqrt((current_pos[0] - target_position[0])**2 + 
                           (current_pos[1] - target_position[1])**2)
        info_text += f"Distance to Goal: {goal_dist:.1f}"
        
        ax2.text(0.5, 5, info_text, fontsize=12, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Add progress bar
        progress = frame / (len(positions) - 1) if len(positions) > 1 else 0
        ax2.barh(1, progress * 8, height=0.5, left=1, color='green', alpha=0.7)
        ax2.barh(1, (1-progress) * 8, height=0.5, left=1+progress*8, color='lightgray', alpha=0.7)
        ax2.text(5, 1, f'Progress: {progress*100:.1f}%', ha='center', va='center', fontweight='bold')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(positions), 
                                  interval=interval, repeat=True, blit=False)
    
    # Save as GIF if requested
    if save_gif:
        print("Saving animation as GIF... (this may take a moment)")
        try:
            gif_filename = f'Q_learning_animated_evaluation_{env_type}'
            if episode_num is not None:
                gif_filename += f'_episode_{episode_num}'
            gif_filename += '.gif'
            
            anim.save(gif_filename, writer='pillow', fps=2)
            print(f"Animation saved as '{gif_filename}'")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Trying to save as MP4...")
            try:
                mp4_filename = gif_filename.replace('.gif', '.mp4')
                anim.save(mp4_filename, writer='ffmpeg', fps=2)
                print(f"Animation saved as '{mp4_filename}'")
            except Exception as e2:
                print(f"Error saving MP4: {e2}")
                print("Animation will only be displayed, not saved.")
    
    plt.tight_layout()
    plt.show()
    
    return anim, {'success': success, 'steps': steps, 'reward': episode_reward}


def run_animated_evaluation(agent, env, starting_position, target_position, speed='normal'):
    """Convenience function to run animated evaluation with different speeds"""
    interval_map = {
        'slow': 1000,
        'normal': 600,
        'fast': 300,
        'very_fast': 150
    }
    
    interval = interval_map.get(speed, 600)
    
    return animate_evaluation(agent, env, starting_position, target_position,
                            save_gif=True, interval=interval)


def animate_randomized_evaluations(agent, env, starting_position, target_position, 
                                  num_episodes=3, model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                  save_gifs=True, interval=500):
    """Create animations for multiple randomized environment evaluations"""
    print(f"\nCreating {num_episodes} randomized environment animations...")
    
    results = []
    animations = []
    
    for episode in range(num_episodes):
        print(f"\n--- Animating Randomized Episode {episode + 1}/{num_episodes} ---")
        
        try:
            anim, result = animate_evaluation(
                agent=agent,
                env=env,
                starting_position=starting_position,
                target_position=target_position,
                model_path=model_path,
                save_gif=save_gifs,
                interval=interval,
                use_randomized_env=True,
                episode_num=episode + 1
            )
            
            animations.append(anim)
            results.append(result)
            
            print(f"Episode {episode + 1}: {'SUCCESS' if result['success'] else 'FAILED'} "
                  f"- Steps: {result['steps']}, Reward: {result['reward']:.2f}")
            
        except Exception as e:
            print(f"Error creating animation for episode {episode + 1}: {e}")
            continue
    
    # Summary
    successful_episodes = sum(1 for r in results if r['success'])
    success_rate = (successful_episodes / len(results)) * 100 if results else 0
    
    print(f"\n=== RANDOMIZED ANIMATION SUMMARY ===")
    print(f"Episodes Animated: {len(results)}")
    print(f"Success Rate: {success_rate:.1f}% ({successful_episodes}/{len(results)})")
    
    if results:
        avg_steps = np.mean([r['steps'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Reward: {avg_reward:.2f}")
    
    return animations, results


def animate_comparison(agent, env, starting_position, target_position,
                      model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                      save_gifs=True, interval=500):
    """Create side-by-side animations comparing training vs randomized environment"""
    print("\nCreating comparison animations (Training vs Randomized)...")
    
    # Animate training environment
    print("\n--- Training Environment Animation ---")
    training_anim, training_result = animate_evaluation(
        agent=agent,
        env=env,
        starting_position=starting_position,
        target_position=target_position,
        model_path=model_path,
        save_gif=save_gifs,
        interval=interval,
        use_randomized_env=False,
        episode_num=None
    )
    
    # Animate randomized environment
    print("\n--- Randomized Environment Animation ---")
    randomized_anim, randomized_result = animate_evaluation(
        agent=agent,
        env=env,
        starting_position=starting_position,
        target_position=target_position,
        model_path=model_path,
        save_gif=save_gifs,
        interval=interval,
        use_randomized_env=True,
        episode_num=1
    )
    
    # Print comparison
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Training Environment: {'SUCCESS' if training_result['success'] else 'FAILED'} "
          f"- Steps: {training_result['steps']}, Reward: {training_result['reward']:.2f}")
    print(f"Randomized Environment: {'SUCCESS' if randomized_result['success'] else 'FAILED'} "
          f"- Steps: {randomized_result['steps']}, Reward: {randomized_result['reward']:.2f}")
    
    return {
        'training': {'animation': training_anim, 'result': training_result},
        'randomized': {'animation': randomized_anim, 'result': randomized_result}
    }


if __name__ == "__main__":
    print("This is an animation module for Q-learning evaluation.")
    print("Import this module and use the animation functions.")
    print("\nAvailable functions:")
    print("1. animate_evaluation() - Single episode animation")
    print("2. animate_randomized_evaluations() - Multiple randomized environments")
    print("3. animate_comparison() - Compare training vs randomized")
    print("4. run_animated_evaluation() - Convenience function with speed options")
    print("\nExample usage:")
    print("  from q_learning_animation import animate_evaluation, animate_randomized_evaluations")
    print("  # Single episode on training environment")
    print("  anim, results = animate_evaluation(agent, env, start_pos, target_pos, use_randomized_env=False)")
    print("  # Single episode on randomized environment") 
    print("  anim, results = animate_evaluation(agent, env, start_pos, target_pos, use_randomized_env=True)")
    print("  # Multiple randomized environments")
    print("  anims, results = animate_randomized_evaluations(agent, env, start_pos, target_pos, num_episodes=3)")
