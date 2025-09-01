"""
DQN Animation Module
Provides animated visualization for DQN agent evaluation
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    if len(path) < 3:
        return path
    shortcut = [path[0]]
    for i in range(1, len(path)-1):
        prev, curr, nxt = path[i-1], path[i], path[i+1]
        if (nxt[0]-prev[0])*(curr[1]-prev[1]) == (curr[0]-prev[0])*(nxt[1]-prev[1]):
            continue
        shortcut.append(curr)
    shortcut.append(path[-1])
    return shortcut


def animate_evaluation(agent, env, starting_position, target_position, 
                      model_path='saved_models/dqn_final_model.pth', 
                      save_gif=True, interval=500):
    """Create an animated visualization of the DQN agent moving through the environment"""
    print(f"\nCreating animated evaluation from {model_path}...")
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using current agent state.")
    except Exception as e:
        print(f"Error loading model: {e}. Using current agent state.")
    
    # Run one episode to collect the path
    state = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    positions = [(env.vector_agentState[0], env.vector_agentState[1])]
    actions_taken = []
    rewards_received = []
    q_values_history = []
    
    print("Running episode for animation...")
    while not done and steps < env.max_episode_steps:
        # Use greedy action for evaluation (epsilon=0)
        action = agent.get_action(state, epsilon=0.0)
        
        # Get Q-values for visualization (if available)
        try:
            # Get Q-values from the DQN for the current state
            state_tensor = agent.preprocess_state(state) if hasattr(agent, 'preprocess_state') else state
            q_vals = agent.policy_net(state_tensor).detach().cpu().numpy().flatten()
            q_values_history.append(q_vals)
        except:
            q_values_history.append(None)
        
        next_state, next_state_flag, reward, done, _ = env.step(action)
        
        positions.append((env.vector_agentState[0], env.vector_agentState[1]))
        actions_taken.append(action)
        rewards_received.append(reward)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    success = next_state_flag == 'goal'
    print(f"Episode completed: {'SUCCESS' if success else 'FAILED'} - Reward: {episode_reward:.2f}, Steps: {steps}")

    # Interactive menu for path smoothing
    while True:
        print("\nChoose path smoothing method:")
        print("1. No smoothing (raw path)")
        print("2. Moving Average")
        print("3. B-Spline Interpolation")
        print("4. Shortcutting (remove zig-zags)")
        print("5. Exit menu")
        choice = input("Enter choice (1-5): ").strip()
        if choice == '2':
            smoothed_positions = moving_average_path(positions)
            print("Applied Moving Average smoothing.")
        elif choice == '3':
            try:
                smoothed_positions = b_spline_path(positions)
                print("Applied B-Spline smoothing.")
            except Exception as e:
                print(f"B-Spline smoothing failed: {e}. Using raw path.")
                smoothed_positions = positions
        elif choice == '4':
            smoothed_positions = shortcut_path(positions)
            print("Applied Shortcutting smoothing.")
        elif choice == '1':
            smoothed_positions = positions
            print("Using raw path (no smoothing).")
        elif choice == '5':
            print("Exiting menu.")
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        # Create the animation with the selected path
        fig = plt.figure(figsize=(20, 8))
        ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax3 = plt.subplot2grid((2, 3), (1, 2))
        action_names = ['Up', 'Down', 'Left', 'Right', 'Up-Left', 'Up-Right', 'Down-Left', 'Down-Right']
        def animate_frame(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            ax1.set_xlabel('x (m)', size=12)
            ax1.set_ylabel('y (m)', size=12)
            ax1.grid(True, alpha=0.3)
            for i in range(len(env.Obstacle_x)):
                rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
                                    obstacle_width, obstacle_width, fc='blue', ec="blue", alpha=0.8)
                ax1.add_patch(rectangle)
            ax1.scatter(starting_position[0], starting_position[1], c='green', s=150, marker='s', 
                    label="Start", edgecolors='black', linewidth=2)
            ax1.scatter(target_position[0], target_position[1], c='red', s=150, marker='s', 
                    label="Goal", edgecolors='black', linewidth=2)
            if frame > 0:
                path_x = [pos[0] for pos in smoothed_positions[:frame+1]]
                path_y = [pos[1] for pos in smoothed_positions[:frame+1]]
                ax1.plot(path_x, path_y, 'orange', linewidth=3, alpha=0.7, label='Path')
                if len(path_x) > 1:
                    ax1.scatter(path_x[:-1], path_y[:-1], c='orange', s=30, alpha=0.6)
            current_pos = smoothed_positions[frame]
            ax1.scatter(current_pos[0], current_pos[1], c='purple', s=200, marker='o', 
                    label='DQN Agent', edgecolors='white', linewidth=3, zorder=10)
            if frame > 0 and frame < len(smoothed_positions):
                prev_pos = smoothed_positions[frame-1]
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                if dx != 0 or dy != 0:
                    ax1.arrow(prev_pos[0], prev_pos[1], dx*0.7, dy*0.7, 
                            head_width=2, head_length=2, fc='purple', ec='purple', alpha=0.8)
            ax1.set_title(f'DQN Agent Animation - Step {frame}/{len(smoothed_positions)-1}\n'
                        f'{"SUCCESS!" if success and frame == len(smoothed_positions)-1 else ""}', 
                        fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax2.set_xlim(0, 10)
            ax2.set_ylim(0, 10)
            ax2.axis('off')
            info_text = f"Step: {frame}/{len(smoothed_positions)-1}\n\n"
            info_text += f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})\n\n"
            if frame > 0 and frame-1 < len(actions_taken):
                action_idx = actions_taken[frame-1]
                action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
                info_text += f"Last Action: {action_name}\n"
                info_text += f"Last Reward: {rewards_received[frame-1]:.2f}\n\n"
            info_text += f"Total Reward: {sum(rewards_received[:frame]):.2f}\n"
            info_text += f"Status: {'Complete' if frame == len(smoothed_positions)-1 else 'Running'}\n"
            if frame == len(smoothed_positions)-1:
                info_text += f"Result: {'SUCCESS' if success else 'FAILED'}\n"
            goal_dist = np.sqrt((current_pos[0] - target_position[0])**2 + 
                            (current_pos[1] - target_position[1])**2)
            info_text += f"Distance to Goal: {goal_dist:.1f}"
            ax2.text(0.5, 5, info_text, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            progress = frame / (len(smoothed_positions) - 1) if len(smoothed_positions) > 1 else 0
            ax2.barh(1, progress * 8, height=0.4, left=1, color='green', alpha=0.7)
            ax2.barh(1, (1-progress) * 8, height=0.4, left=1+progress*8, color='lightgray', alpha=0.7)
            ax2.text(5, 1, f'Progress: {progress*100:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10)
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, 10)
            ax3.axis('off')
            if frame < len(q_values_history) and q_values_history[frame] is not None:
                q_vals = q_values_history[frame]
                q_text = "Q-Values:\n\n"
                max_q_idx = np.argmax(q_vals)
                for i, (action_name, q_val) in enumerate(zip(action_names[:len(q_vals)], q_vals)):
                    color = "🟢" if i == max_q_idx else "⚪"
                    q_text += f"{color} {action_name}: {q_val:.3f}\n"
                ax3.text(0.5, 7, q_text, fontsize=9, ha='left', va='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                if len(q_vals) > 0:
                    y_pos = np.arange(len(q_vals))
                    colors = ['green' if i == max_q_idx else 'lightblue' for i in range(len(q_vals))]
                    bar_height = 2.5 / len(q_vals) if len(q_vals) > 0 else 0.3
                    for i, q_val in enumerate(q_vals):
                        bar_width = (q_val - np.min(q_vals)) / (np.max(q_vals) - np.min(q_vals) + 1e-6) * 4
                        ax3.barh(4 - i * bar_height, bar_width, height=bar_height*0.8, 
                                left=5, color=colors[i], alpha=0.7)
            else:
                ax3.text(0.5, 5, "Q-Values:\nNot available", fontsize=11, ha='left', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax3.set_title("Neural Network Output", fontsize=10, fontweight='bold')
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(smoothed_positions), 
                                    interval=interval, repeat=True, blit=False)
        if save_gif:
            print("Saving animation as GIF... (this may take a moment)")
            try:
                anim.save('DQN_animated_evaluation.gif', writer='pillow', fps=2)
                print("Animation saved as 'DQN_animated_evaluation.gif'")
            except Exception as e:
                print(f"Error saving GIF: {e}")
                print("Trying to save as MP4...")
                try:
                    anim.save('DQN_animated_evaluation.mp4', writer='ffmpeg', fps=2)
                    print("Animation saved as 'DQN_animated_evaluation.mp4'")
                except Exception as e2:
                    print(f"Error saving MP4: {e2}")
                    print("Animation will only be displayed, not saved.")
        plt.tight_layout()
        plt.show()
    return anim, {'success': success, 'steps': steps, 'reward': episode_reward, 'q_values': q_values_history}


def animate_training_episode(agent, env, starting_position, target_position, episode_num=0, 
                           save_gif=False, interval=300):
    """Create an animated visualization of a training episode (with exploration)"""
    print(f"\nCreating training episode animation (Episode {episode_num})...")
    
    # Run one episode to collect the path (with exploration)
    state = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    epsilon = max(0.1, 0.6 - episode_num * 0.001)  # Decaying epsilon
    
    # Store all positions and states for animation
    positions = [(env.vector_agentState[0], env.vector_agentState[1])]
    actions_taken = []
    rewards_received = []
    exploration_flags = []
    
    print(f"Running training episode with epsilon={epsilon:.3f}...")
    while not done and steps < env.max_episode_steps:
        # Use epsilon-greedy action for training
        action = agent.get_action(state, epsilon=epsilon)
        
        # Determine if this was an exploration move
        greedy_action = agent.get_action(state, epsilon=0.0)
        is_exploration = (action != greedy_action)
        exploration_flags.append(is_exploration)
        
        next_state, next_state_flag, reward, done, _ = env.step(action)
        
        positions.append((env.vector_agentState[0], env.vector_agentState[1]))
        actions_taken.append(action)
        rewards_received.append(reward)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    success = next_state_flag == 'goal'
    print(f"Training episode completed: {'SUCCESS' if success else 'FAILED'} - Reward: {episode_reward:.2f}, Steps: {steps}")
    
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
        
        # Plot obstacles
        for i in range(len(env.Obstacle_x)):
            rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
                                obstacle_width, obstacle_width, fc='blue', ec="blue", alpha=0.8)
            ax1.add_patch(rectangle)
        
        # Plot start and goal
        ax1.scatter(starting_position[0], starting_position[1], c='green', s=150, marker='s', 
                   label="Start", edgecolors='black', linewidth=2)
        ax1.scatter(target_position[0], target_position[1], c='red', s=150, marker='s', 
                   label="Goal", edgecolors='black', linewidth=2)
        
        # Plot path up to current frame with different colors for exploration vs exploitation
        if frame > 0:
            for i in range(1, min(frame+1, len(positions))):
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                
                # Color based on exploration
                color = 'red' if i-1 < len(exploration_flags) and exploration_flags[i-1] else 'orange'
                alpha = 0.5 if color == 'red' else 0.8
                
                ax1.plot([prev_pos[0], curr_pos[0]], [prev_pos[1], curr_pos[1]], 
                        color=color, linewidth=2, alpha=alpha)
        
        # Plot current agent position
        current_pos = positions[frame]
        ax1.scatter(current_pos[0], current_pos[1], c='purple', s=200, marker='o', 
                   label='DQN Agent (Training)', edgecolors='white', linewidth=3, zorder=10)
        
        ax1.set_title(f'DQN Training Episode {episode_num} - Step {frame}/{len(positions)-1}\n'
                     f'Epsilon: {epsilon:.3f} | {"SUCCESS!" if success and frame == len(positions)-1 else ""}', 
                     fontsize=14, fontweight='bold')
        
        # Custom legend
        import matplotlib.lines as mlines
        exploit_line = mlines.Line2D([], [], color='orange', linewidth=2, label='Exploitation')
        explore_line = mlines.Line2D([], [], color='red', linewidth=2, label='Exploration')
        ax1.legend(handles=[exploit_line, explore_line], loc='upper right')
        
        # Information panel
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Display current information
        info_text = f"Training Episode {episode_num}\n\n"
        info_text += f"Step: {frame}/{len(positions)-1}\n"
        info_text += f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})\n"
        info_text += f"Epsilon: {epsilon:.3f}\n\n"
        
        if frame > 0 and frame-1 < len(actions_taken):
            action_idx = actions_taken[frame-1]
            action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
            is_explore = exploration_flags[frame-1] if frame-1 < len(exploration_flags) else False
            action_type = "🎲 Explore" if is_explore else "🎯 Exploit"
            
            info_text += f"Last Action: {action_name}\n"
            info_text += f"Action Type: {action_type}\n"
            info_text += f"Reward: {rewards_received[frame-1]:.2f}\n\n"
        
        info_text += f"Total Reward: {sum(rewards_received[:frame]):.2f}\n"
        info_text += f"Status: {'Complete' if frame == len(positions)-1 else 'Training'}\n"
        
        if frame == len(positions)-1:
            info_text += f"Result: {'SUCCESS' if success else 'FAILED'}"
        
        ax2.text(1, 5, info_text, fontsize=11, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(positions), 
                                  interval=interval, repeat=True, blit=False)
    
    # Save as GIF if requested
    if save_gif:
        filename = f'DQN_training_episode_{episode_num}.gif'
        print(f"Saving training animation as {filename}...")
        try:
            anim.save(filename, writer='pillow', fps=3)
            print(f"Training animation saved as '{filename}'")
        except Exception as e:
            print(f"Error saving training animation: {e}")
    
    plt.tight_layout()
    plt.show()
    
    return anim, {'success': success, 'steps': steps, 'reward': episode_reward, 'epsilon': epsilon}


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


if __name__ == "__main__":
    print("This is an animation module for DQN evaluation.")
    print("Import this module and use animate_evaluation() or animate_training_episode() functions.")
    print("Example usage:")
    print("  from dqn_animation import animate_evaluation, animate_training_episode")
    print("  anim, results = animate_evaluation(agent, env, start_pos, target_pos)")
    print("  anim, results = animate_training_episode(agent, env, start_pos, target_pos, episode_num=50)")
