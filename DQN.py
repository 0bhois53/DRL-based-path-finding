import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np
from Agent import DQNAgent
from environment import Environment
from environment import final_states
from environment import obstacle_width
import os

os.makedirs('DQN_visuals', exist_ok=True)

def load_selected_points(filename='selected_points.txt'):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            start = [int(x) for x in lines[0].strip().split(',')]
            target = [int(x) for x in lines[1].strip().split(',')]
        print(f"Loaded custom points - Start: {start}, Target: {target}")
        return start, target
    except:
        # Default values if file doesn't exist
        print("Using default points - Start: [10, 0], Target: [90, 100]")
        return [10, 0], [90, 100]

TARGET_UPDATE = 5
num_episodes = 350
hidden = 128
gamma = 0.99
replay_buffer_size = 100000
batch_size = 128
from scipy.interpolate import splprep, splev
eps_stop = 0.1
epsilon=eps = 0.6
Start_epsilon_decaying = 0
End_epsilon_decaying = num_episodes
epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)

# Load custom start and target positions or use defaults
starting_position, target_position = load_selected_points()

n_actions = 8

    # Advanced path smoothing with adaptive parameters
def smooth_path(x, y, s=2, smoothing_factor='adaptive'):
    """
    Custom path smoothing with multiple techniques
    Args:
        x, y: path coordinates
        s: smoothing parameter
        smoothing_factor: 'adaptive', 'conservative', or 'aggressive'
    """
    if len(x) < 4:
        return x, y  
    
    if smoothing_factor == 'adaptive':
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        s = max(1, min(5, path_length / 20))  # Dynamic smoothing
    elif smoothing_factor == 'conservative':
        s = s * 0.5
    elif smoothing_factor == 'aggressive':
        s = s * 2.0
    
    tck, u = splprep([x, y], s=s)
    unew = np.linspace(0, 1, max(100, len(x)*3))
    out = splev(unew, tck)
    return out[0], out[1]

state_space_dim = 2
env = Environment(starting_position, target_position, 100, 100, n_actions)
class DQNPerformanceTracker:
    """Custom performance tracking and analysis class"""
    def __init__(self):
        self.episode_metrics = []
        self.success_episodes = []
        self.failure_episodes = []
    
    def log_episode(self, episode, reward, steps, success, path_efficiency=None):
        """Log comprehensive episode metrics"""
        metrics = {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'success': success,
            'efficiency': path_efficiency or steps,
            'timestamp': time.time()
        }
        self.episode_metrics.append(metrics)
        
        if success:
            self.success_episodes.append(episode)
        else:
            self.failure_episodes.append(episode)
    
    def analyze_learning_curve(self):
        """Analyze learning progression with custom metrics"""
        if len(self.episode_metrics) < 10:
            return {}
        
        avg_steps_recent = np.mean([m['steps'] for m in self.episode_metrics[-20:]])
        
        return {
            'avg_steps_recent': avg_steps_recent,
            'total_episodes': len(self.episode_metrics),
            'learning_stability': self._calculate_stability()
        }
    
    def _calculate_stability(self):
        """Custom stability metric based on reward variance"""
        if len(self.episode_metrics) < 20:
            return 0.0
        recent_rewards = [m['reward'] for m in self.episode_metrics[-20:]]
        return 1.0 / (1.0 + np.std(recent_rewards))

# Initialize performance tracker
performance_tracker = DQNPerformanceTracker()

if __name__=="__main__":
    import time
    import psutil
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                 hidden, gamma)
    random.seed(20)
    env.reset()

    # Training loop
    cumulative_rewards = []
    Num_steps = []
    counter_reach_goal = 0

    final_path=[]
    visited_X = [starting_position[0]]
    visited_Y = [starting_position[1]]

    # Create directory for models
    os.makedirs('saved_models', exist_ok=True)
    best_reward = float('-inf')

    start_time = time.time()
    process = psutil.Process()
    cpu_usages = []
    mem_usages = []

    for ep in range(num_episodes):
        # Monitor resources at the start of each episode
        cpu_usages.append(process.cpu_percent(interval=None))
        mem_usages.append(process.memory_info().rss / (1024 * 1024))  # in MB
        # Initialize the environment and state
        # print('training started ...')
        state = env.reset()
        done = False
        eps -= epsilon_decaying
        epsilon = max(0.01, eps)
        cum_reward = 0
        counter = 0
        number_of_steps_taken_to_terminal = 0
        visited_X_final = []
        visited_Y_final = []


         
        
        #print("episode number: ",ep)
        while not done and counter < env.max_episode_steps:
            action = agent.get_action(state, epsilon)

            visited_X_final.append(env.vector_agentState[0])
            visited_Y_final.append(env.vector_agentState[1])

            next_state,next_state_flag, reward, done, _ = env.step(action)
        
            cum_reward += reward
            
            agent.store_transition(state, action, next_state, reward, done) 
            agent.update_network() 

            state = next_state
            counter +=1
            number_of_steps_taken_to_terminal  += 1
        
        #print(state)
        if done:
            # Enhanced logging with performance tracking
            path_efficiency = number_of_steps_taken_to_terminal / max(1, len(visited_X_final))
            performance_tracker.log_episode(ep, cum_reward, number_of_steps_taken_to_terminal, 
                                           next_state_flag == 'goal', path_efficiency)
            
            print(f'Episode {ep}: Steps: {number_of_steps_taken_to_terminal}, Reward: {cum_reward:.2f}')
            Num_steps.append(number_of_steps_taken_to_terminal)
            cumulative_rewards.append(cum_reward)
            print("episode: %d: reward: %6.2f" % ( ep, cum_reward))
            
            # Dynamic analysis every 25 episodes
            if ep % 25 == 0 and ep > 0:
                analysis = performance_tracker.analyze_learning_curve()
                print(f"Learning Analysis - Avg Steps: {analysis.get('avg_steps_recent', 0):.1f}, "
                      f"Stability: {analysis.get('learning_stability', 0):.3f}")
            
            print("**********************************************")
            # Count success if agent reached the goal (customize if needed)
            if next_state_flag == 'goal':
                counter_reach_goal += 1
            # Save model if it achieves better reward
            if cum_reward > best_reward:
                best_reward = cum_reward
                agent.save_model(f'saved_models/dqn_best_model.pth')
            # Save checkpoint every 50 episodes
            if ep % 50 == 0:
                agent.save_model(f'saved_models/dqn_checkpoint_ep{ep}.pth')

        # Update the target network, copying all weights and biases in DQN
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network()
    env.final()

    # Save final model
    agent.save_model('saved_models/dqn_final_model.pth')
    print("Training completed. Final model saved to 'saved_models/dqn_final_model.pth'")
    
    # Evaluate the trained model
    def evaluate_model(agent, env, num_eval_episodes=10, model_path='saved_models/dqn_final_model.pth'):
        """Evaluate the trained DQN model"""
        print(f"\nEvaluating model from {model_path}...")
        agent.load_model(model_path)
        
        success_count = 0
        total_rewards = []
        total_steps = []
        
        for eval_ep in range(num_eval_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            eval_path_x = [env.vector_agentState[0]]
            eval_path_y = [env.vector_agentState[1]]
            
            while not done and steps < env.max_episode_steps:
                # Use greedy action (epsilon=0) for evaluation
                action = agent.get_action(state, epsilon=0.0)
                next_state, next_state_flag, reward, done, _ = env.step(action)
                
                eval_path_x.append(env.vector_agentState[0])
                eval_path_y.append(env.vector_agentState[1])
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            if next_state_flag == 'goal':
                success_count += 1
                print(f"Eval Episode {eval_ep + 1}: SUCCESS - Reward: {episode_reward:.2f}, Steps: {steps}")
            else:
                print(f"Eval Episode {eval_ep + 1}: FAILED - Reward: {episode_reward:.2f}, Steps: {steps}")
            
            # Save path visualization for first 3 episodes
            if eval_ep < 3:
                plt.figure(figsize=(8, 6))
                plt.plot(eval_path_x, eval_path_y, 'g-', linewidth=2, label=f'Eval Path {eval_ep + 1}')
                plt.scatter(starting_position[0], starting_position[1], c='green', s=100, label="Start")
                plt.scatter(target_position[0], target_position[1], c='red', s=100, label="Target")
                
                # Plot obstacles
                for i in range(len(env.Obstacle_x)):
                    rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
                                            obstacle_width, obstacle_width, fc='blue', ec="blue")
                    plt.gca().add_patch(rectangle)

                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.title(f'DQN Evaluation Episode {eval_ep + 1} - {"SUCCESS" if next_state_flag == "goal" else "FAILED"}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'DQN_Eval_Episode_{eval_ep + 1}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Print evaluation summary
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Best Reward: {max(total_rewards):.2f}")
        print(f"Worst Reward: {min(total_rewards):.2f}")
        
        return avg_reward, avg_steps
    
    # Run evaluation
    evaluate_model(agent, env, num_eval_episodes=10)
    
# Print additional training parameters
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average CPU usage per episode: {np.mean(cpu_usages):.2f}%")
    print(f"Peak memory usage: {np.max(mem_usages):.2f} MB")

    # Plot cumulative rewards (adjust range to match actual data length)
    plt.figure(tight_layout=True)
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label='cumulative rewards', color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Accumulated reward', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('DQN_visuals/DQN_Accumulated_Reward.png', format='png', dpi=300)

    # Plot number of steps (adjust range to match actual data length)
    plt.figure(tight_layout=True)
    plt.plot(range(len(Num_steps)), Num_steps, color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Taken steps', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('DQN_visuals/DQN_Steps_per_Episode.png', format='png', dpi=300)

    ### Plot the trajectory
    final_path=list(final_states().values())
    print(final_path)
    for i in range(len(final_path)):
        visited_X.append(final_path[i][0])
        visited_Y.append(final_path[i][1])

    ### Plot the trajectory
    x_shortest = np.append(np.array(visited_X), env.Terminal[0])
    y_shortest = np.append(np.array(visited_Y), env.Terminal[1])

    x_final = np.append(np.array(visited_X_final), env.Terminal[0])
    y_final = np.append(np.array(visited_Y_final), env.Terminal[1])

    # B-spline smoothing for final path
    # Only apply B-spline smoothing if enough unique points
    unique_points = list(dict.fromkeys(zip(x_final, y_final)))
    if len(unique_points) >= 4:
        x_unique, y_unique = zip(*unique_points)
        x_smooth, y_smooth = smooth_path(x_unique, y_unique)
    else:
        print("Not enough unique points for B-spline smoothing. Using raw path.")
        x_smooth, y_smooth = x_final, y_final

    x_o = env.Obstacle_x 
    y_o = env.Obstacle_y

    plt.figure()

   
    plt.quiver(x_shortest[:-1], y_shortest[:-1], x_shortest[1:]-x_shortest[:-1], y_shortest[1:]-y_shortest[:-1], scale_units='xy', angles='xy', scale=1)

    # Plot smoothed path (B-spline)
    plt.plot(x_smooth, y_smooth, color='orange', linewidth=2, label='Smoothed Path (B-spline)')



    for i in range(len(env.Obstacle_x)):
        rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
                                obstacle_width, obstacle_width, fc='blue', ec="blue")
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0],starting_position[1], ec = 'k', c ='red', s=100, label ="Start")
    plt.scatter(target_position[0],target_position[1], ec = 'k', c ='red', s =100,label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('DQN_visuals/DQN_Shortest_Path.png', format='png', dpi=300)

    plt.figure()
   
    plt.quiver(x_final[:-1], y_final[:-1], x_final[1:]-x_final[:-1], y_final[1:]-y_final[:-1], scale_units='xy', angles='xy', scale=1)

    # Plot smoothed path (B-spline)
    plt.plot(x_smooth, y_smooth, color='orange', linewidth=2, label='Smoothed Path (B-spline)')

    #plt.scatter(x_s, y_s, c = 'k' ,marker = "o",label = 'Sensor')

    rectangle = Rectangle(( 10* (x_o[i]-0.5), 10*(10 - y_o[i] -0.5)), obstacle_width, obstacle_width, fc='blue',ec="blue")
    plt.gca().add_patch(rectangle)
    plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0],starting_position[1], ec = 'k', c ='red', s=100, label ="Start")
    plt.scatter(target_position[0],target_position[1], ec = 'k', c ='red', s =100,label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('DQN_visuals/DQN_Final_Path.png', format='png', dpi=300)
    plt.show()

    # Animation options
    print("\n" + "="*60)
    print("DQN ANIMATION OPTIONS")
    print("="*60)
    
    try:
        import sys
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == '--animate':
                print("Running animated evaluation...")
                from dqn_animation import animate_evaluation
                anim, results = animate_evaluation(agent, env, starting_position, target_position,
                                                 model_path='saved_models/dqn_final_model.pth',
                                                 save_gif=True, interval=600)
            elif sys.argv[1] == '--animate-fast':
                print("Running fast animated evaluation...")
                from dqn_animation import animate_evaluation
                anim, results = animate_evaluation(agent, env, starting_position, target_position,
                                                 model_path='saved_models/dqn_final_model.pth',
                                                 save_gif=True, interval=300)
            elif sys.argv[1] == '--animate-training':
                print("Running animated training episode...")
                from dqn_animation import animate_training_episode
                anim, results = animate_training_episode(agent, env, starting_position, target_position,
                                                       episode_num=num_episodes, save_gif=True)
        else:
            # Ask user if they want to see animated evaluation
            user_input = input("\nWould you like to see an animated evaluation? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                print("Creating animated evaluation...")
                from dqn_animation import animate_evaluation
                anim, results = animate_evaluation(agent, env, starting_position, target_position,
                                                 model_path='saved_models/dqn_final_model.pth',
                                                 save_gif=True, interval=600)
            
            # Ask about training animation
            user_input = input("Would you like to see an animated training episode demo? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                print("Creating animated training episode...")
                from dqn_animation import animate_training_episode
                anim, results = animate_training_episode(agent, env, starting_position, target_position,
                                                       episode_num=num_episodes, save_gif=True)
                
    except KeyboardInterrupt:
        print("\nAnimation options skipped.")
    except Exception as e:
        print(f"Animation error: {e}")
        print("Continuing without animation...")
    
    print("\nDQN training and evaluation completed!")

