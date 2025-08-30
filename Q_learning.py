from environment import Environment
from environment import final_states 
from environment import obstacle_width
from Agent import QLearningTable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from scipy.interpolate import splprep, splev

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

gamma = 0.99
num_episodes=500
epsilon=eps =0.2
Start_epsilon_decaying = 0
#End_epsilon_decaying = num_episodes // 1
End_epsilon_decaying = num_episodes
epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)

n_actions = 8

# Load custom start and target positions or use defaults
starting_position, target_position = load_selected_points()
env = Environment(starting_position, target_position, 100, 100, n_actions)

# Create directory for saving models
import os
os.makedirs('saved_models_Q_learning', exist_ok=True)
os.makedirs('Q_learning_visuals', exist_ok= True)

def update():
    import time
    try:
        import psutil
        psutil_available = True
    except ImportError:
        psutil_available = False
    start_time = time.time()
    process = None
    if psutil_available:
        process = psutil.Process(os.getpid())
    # Resulted list for the plotting Episodes via Steps
    Num_steps = []
    # Summed costs for all episodes in resulted list
    cumulative_rewards = []

    final_path=[]
    visited_X = [starting_position[0]]
    visited_Y = [starting_position[1]]

    for ep in range(num_episodes): 
        # Initial state
        state = env.reset() 
        done = False
        global eps
        eps -= epsilon_decaying
        epsilon = max(0.01, eps)
        cum_reward = 0 # Cummulative reward  for each episode
        number_of_steps_taken_to_terminal = 0 # Updating number of Steps for each Episode
        visited_X_final = []
        visited_Y_final = []
        
        while not done :
            # agent chooses action based on state
            action = agent.get_action(str(state),epsilon) 
            visited_X_final.append(env.vector_agentState[0])
            visited_Y_final.append(env.vector_agentState[1])

            # agent takes an action and get the next state and reward
            next_state, next_state_flag,reward, done, _ = env.step(action) 

            # agent learns from this transition and calculating the cost
            cum_reward += agent.learn(str(state), action, reward, str(next_state),next_state_flag)

            # Swapping the states - current and next
            state = next_state

            # Calculating number of Steps in the current Episode
            number_of_steps_taken_to_terminal += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                print('number of steps taken by the agent: ', number_of_steps_taken_to_terminal)
                Num_steps.append(number_of_steps_taken_to_terminal)
                cumulative_rewards.append(cum_reward)
                
                # Save model if it achieves better reward
                if len(cumulative_rewards) == 1 or cum_reward > max(cumulative_rewards[:-1]):
                    agent.save_model('saved_models_Q_learning/q_learning_best_model.pkl')
                    print(f"New best model saved with reward: {cum_reward}")
                
                # Save checkpoint every 50 episodes
                if ep % 50 == 0:
                    agent.save_model(f'saved_models_Q_learning/q_learning_checkpoint_ep{ep}.pkl')
                
                print("episode: %d: reward: %6.2f" % ( ep, cum_reward))
                print("**********************************************")
                break

    # Showing the Q-table with values for each action
    agent.print_q_table()
    # Showing the final route
    env.final()
    
    # Save final model
    agent.save_model('saved_models_Q_learning/q_learning_final_model.pkl')
    print("Training completed. Final model saved to 'saved_models_Q_learning/q_learning_final_model.pkl'")

    # Evaluate the trained model on training environment first
    print("\n" + "="*60)
    print("EVALUATING ON TRAINING ENVIRONMENT (Same obstacles as training)")
    print("="*60)
    training_eval_results = evaluate_model(agent, env, num_eval_episodes=5, 
                                         model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                         use_randomized_env=False, 
                                         start_pos=starting_position, 
                                         target_pos=target_position)

    # Evaluate the trained model on randomized environments 
    print("\n" + "="*60)
    print("EVALUATING ON RANDOMIZED ENVIRONMENTS (Testing generalization)")
    print("="*60)
    randomized_eval_results = evaluate_model(agent, env, num_eval_episodes=10, 
                                            model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                            use_randomized_env=True, 
                                            start_pos=starting_position, 
                                            target_pos=target_position)

    # Training time and resource usage
    end_time = time.time()
    training_time = end_time - start_time
    if psutil_available and process:
        cpu_usage = process.cpu_percent(interval=1)
        mem_usage = process.memory_info().rss / (1024 ** 2)
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"CPU Usage (%): {cpu_usage}")
        print(f"Memory Usage (MB): {mem_usage:.2f}")
    else:
        print(f"Training Time: {training_time:.2f} seconds")
        print("psutil not installed. To track CPU and memory usage, run 'pip install psutil'.")

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), cumulative_rewards, label='cumulative rewards', color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Accumulated reward', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('Q_learning_visuals/Q_learning_Accumulated_Reward.png', format='png', dpi=300)

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), Num_steps, color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Taken steps', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('Q_learning_visuals/Q_learning_Steps_per_Episode.png', format='png', dpi=300)

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
    x_o = env.Obstacle_x 
    y_o = env.Obstacle_y

    # B-spline smoothing function
    def smooth_path(x, y, s=2):
        if len(x) < 4:
            return x, y  # Not enough points to smooth
        tck, u = splprep([x, y], s=s)
        unew = np.linspace(0, 1, max(100, len(x)*3))
        out = splev(unew, tck)
        return out[0], out[1]

    # Smooth the final path
    x_smooth, y_smooth = smooth_path(x_final, y_final)

    plt.figure()
    plt.quiver(x_shortest[:-1], y_shortest[:-1], x_shortest[1:]-x_shortest[:-1], y_shortest[1:]-y_shortest[:-1], scale_units='xy', angles='xy', scale=1)

    # Plot smoothed path (B-spline)
    plt.plot(x_smooth, y_smooth, color='orange', linewidth=2, label='Smoothed Path (B-spline)')

    
    for i in range(len(x_o)):
        rectangle = mpatches.Rectangle((10 * (x_o[i] - 0.5), 10 * (10 - y_o[i] - 0.5)), obstacle_width, obstacle_width, fc='blue', ec='blue')
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0], starting_position[1], edgecolors='k', facecolors='red', s=100, label="Start")
    plt.scatter(target_position[0], target_position[1], edgecolors='k', facecolors='red', s=100, label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Q_learning_visuals/Q_learning_Shortest_Path.png', format='png', dpi=300)

    plt.figure()
    plt.quiver(x_final[:-1], y_final[:-1], x_final[1:]-x_final[:-1], y_final[1:]-y_final[:-1], scale_units='xy', angles='xy', scale=1)

    # Plot smoothed path (B-spline)
    plt.plot(x_smooth, y_smooth, color='orange', linewidth=2, label='Smoothed Path (B-spline)')

    for i in range(len(x_o)):
        rectangle = mpatches.Rectangle((10 * (x_o[i] - 0.5), 10 * (10 - y_o[i] - 0.5)), obstacle_width, obstacle_width, fc='blue', ec='blue')
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0], starting_position[1], edgecolors='k', facecolors='red', s=100, label="Start")
    plt.scatter(target_position[0], target_position[1], edgecolors='k', facecolors='red', s=100, label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Q_learning_visuals/Q_learning_Final_Path.png', format='png', dpi=300)
    plt.show()
    # # Plotting the results
    
    # Return evaluation results for comparison
    return training_eval_results, randomized_eval_results

def create_randomized_environment(start_pos, target_pos, grid_size_x=100, grid_size_y=100, n_actions=8):
    """Create a new environment with randomized obstacles for evaluation"""
    print("Creating new evaluation environment with randomized obstacles...")
    eval_env = Environment(start_pos, target_pos, grid_size_x, grid_size_y, n_actions)
    print(f"Created evaluation environment with {len(eval_env.Obstacle_x)} obstacles")
    return eval_env

def evaluate_model(agent, env, num_eval_episodes=10, model_path='saved_models_Q_learning/q_learning_final_model.pkl', 
                  use_randomized_env=True, start_pos=None, target_pos=None):
    """Evaluate the trained Q-Learning model on randomized environments"""
    print(f"\nEvaluating model from {model_path}...")
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using current agent state.")
    except Exception as e:
        print(f"Error loading model: {e}. Using current agent state.")
    
    # Use provided positions or get from original environment
    if start_pos is None:
        start_pos = [env.vector_state0[0], env.vector_state0[1]]
    if target_pos is None:
        target_pos = [env.Terminal[0], env.Terminal[1]]
    
    success_count = 0
    total_rewards = []
    total_steps = []
    evaluation_environments = []
    
    for eval_ep in range(num_eval_episodes):
        # Create a new randomized environment for each evaluation episode if requested
        if use_randomized_env:
            eval_env = create_randomized_environment(start_pos, target_pos, 100, 100, env.num_actions)
            evaluation_environments.append(eval_env)
            print(f"Episode {eval_ep + 1}: Created new environment with {len(eval_env.Obstacle_x)} obstacles")
        else:
            eval_env = env
            evaluation_environments.append(eval_env)
        
        state = eval_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        eval_path_x = [eval_env.vector_agentState[0]]
        eval_path_y = [eval_env.vector_agentState[1]]
        
        while not done and steps < eval_env.max_episode_steps:
            # Use greedy action (epsilon=0) for evaluation
            # Save current epsilon and set to 0 for greedy evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            
            action = agent.get_action(str(state), 0.0)
            next_state, next_state_flag, reward, done, _ = eval_env.step(action)
            
            # Restore original epsilon
            agent.epsilon = original_epsilon
            
            eval_path_x.append(eval_env.vector_agentState[0])
            eval_path_y.append(eval_env.vector_agentState[1])
            
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
            plt.figure(figsize=(10, 8))
            plt.plot(eval_path_x, eval_path_y, 'g-', linewidth=3, label=f'Agent Path', alpha=0.8)
            plt.scatter(start_pos[0], start_pos[1], c='green', s=150, 
                       label="Start", edgecolors='black', linewidth=2, zorder=5)
            plt.scatter(target_pos[0], target_pos[1], c='red', s=150, 
                       label="Target", edgecolors='black', linewidth=2, zorder=5)
            
            # Plot obstacles for the current evaluation environment
            for i in range(len(eval_env.Obstacle_x)):
                rectangle = mpatches.Rectangle((10 * (eval_env.Obstacle_x[i] - 0.5), 10 * (10 - eval_env.Obstacle_y[i] - 0.5)), 
                                                obstacle_width, obstacle_width, fc='blue', ec="darkblue", alpha=0.8)
                plt.gca().add_patch(rectangle)

            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.xlabel('x (m)', fontsize=12)
            plt.ylabel('y (m)', fontsize=12)
            env_type = "Randomized" if use_randomized_env else "Training"
            status = "SUCCESS" if next_state_flag == "goal" else "FAILED"
            plt.title(f'Q-Learning Evaluation Episode {eval_ep + 1}\n{env_type} Environment - {status}', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.gca().set_aspect('equal', adjustable='box')
            
            filename_suffix = "_Randomized" if use_randomized_env else "_Training"
            plt.savefig(f'Q_learning_Eval{filename_suffix}_Episode_{eval_ep + 1}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Print evaluation summary
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = (success_count / num_eval_episodes) * 100
    
    env_type_str = "RANDOMIZED ENVIRONMENT" if use_randomized_env else "TRAINING ENVIRONMENT"
    print(f"\n=== EVALUATION RESULTS ({env_type_str}) ===")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{num_eval_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Best Reward: {max(total_rewards):.2f}")
    print(f"Worst Reward: {min(total_rewards):.2f}")
    
    if use_randomized_env:
        print(f"Evaluated on {num_eval_episodes} different randomized environments")
        print(f"Each environment had between 8-15 randomly placed obstacles")
        if success_count > 0:
            print(f"Agent successfully generalized to {success_count} out of {num_eval_episodes} new environments")
    
    if success_count > 0:
        successful_rewards = [total_rewards[i] for i in range(num_eval_episodes) 
                            if total_steps[i] < evaluation_environments[i].max_episode_steps and total_rewards[i] > 0]
        successful_steps = [total_steps[i] for i in range(num_eval_episodes) 
                          if total_steps[i] < evaluation_environments[i].max_episode_steps and total_rewards[i] > 0]
        if successful_rewards:
            print(f"Average Successful Episode Reward: {np.mean(successful_rewards):.2f}")
            print(f"Average Successful Episode Steps: {np.mean(successful_steps):.1f}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'total_rewards': total_rewards,
        'total_steps': total_steps,
        'use_randomized_env': use_randomized_env,
        'environments': evaluation_environments
    }





if __name__ == "__main__":
    import sys
    
    agent = QLearningTable(actions=list(range(n_actions)),
                    learning_rate=0.1,
                    reward_decay=0.9,
                    e_greedy=0.9) 
    # Run training and evaluation
    training_eval_results, randomized_eval_results = update() 
    print("Training completed.")
    
    # Optional: Run additional evaluation with different parameters
    print("\n" + "="*50)
    print("RUNNING ADDITIONAL RANDOMIZED ENVIRONMENT EVALUATION...")
    print("="*50)
    additional_eval_results = evaluate_model(agent, env, num_eval_episodes=5, 
                                           model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                           use_randomized_env=True,
                                           start_pos=starting_position, 
                                           target_pos=target_position)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Training Environment Success Rate: {training_eval_results['success_rate']:.1f}%")
    print(f"Randomized Environment Success Rate: {randomized_eval_results['success_rate']:.1f}%")
    
    generalization_performance = randomized_eval_results['success_rate'] / max(training_eval_results['success_rate'], 1) * 100
    print(f"Generalization Performance: {generalization_performance:.1f}% of training performance")
    
    if randomized_eval_results['success_rate'] >= 70:
        print("✅ EXCELLENT: Agent shows strong generalization to new environments!")
    elif randomized_eval_results['success_rate'] >= 50:
        print("✅ GOOD: Agent shows decent generalization to new environments!")
    elif randomized_eval_results['success_rate'] >= 30:
        print("⚠️  FAIR: Agent shows limited generalization to new environments.")
    else:
        print("❌ POOR: Agent struggles to generalize to new environments.")
    
    # Ask user if they want to see animated evaluation
    try:
        user_input = input("\nWould you like to see animated evaluations? (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            print("\nChoose animation type:")
            print("1. Training environment only")
            print("2. Randomized environment only") 
            print("3. Multiple randomized environments (3 episodes)")
            print("4. Comparison (Training vs Randomized)")
            
            try:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == '1':
                    print("Creating training environment animation...")
                    from q_learning_animation import animate_evaluation
                    anim, results = animate_evaluation(agent, env, starting_position, target_position,
                                                     model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                                     save_gif=True, interval=600, use_randomized_env=False)
                
                elif choice == '2':
                    print("Creating randomized environment animation...")
                    from q_learning_animation import animate_evaluation
                    anim, results = animate_evaluation(agent, env, starting_position, target_position,
                                                     model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                                     save_gif=True, interval=600, use_randomized_env=True)
                
                elif choice == '3':
                    print("Creating multiple randomized environment animations...")
                    from q_learning_animation import animate_randomized_evaluations
                    anims, results = animate_randomized_evaluations(agent, env, starting_position, target_position,
                                                                   num_episodes=3,
                                                                   model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                                                   save_gifs=True, interval=600)
                
                elif choice == '4':
                    print("Creating comparison animations...")
                    from q_learning_animation import animate_comparison
                    comparison_results = animate_comparison(agent, env, starting_position, target_position,
                                                          model_path='saved_models_Q_learning/q_learning_final_model.pkl',
                                                          save_gifs=True, interval=600)
                
                else:
                    print("Invalid choice. Skipping animation.")
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or cancelled. Skipping animation.")
                
    except KeyboardInterrupt:
        print("\nSkipping animated evaluation.")