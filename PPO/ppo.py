import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, final_states
import os
from scipy.interpolate import splprep, splev


os.makedirs('PPO_runs', exist_ok=True)
os.makedirs('PPO_visuals', exist_ok=True)

def load_selected_points(filename='T:/PPO/DRL-based-path-finding/PPO/selected_points.txt'):
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

# PPO Actor-Critic Networks
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, action_dim)
		
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return torch.softmax(x, dim=-1)

class Critic(nn.Module):
	def __init__(self, state_dim, hidden_dim=256):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)
		
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x
# PPO Agent
class PPOAgent:
	def __init__(self, state_dim, action_dim, lr=2e-4, gamma=0.999, clip=0.2, update_epochs=15, 
				 hidden_dim=256, entropy_coeff=0.045, device=None):
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic = Critic(state_dim, hidden_dim).to(self.device)
		self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
		self.gamma = gamma
		self.clip = clip
		self.update_epochs = update_epochs
		self.entropy_coeff = entropy_coeff

	def select_action(self, state):
		state = torch.FloatTensor(state).to(self.device)
		probs = self.actor(state)
		dist = torch.distributions.Categorical(probs)
		action = dist.sample()
		return action.item(), dist.log_prob(action), dist.entropy()

	def evaluate(self, state, action):
		probs = self.actor(state)
		dist = torch.distributions.Categorical(probs)
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		value = self.critic(state)
		return log_prob, entropy, value

	def save(self, path):
		torch.save({
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
			'optimizer': self.optimizer.state_dict()
		}, path)
    
	def load(self, path):
		checkpoint = torch.load(path)
		self.actor.load_state_dict(checkpoint['actor'])
		self.critic.load_state_dict(checkpoint['critic'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])

# Path smoothening algorithms
def moving_average_path(path, window=3):
	path = np.array(path)
	smoothed = np.convolve(path[:,0], np.ones(window)/window, mode='same'), np.convolve(path[:,1], np.ones(window)/window, mode='same')
	return np.stack(smoothed, axis=1)

def spline_smooth_path(path, s=2):
	path = np.array(path)
	# Remove duplicate points
	unique_points = list(dict.fromkeys(map(tuple, path)))
	
	if len(unique_points) < 4:
		return path
	try:
		x, y = zip(*unique_points)
		x = np.array(x)
		y = np.array(y)
		if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
			raise ValueError("Invalid path shape for spline smoothing.")
		if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
			raise ValueError("Non-finite values in path for spline smoothing.")
		tck, u = splprep([x, y], s=s)
		x_new, y_new = splev(np.linspace(0,1,max(100, len(x)*3)), tck)
		return np.stack([x_new, y_new], axis=1)
	except Exception as e:
		print(f"Spline smoothing failed: {e}. Using raw path.")
		return path

# Training and visualization
def train_ppo(env, agent, episodes=500, max_steps=300, batch_size=128, save_path='PPO_runs/ppo_gridworld.pt'):
	all_rewards = []  # Stores episode rewards
	all_steps = []
	best_path = None
	best_steps = float('inf')
	final_path = None
	memory = []
	for ep in range(episodes):
		state = env.reset()
		state_flat = state.flatten()
		ep_reward = 0
		ep_steps = 0
		traj = []
		done = False
		while not done and ep_steps < max_steps:
			action, log_prob, entropy = agent.select_action(state_flat)
			next_state, next_state_flag, reward, done, _ = env.step(action)
			next_state_flat = next_state.flatten()
			traj.append((state_flat, action, log_prob, reward, next_state_flat, done))
			ep_reward += reward  # Accumulate reward per step
			ep_steps += 1
			state_flat = next_state_flat
		all_rewards.append(ep_reward)  # Store episode reward at end of episode
		all_steps.append(ep_steps)
		memory.extend(traj)
		# Save best path
		if done and ep_steps < best_steps:
			best_steps = ep_steps
			best_path = [env.dic[i] for i in range(len(env.dic))]
		# Save final path (last successful)
		if done:
			final_path = [env.dic[i] for i in range(len(env.dic))]
		# PPO update
		if len(memory) >= batch_size:
			states, actions, log_probs, rewards, next_states, dones = zip(*memory)
			states = torch.FloatTensor(states).to(agent.device)
			actions = torch.LongTensor(actions).to(agent.device)
			old_log_probs = torch.stack(log_probs).detach().to(agent.device)
			returns = compute_gae(agent, rewards, states, next_states, dones)
			returns = torch.FloatTensor(returns).to(agent.device)
			for _ in range(agent.update_epochs):
				log_prob, entropy, values = agent.evaluate(states, actions)
				advantage = returns - values.squeeze()
				ratio = torch.exp(log_prob - old_log_probs)
				surr1 = ratio * advantage
				surr2 = torch.clamp(ratio, 1-agent.clip, 1+agent.clip) * advantage
				actor_loss = -torch.min(surr1, surr2).mean()
				critic_loss = advantage.pow(2).mean()
				loss = actor_loss + 0.5 * critic_loss - agent.entropy_coeff * entropy.mean()
				agent.optimizer.zero_grad()
				loss.backward()
				agent.optimizer.step()
			memory = []
		# Save model
		if (ep+1) % 50 == 0:
			agent.save(save_path)
		print(f"Episode {ep+1}: Reward={ep_reward}, Steps={ep_steps}")
		np.savetxt('rewards_ppo.txt', all_rewards)
	# Final save
	agent.save(save_path)

	# Visualizations
	plt.figure(tight_layout=True)
	plt.plot(range(len(all_rewards)), all_rewards, label='cumulative rewards', color='b')
	plt.xlabel('Episode',size = '14')
	plt.ylabel('Accumulated reward', size = '14')
	plt.grid(False)
	plt.xticks(size = '12')
	plt.yticks(size = '12')
	plt.savefig(r'T:\PPO\DRL-based-path-finding\PPO\PPO_visuals\ppo_training_curve.png', format='png', dpi=300)
	plt.close()

	# Save steps per episode
	plt.figure(tight_layout=True)
	plt.plot(range(len(all_steps)), all_steps, color='b')
	plt.xlabel('Episode',size = '14')
	plt.ylabel('Taken steps', size = '14')
	plt.grid(False)
	plt.xticks(size = '12')
	plt.yticks(size = '12')
	plt.savefig(r'T:\PPO\DRL-based-path-finding\PPO\PPO_visuals\ppo_steps_per_episode.png', format='png', dpi=300)
	plt.close()

	# Path visualizations with smoothing and obstacles
	def plot_path_with_obstacles(path, filename, title, env, smoothing=True):
		from matplotlib.patches import Rectangle
		path = np.array(path)
		# Ensure path starts at start point
		if not np.allclose(path[0], env.vector_state0):
			path = np.vstack([env.vector_state0, path])
		x = path[:,0]
		y = path[:,1]
		# Path smoothing (B-spline)
		if smoothing and len(path) >= 4:
			from scipy.interpolate import splprep, splev
			unique_points = list(dict.fromkeys(zip(x, y)))
			if len(unique_points) >= 4:
				x_unique, y_unique = zip(*unique_points)
				tck, u = splprep([x_unique, y_unique], s=2)
				unew = np.linspace(0, 1, max(100, len(x_unique)*3))
				x_smooth, y_smooth = splev(unew, tck)
			else:
				x_smooth, y_smooth = x, y
		else:
			x_smooth, y_smooth = x, y
		plt.figure()
		plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
		plt.plot(x_smooth, y_smooth, color='orange', linewidth=2, label='Smoothed Path (B-spline)')
		# Plot obstacles
		for i in range(len(env.Obstacle_x)):
			rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
								  env.obstacle[i][2], env.obstacle[i][3], fc='blue', ec="blue")
			plt.gca().add_patch(rectangle)
		plt.scatter(env.vector_state0[0], env.vector_state0[1], ec = 'k', c ='red', s=100, label ="Start")
		plt.scatter(env.Terminal[0], env.Terminal[1], ec = 'k', c ='red', s =100,label="Target")
		plt.grid(linestyle=':')
		plt.xlim(0,100)
		plt.ylim(0,100)
		plt.xlabel('x (m)',size = '14')
		plt.ylabel('y (m)',size = '14')
		plt.xticks(size = '12')
		plt.yticks(size = '12')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.title(title)
		plt.savefig(filename, format='png', dpi=300)
		plt.close()

	# Shortest path visualization (smoothed)
	if best_path:
		plot_path_with_obstacles(best_path, r'T:\PPO\DRL-based-path-finding\PPO\PPO_runs\ppo_shortest_path.png', 'Shortest Steps Path (Smoothed)', env)
		# Original path visualization (no smoothing)
		def plot_path_original(path, filename, title, env):
			from matplotlib.patches import Rectangle
			path = np.array(path)
			if not np.allclose(path[0], env.vector_state0):
				path = np.vstack([env.vector_state0, path])
			x = path[:,0]
			y = path[:,1]
			plt.figure()
			plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
			plt.plot(x, y, color='green', linewidth=2, label='Original Path')
			for i in range(len(env.Obstacle_x)):
				rectangle = Rectangle((10 * (env.Obstacle_x[i] - 0.5), 10 * (10 - env.Obstacle_y[i] - 0.5)), 
									  env.obstacle[i][2], env.obstacle[i][3], fc='blue', ec="blue")
				plt.gca().add_patch(rectangle)
			plt.scatter(env.vector_state0[0], env.vector_state0[1], ec = 'k', c ='red', s=100, label ="Start")
			plt.scatter(env.Terminal[0], env.Terminal[1], ec = 'k', c ='red', s =100,label="Target")
			plt.grid(linestyle=':')
			plt.xlim(0,100)
			plt.ylim(0,100)
			plt.xlabel('x (m)',size = '14')
			plt.ylabel('y (m)',size = '14')
			plt.xticks(size = '12')
			plt.yticks(size = '12')
			plt.gca().set_aspect('equal', adjustable='box')
			plt.title(title)
			plt.savefig(filename, format='png', dpi=300)
			plt.close()
		plot_path_original(best_path, r'T:\PPO\DRL-based-path-finding\PPO\PPO_runs\ppo_shortest_path_original.png', 'Shortest Steps Path (Original)', env)
	# Final path visualization (smoothed)
	if final_path:
		plot_path_with_obstacles(final_path, r'T:\PPO\DRL-based-path-finding\PPO\PPO_runs\ppo_final_path.png', 'Final Path (Smoothed)', env)
		# Final path original visualization
		plot_path_original(final_path, r'T:\PPO\DRL-based-path-finding\PPO\PPO_runs\ppo_final_path_original.png', 'Final Path (Original)', env)
	return best_path, final_path, all_rewards, all_steps


	# (Removed erroneous unsmoothed path plotting block; use plot_path_with_obstacles for path visualization)

def compute_gae(agent, rewards, states, next_states, dones, gamma=0.99, lam=0.95):
	values = agent.critic(states).detach().cpu().numpy().squeeze()
	next_values = agent.critic(torch.FloatTensor(next_states).to(agent.device)).detach().cpu().numpy().squeeze()
	returns = []
	gae = 0
	for i in reversed(range(len(rewards))):
		delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
		gae = delta + gamma * lam * (1 - dones[i]) * gae
		returns.insert(0, gae + values[i])
	return returns

def plot_and_save(data, filename, title, xlabel, ylabel):
	plt.figure()
	plt.plot(data)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(filename)
	plt.close()

def plot_path(path, filename, title):
	path = np.array(path)
	plt.figure()
	plt.plot(path[:,0], path[:,1], marker='o')
	plt.title(title)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.savefig(filename)
	plt.close()

# Interactive menu for path smoothening and animation
def interactive_menu(path):
	env = globals().get('env', None)
	# Ensure path starts at start point before smoothing
	if env and not np.allclose(path[0], env.vector_state0):
		path = np.vstack([env.vector_state0, path])
	print("Select path smoothening algorithm:")
	print("1. Moving Average")
	print("2. Spline Interpolation")
	choice = input("Enter choice (1/2): ")
	if choice == '1':
		smoothed = moving_average_path(path)
		name = 'Moving Average'
	elif choice == '2':
		smoothed = spline_smooth_path(path)
		name = 'Spline Interpolation'
	else:
		print("Invalid choice. Using original path.")
		smoothed = np.array(path)
		name = 'Original'
	animate_path(smoothed, name)

def animate_path(path, name):
	import matplotlib.patches as mpatches
	fig, ax = plt.subplots(figsize=(8, 8))
	env = globals().get('env', None)
	path = np.array(path)
	# Ensure path starts at start point
	if env and not np.allclose(path[0], env.vector_state0):
		path = np.vstack([env.vector_state0, path])
	start = env.vector_state0 if env else path[0]
	goal = env.Terminal if env else path[-1]
	for i in range(1, len(path)):
		ax.clear()
		ax.set_xticks(np.arange(0, 101, 10))
		ax.set_yticks(np.arange(0, 101, 10))
		ax.grid(True, linestyle=':', alpha=0.5)
		ax.set_xlim(0, 100)
		ax.set_ylim(0, 100)
		if env:
			for j in range(len(env.Obstacle_x)):
				rect = mpatches.Rectangle((10 * (env.Obstacle_x[j] - 0.5), 10 * (10 - env.Obstacle_y[j] - 0.5)),
										 env.obstacle[j][2], env.obstacle[j][3], fc='blue', ec='blue')
				ax.add_patch(rect)
		ax.plot(path[:i,0], path[:i,1], 'g-', linewidth=2, label='Path')
		ax.scatter(path[:i,0], path[:i,1], c='green', s=40)
		ax.scatter(start[0], start[1], c='red', s=100, label='Start', edgecolors='k', zorder=5)
		ax.scatter(goal[0], goal[1], c='orange', s=100, label='Goal', edgecolors='k', zorder=5)
		ax.set_xlabel('x (m)', size=14)
		ax.set_ylabel('y (m)', size=14)
		ax.set_title(f'PPO Path Animation - {name} | Step {i}/{len(path)-1}')
		ax.legend(loc='upper right')
		ax.set_aspect('equal', adjustable='box')
		plt.pause(0.1)
	plt.show()

# Main entry point
if __name__ == "__main__":
	initial_position, target_position = load_selected_points()
	X_max, Y_max = 10, 10
	num_actions = 8
	env = Environment(initial_position, target_position, X_max, Y_max, num_actions)
	state_dim = env.state0.size
	action_dim = num_actions
	agent = PPOAgent(state_dim, action_dim)
	best_path, final_path, rewards, steps = train_ppo(env, agent)
	# Interactive menu for path animation
	if best_path:
		interactive_menu(best_path)



