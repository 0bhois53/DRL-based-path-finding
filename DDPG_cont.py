import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from Environment2 import Env
import os


# HER-enabled replay buffer using 'future' strategy per episode

class HERReplayBuffer:
	def __init__(self, max_size=200000, her_k=4):
		self.max_size = int(max_size)
		self.storage = deque(maxlen=self.max_size)
		self.her_k = int(her_k) # number of future goals to sample per transition
		self.episode_buffer = [] # temporary per-episode buffer to apply HER at episode end
		self.success_tol = 1.5  # Default tolerance for success

	def _add_transition(self, transition):
		# transition: (s, a, r, s', done, goal)
		self.storage.append(transition)

	def add_episode_transition(self, transition):
		# store into episode buffer; will be committed at episode end or periodically
		self.episode_buffer.append(transition)

	def commit_episode(self):
		# apply 'future' HER relabeling for the stored episode transitions
		ep = self.episode_buffer
		L = len(ep)
		for idx, trans in enumerate(ep):
			s, a, r, s2, done, goal = trans
			# add original
			self._add_transition((s, a, r, s2, done, goal))
			# sample up to her_k future achieved goals
			for _ in range(self.her_k):
				future_idx = np.random.randint(idx+1, L)
				# choose the achieved goal as the agent position at future_idx's next state
				achieved = ep[future_idx][3] # s' of future transition
				new_goal = achieved[:2].copy() # assume s is [ax,ay,gx,gy]
				# recompute reward for new goal (sparse): success if next_state within tol
				achieved_dist = np.linalg.norm(s2[:2] - new_goal)
				new_r = 100.0 if achieved_dist <= self.success_tol else (-achieved_dist*0.1 - 1.0)
				# relabel states: replace goal part of s and s2
				s_rel = s.copy()
				s_rel[2:4] = new_goal
				s2_rel = s2.copy()
				s2_rel[2:4] = new_goal
				self._add_transition((s_rel, a, new_r, s2_rel, done, new_goal))
		# clear episode buffer
		self.episode_buffer = []

	def add(self, transition):
		# direct add (if you want to bypass HER)
		self._add_transition(transition)

	def sample(self, batch_size):
		batch = random.sample(self.storage, batch_size)
		state, action, reward, next_state, done, goal = zip(*batch)
		state = torch.FloatTensor(np.array(state))
		action = torch.FloatTensor(np.array(action))
		reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
		next_state = torch.FloatTensor(np.array(next_state))
		done = torch.FloatTensor(np.array(done)).unsqueeze(1)
		goal = torch.FloatTensor(np.array(goal))
		return state, action, reward, next_state, done, goal
# networks
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden=128):
		super(Actor, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, action_dim),
			nn.Tanh()
		)
	def forward(self, x):
		return self.net(x)

class CriticTwin(nn.Module):
	def __init__(self, state_dim, action_dim, hidden=128):
		super(CriticTwin, self).__init__()
		# Q1
		self.q1 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, 1)
		)
		# Q2
		self.q2 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, 1)
		)
	def forward(self, state, action):
		sa = torch.cat([state, action], dim=1)
		return self.q1(sa), self.q2(sa)

class DDPGAgent:
	def __init__(self, state_dim, action_dim, max_action=0.5, device=None, success_tol=1.5):
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.action_dim = action_dim
		self.max_action = float(max_action)
		self.actor = Actor(state_dim, action_dim).to(self.device)
		self.actor_target = Actor(state_dim, action_dim).to(self.device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
		self.critic = CriticTwin(state_dim, action_dim).to(self.device)
		self.critic_target = CriticTwin(state_dim, action_dim).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
		self.replay = HERReplayBuffer(max_size=200000)
		self.gamma = 0.99
		self.tau = 1e-3
		self.noise_std = 0.3
		self.noise_decay = 1e-5
		self.success_tol = success_tol

	def select_action(self, state, noise=True):
		state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		with torch.no_grad():
			a = self.actor(state_t).cpu().numpy().flatten()
		if noise:
			a = a + np.random.normal(0, self.noise_std, size=a.shape)
		a = np.clip(a, -1.0, 1.0) * self.max_action
		# decay noise
		self.noise_std = max(0.02, self.noise_std * (1.0 - self.noise_decay))
		return a

	def train(self, batch_size=64, policy_noise=0.2, noise_clip=0.2, policy_freq=2):
		if len(self.replay.storage) < batch_size:
			return
		state, action, reward, next_state, done, goal = self.replay.sample(batch_size)
		state = state.to(self.device)
		action = action.to(self.device)
		reward = reward.to(self.device)
		next_state = next_state.to(self.device)
		done = done.to(self.device)
		# goal is available if needed for HER
		with torch.no_grad():
			# target action smoothing
			noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0) * self.max_action
			q1_t, q2_t = self.critic_target(next_state, next_action)
			target_q = torch.min(q1_t, q2_t)
			target_q = reward + (1 - done) * self.gamma * target_q
		# current Q estimates
		current_q1, current_q2 = self.critic(state, action)
		critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
		self.critic_opt.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
		self.critic_opt.step()
		# actor update
		actor_loss = -self.critic.q1(torch.cat([state, self.actor(state) * self.max_action], dim=1)).mean() if hasattr(self.critic, 'q1') else self.critic(state, self.actor(state) * self.max_action)[0].mean()
		self.actor_opt.zero_grad()
		actor_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
		self.actor_opt.step()
		# soft update
		for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
			tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
		for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
			tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

	def save(self, fname):
		torch.save({
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
		}, fname)

	def load(self, fname):
		ckpt = torch.load(fname, map_location=self.device)
		self.actor.load_state_dict(ckpt['actor'])
		self.critic.load_state_dict(ckpt['critic'])

def train_loop(episodes=500, max_steps=200, render=False):
	# Load start and goal from selected_points.txt if available
	start_pos, goal_pos = None, None
	if os.path.exists('selected_points.txt'):
		with open('selected_points.txt', 'r') as f:
			lines = f.readlines()
			if len(lines) >= 2:
				start_pos = [float(x) for x in lines[0].strip().split(',')]
				goal_pos = [float(x) for x in lines[1].strip().split(',')]
	env = Env(default_start=start_pos if start_pos else (1.0,1.0), default_goal=goal_pos if goal_pos else (9.0,9.0))
	env.reset(randomize_obstacles=True)
	print("Random Obstacles:", env.obstacles)
	state_dim = env.reset().shape[0]
	action_dim = 2
	agent = DDPGAgent(state_dim, action_dim, max_action=0.5, success_tol=env.success_tol)
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	batch_size = 64
	total_steps = 0
	success_history = []
	cumulative_rewards = []
	steps_per_episode = []
	final_path_x = []
	final_path_y = []
	shortest_path_x = [env.default_start[0]]
	shortest_path_y = [env.default_start[1]]
	for ep in range(episodes):
		obs = env.reset()
		ep_return = 0.0
		agent.replay.episode_buffer = []  # reset episode buffer
		visited_x = [obs[0]]
		visited_y = [obs[1]]
		for t in range(max_steps):
			action = agent.select_action(obs, noise=True)
			next_obs, flag, reward, done, _ = env.step(action)
			goal = env.Terminal.copy() if hasattr(env, 'Terminal') else None
			agent.replay.add_episode_transition((obs, action, reward, next_obs, float(done), goal))
			agent.train(batch_size=batch_size)
			obs = next_obs
			ep_return += reward
			total_steps += 1
			visited_x.append(obs[0])
			visited_y.append(obs[1])
			if done:
				success = 1 if flag == 'goal' else 0
				success_history.append(success)
				break
		# Commit HER episode transitions at episode end
		agent.replay.commit_episode()
		cumulative_rewards.append(ep_return)
		steps_per_episode.append(len(visited_x)-1)
		# Save final path for last episode
		if ep == episodes-1:
			final_path_x = visited_x.copy()
			final_path_y = visited_y.copy()
		# Save shortest path (straight line) for visualization
		if ep == 0:
			shortest_path_x = [env.default_start[0], env.Terminal[0]]
			shortest_path_y = [env.default_start[1], env.Terminal[1]]
		if (ep + 1) % 10 == 0:
			avg_success = np.mean(success_history[-50:]) if len(success_history) > 0 else 0.0
			print(f"EP {ep+1} | return {ep_return:.2f} | recent_success {avg_success:.3f} | noise {agent.noise_std:.3f}")
		# optional: save model periodically
		if (ep + 1) % 200 == 0:
			agent.save(f"ddpg_checkpoint_ep{ep+1}.pt")
	# --- Visualization ---
	plt.figure(tight_layout=True)
	plt.plot(range(episodes), cumulative_rewards, label='Cumulative Rewards', color='b')
	plt.xlabel('Episode', size=14)
	plt.ylabel('Accumulated Reward', size=14)
	plt.title('DDPG Accumulated Rewards')
	plt.grid(True)
	plt.xticks(size=12)
	plt.yticks(size=12)
	plt.legend()
	plt.show()

	plt.figure(tight_layout=True)
	plt.plot(range(episodes), steps_per_episode, color='g')
	plt.xlabel('Episode', size=14)
	plt.ylabel('Steps Taken', size=14)
	plt.title('DDPG Steps per Episode')
	plt.grid(True)
	plt.xticks(size=12)
	plt.yticks(size=12)
	plt.show()

	# Final path visualization
	plt.figure(figsize=(8, 8))
	plt.plot(final_path_x, final_path_y, 'r-', linewidth=2, label='Final Path')
	plt.scatter(env.default_start[0], env.default_start[1], c='green', s=100, label='Start')
	plt.scatter(env.Terminal[0], env.Terminal[1], c='red', s=100, label='Goal')
	# Plot obstacles if available
	if hasattr(env, 'obstacles') and env.obstacles:
		for obs in env.obstacles:
			if len(obs) == 4:
				plt.gca().add_patch(Rectangle((obs[0], obs[1]), obs[2], obs[3], fc='blue', ec='blue'))
	plt.xlabel('x (m)', size=14)
	plt.ylabel('y (m)', size=14)
	plt.title('DDPG Final Path')
	plt.grid(True)
	plt.legend()
	plt.xlim(0, env.X_max)
	plt.ylim(0, env.Y_max)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

	# Shortest path visualization (straight line)
	plt.figure(figsize=(8, 8))
	plt.plot(shortest_path_x, shortest_path_y, 'b--', linewidth=2, label='Shortest Path')
	plt.scatter(env.default_start[0], env.default_start[1], c='green', s=100, label='Start')
	plt.scatter(env.Terminal[0], env.Terminal[1], c='red', s=100, label='Goal')
	plt.xlabel('x (m)', size=14)
	plt.ylabel('y (m)', size=14)
	plt.title('DDPG Shortest Path')
	plt.grid(True)
	plt.legend()
	plt.xlim(0, env.X_max)
	plt.ylim(0, env.Y_max)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()
	return agent

if __name__ == '__main__':
	agent = train_loop(episodes=500)