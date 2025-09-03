# TD3/DDPG Maze Pathfinding

This project implements a TD3/DDPG agent for continuous pathfinding in a 10x10 grid maze using Stable Baselines3 and Gymnasium.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the agent:
   ```bash
   python train_td3.py
   ```
3. Evaluate and visualize:
   ```bash
   python evaluate_td3.py
   ```

## Files
- `maze_env.py`: Custom Gymnasium environment for the maze.
- `train_td3.py`: Training script for TD3/DDPG agent.
- `evaluate_td3.py`: Evaluation and visualization script.
- `requirements.txt`: Python dependencies.
- `saved_models/`: Directory for saved models.
- `visuals/`: Directory for plots and visualizations.
