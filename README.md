# DRL-based Path Finding

This repository contains implementations of various Deep Reinforcement Learning (DRL) algorithms for path finding in maze-like environments. The project demonstrates and compares different RL approaches, including DQN, PPO, Q-learning, and TD3, applied to navigation tasks.

## Project Structure

- `DQN/` - Deep Q-Network implementation and related scripts
- `PPO/` - Proximal Policy Optimization implementation
- `Q_learning/` - Tabular Q-learning implementation
- `maze_td3/` - TD3 algorithm for maze navigation
- `saved_models/` - Pre-trained models and checkpoints
- `visuals/` - Plots and visualizations of training results
- `requirements.txt` - Python dependencies

## Getting Started

### Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/0bhois53/DRL-based-path-finding.git
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Running training and evaluations
- **DQN:**
  ```powershell
  python DQN/DQN.py
  ```
- **PPO:**
  ```powershell
  python PPO/ppo.py
  ```
- **Q-learning:**
  ```powershell
  python Q_learning/Q_learning.py
  ```
- **TD3:**
  ```powershell
  python maze_td3/train_td3.py
  ```
  ```
  python maze_td3/evaluate_td3.py

## Results & Visualizations
- Training rewards and final paths are visualized in the `visuals/` and respective algorithm folders.
- Pre-trained models are available in `saved_models/`.



## License
This project is licensed under the MIT License.

## Contact
For questions or collaborations, please contact the repository owner via GitHub.
