import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from environment import final_states
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Creating class for the Q-learning table
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # List of actions
        self.actions = actions
        # Learning rate
        self.lr = learning_rate
        # Value of gamma
        self.gamma = reward_decay
        # Value of epsilon
        self.epsilon = e_greedy
        # Creating full Q-table for all cells
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # Creating Q-table for cells of the final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # Function for choosing the action for the agent
    def get_action(self, observation,epsilon):
        # Checking if the state exists in the table
        self.check_state_exist(observation)
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        sample = random.random()
        
        if sample > epsilon:
            state_action = self.q_table.loc[observation, :] 
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # Choosing random action - left 10 % for choosing randomly
            action = np.random.choice(self.actions)
        return action

    # Function for learning and updating Q-table with new knowledge
    def learn(self, state, action, reward, next_state,next_state_flag):
        self.check_state_exist(next_state) 

        # Current state in the current position 
        q_predict = self.q_table.loc[state, action]  

        # Checking if the next state is free or it is obstacle or goal
        if next_state_flag != 'goal' or next_state_flag != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict) 

        return self.q_table.loc[state, action]
    # Save the model
    def save_model(self, filepath):
        """Save the Q-tables to a file."""
        save_data = {
            'q_table': self.q_table,
            'q_table_final': self.q_table_final,
            'learning_rate': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        pd.to_pickle(save_data, filepath)
    #load the model
    def load_model(self, filepath):
        """Load the Q-tables from a file."""
        save_data = pd.read_pickle(filepath)
        self.q_table = save_data['q_table']
        self.q_table_final = save_data['q_table_final']
        self.lr = save_data['learning_rate']
        self.gamma = save_data['gamma']
        self.epsilon = save_data['epsilon']

    # Adding to the Q-table new states
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat([
                self.q_table,
                pd.DataFrame([[0]*len(self.actions)], columns=self.q_table.columns, index=[state])
            ])

    # Printing the Q-table with states
    def print_q_table(self):
        # Getting the coordinates of final route from env.py
        e = final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        for i in range(len(e)):
            state = str(e[i])  
            # Going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)
    