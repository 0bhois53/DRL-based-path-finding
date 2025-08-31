import numpy as np
import warnings

thr = 10 # threshold distance to the terminal for making decision 
v = 10 
obstacle_width=10

warnings.simplefilter("ignore", UserWarning)

#  coordinates for the final route
final_route = {}

class Environment(object):
  def __init__(self, initial_position, target_position,X_max, Y_max, num_actions):
    #Initial state of the system:
    self.state0 = np.zeros((2,11,11)) 
    self.state0[0][10][1] = 1 # robot initial position

    # Convert initial and target positions to grid coordinates
    start_grid_x = int(initial_position[0] / 10)
    start_grid_y = int((100 - initial_position[1]) / 10)  # Convert to grid coordinates
    target_grid_x = int(target_position[0] / 10)
    target_grid_y = int((100 - target_position[1]) / 10)  # Convert to grid coordinates
    
    print(f"Start grid position: ({start_grid_x}, {start_grid_y})")
    print(f"Target grid position: ({target_grid_x}, {target_grid_y})")

    # Generate random number of obstacles between 8 and 15
    num_obstacles = np.random.randint(8, 16)
    
    # Generate random obstacle positions, avoiding initial and target positions
    self.Obstacle_x = []
    self.Obstacle_y = []
    
    # Define safety margin around start and target (1 grid cell)
    safety_margin = 1
    
    while len(self.Obstacle_x) < num_obstacles:
        x = np.random.randint(0, 11)  # Full grid bounds
        y = np.random.randint(0, 11)
        
        # Check if position is too close to start or target
        too_close_to_start = (abs(x - start_grid_x) <= safety_margin and 
                            abs(y - start_grid_y) <= safety_margin)
        too_close_to_target = (abs(x - target_grid_x) <= safety_margin and 
                             abs(y - target_grid_y) <= safety_margin)
        
        # Only place obstacle if it's not too close to start/target and not already occupied
        if not (too_close_to_start or too_close_to_target):
            if not any((x == ox and y == oy) for ox, oy in zip(self.Obstacle_x, self.Obstacle_y)):
                self.Obstacle_x.append(x)
                self.Obstacle_y.append(y)
                print(f"Placed obstacle at grid position: ({x}, {y})")

    self.vector_obstacle_x=[0]*len(self.Obstacle_x)
    self.vector_obstacle_y=[0]*len(self.Obstacle_x)

    for i in range(len(self.Obstacle_x)):
      self.vector_obstacle_x[i]=10*(self.Obstacle_x[i]-0.5)
      self.vector_obstacle_y[i]=10*(10 - self.Obstacle_y[i] -0.5)
    
    self.obstacle =  [np.zeros((1, 4)).tolist() for i in range(len(self.Obstacle_x))]
    for i in range(len(self.vector_obstacle_x)):
      self.obstacle[i]=[self.vector_obstacle_x[i],self.vector_obstacle_y[i],obstacle_width,obstacle_width]

    
    for i in range(len(self.Obstacle_x)):
      self.state0[1, self.Obstacle_y[i], self.Obstacle_x[i]] = 1 

    self.state0[1][0][9] = 1 #the position of the Terminal
    self.X_max = X_max #range of X: X_max, the min is 0，
    self.Y_max = Y_max #range of Y: Y_max, the min is 0，
    self.vector_state0 = np.asarray(initial_position) #robot initial position, (10,10)
    self.Is_Terminal = False 
    self.vector_agentState = np.copy(self.vector_state0) # state of the agent
    self.agentState = np.copy(self.state0) # state of the agent
    self.Terminal = np.asarray(target_position)  # terminal 2
    self.doneType = 0 
    self.max_episode_steps = 10000 
    self.steps_counter = 0 
    self.num_actions = num_actions #number of actions

  # Dictionaries to draw the final route
    self.dic = {}
    self.final_path = {}
    # Key for the dictionaries
    self.index = 0
    # Writing the final dictionary first time
    self.firstsuc= True
    # Showing the steps for longest found route
    self.longest = 0
    # Showing the steps for the shortest route
    self.shortest = 0

    self.actionspace = {0:[v,0], 1:[0,v], 2: [-v,0], 3: [0,-v], 4: [-v,v], \
                      5:[-v,-v], 6:[v,v], 7: [v,-v]} #8 actions
    # self.actionspace = {0:[v,0], 1:[0,v], 2: [-v,0], 3: [0,-v]} #action space, 4 actions
    
  def reset(self): 
    self.agentState = np.copy(self.state0)
    self.vector_agentState = np.copy(self.vector_state0)
    self.dic = {}
    self.index=0
    self.doneType = 0
    self.steps_counter = 0
    self.Is_Terminal = False
    return self.agentState

  def step(self, action): #agent interact with the environment through action
    V = self.actionspace[action]
    prev_agentState = np.copy(self.vector_agentState)
    self.vector_agentState[0] += V[0]
    self.vector_agentState[1] += V[1]
    #if agent cross the boundary
    if self.vector_agentState[0] < 0:
      self.vector_agentState[0] = 0
    if self.vector_agentState[0] > 100:
      self.vector_agentState[0] = 100
    if self.vector_agentState[1] < 0:
      self.vector_agentState[1] = 0
    if self.vector_agentState[1] > 100:
      self.vector_agentState[1] = 100

    # Check for collision after move
    if self.is_collision(self.vector_agentState):
      # Revert to previous position
      self.vector_agentState = prev_agentState
      # Optionally, end episode on collision
      self.Is_Terminal = True

    # Writing in the dictionary coordinates of found route
    self.dic[self.index] = self.vector_agentState.tolist()

    # Updating key for the dictionary
    self.index += 1

    i_x = np.copy(self.vector_agentState[0])/10
    i_y = 10 - np.copy(self.vector_agentState[1])/10
    self.agentState = np.copy(self.state0)
    self.agentState[0][9][1] = 0
    self.agentState[0, int(i_y), int(i_x)] = 1
    self.steps_counter += 1
    self.Is_Terminal = self.isTerminal() # achieved the goal or not

    reward, next_state_flag = self.get_reward(self.vector_agentState, action)

    return self.agentState, next_state_flag, reward, self.Is_Terminal, None

  # function for judging whether agent achieved the goal
  def isTerminal(self):
    Distance2Terminal = np.linalg.norm(np.subtract(self.vector_agentState , self.Terminal))
    if Distance2Terminal**0.5 == 0: 
      self.doneType = 1
      return True
    else:
      return False

  #function for geting rewards
  def get_reward(self,state,action):

    reward = 0 

    # agent doesn't achieve the goal
    if not self.Is_Terminal:
       #judge whether the agent  crash the obstacle
      if self.is_collision(state):
          reward=-20
          next_state_flag = 'obstacle'
      else:
          if action==0 or action==1 or action==2 or action==3:
            reward=-1
          else:
            reward=-1.5
          next_state_flag = 'continue'

    elif self.doneType == 1:
        reward = 20
        next_state_flag = 'goal'
        # Filling the dictionary first time
        if self.firstsuc == True:
            for j in range(len(self.dic)):
                self.final_path[j] = self.dic[j]
            self.firstsuc = False
            self.longest = len(self.dic)
            self.shortest = len(self.dic)
      # Checking if the currently found route is shorter
        else:
          if len(self.dic) < len(self.final_path):
              # Saving the number of steps for the shortest route
              self.shortest = len(self.dic)
              # Clearing the dictionary for the final route
              self.final_path = {}
              # Reassigning the dictionary
              for j in range(len(self.dic)):
                  self.final_path[j] = self.dic[j] 

          # Saving the number of steps for the longest route
          if len(self.dic) > self.longest:
              self.longest = len(self.dic)
    return reward, next_state_flag 
  
  # Function to show the found route
  def final(self):
      # Showing the number of steps
      print('The shortest route:', self.shortest)
      print('The longest route:', self.longest)
      for j in range(len(self.final_path)):
          #Showing the coordinates of the final route
          #print(self.final_path[j])
          final_route[j] = self.final_path[j]

  def is_collision(self,state):
    delta = 0.5*obstacle_width
    for (x, y, w, h) in self.obstacle: 
      if 0 <= state[0] - (x - delta) <= w  \
            and 0 <= state[1] - (y - delta) <= h :
        return True

def final_states():
    return final_route  