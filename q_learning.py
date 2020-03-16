# Import libraries
import numpy as np

# Set hiperparams 
gamma = 0.75
alpha = 0.9
epochs = 1000

# PART 1 - Define the environment

# Define the states 
location_to_state = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11
}

state_to_location = {state: location for location, state, in location_to_state.items()}

# Define the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Define the rewards

R = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,1,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0]
])

# PART 2 - Build the AI solution with Q-Learning

def start_learning(R, alpha, gamma, epochs):
    # Initialize the Q-values 
    Q = np.zeros((12,12))

    # Implement the Q-learning process
    for i in range(epochs):
        current_state = np.random.randint(0, 12)
        playable_actions = [j for j in range(12) if R[current_state, j] > 0]
        next_state = np.random.choice(playable_actions)
        
        # Find the temporal difference
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD

    return Q

# PART 3 - Go to production

# Define the final function that will return the optimal route
def route(starting_location, ending_location):
    super_reward = 1000
    ending_state = location_to_state[ending_location]
    R_new = np.copy(R)
    R_new[ending_state, ending_state] = super_reward
    Q = start_learning(R_new, alpha, gamma, epochs)
    routes = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        starting_location = next_location
        routes.append(next_location)
        
    return routes
