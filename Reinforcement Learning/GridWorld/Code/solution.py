import matplotlib.pyplot as plt
import numpy as np

# Grid size
grid_size = 4

# Define terminal state
terminal_state = (3, 3)

# Define grid with rewards
rewards = np.full((grid_size, grid_size), -1)  # Create a matrix of -1
rewards[terminal_state] = 0  # Change terminal state reward to 0

# Initial value function
V_0 = np.zeros((grid_size, grid_size))  # V(s) = 0 for all states - initial value.
gamma = 1.0  # Discount factor
th = 1e-4  # Threshold

# Possible actions (up, down, left, right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Check if the agent arrived to the terminal state
def is_terminal(state):
    return state == (3, 3)

# Policy evaluation
def policy_evaluation(V, rewards, gamma, th):
    grid_size = V.shape[0]
    while True:
        delta = 0
        new_V = V.copy()
        
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal((i, j)):
                    continue

                v = V[i, j]
                new_v = 0

                for action in actions:
                    next_i, next_j = i + action[0], j + action[1]
                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                        new_v += (1 / len(actions)) * (rewards[next_i, next_j] + gamma * V[next_i, next_j])
                    else:
                        new_v += (1 / len(actions)) * (rewards[i, j] + gamma * V[i, j])

                new_V[i, j] = new_v
                delta = max(delta, abs(v - new_v))
        
        V = new_V
        if delta < th:
            break

    return V

# Policy improvement
def policy_improvement(V, rewards, gamma, actions):
    grid_size = V.shape[0]
    policy = np.zeros((grid_size, grid_size, len(actions)))

    for i in range(grid_size):
        for j in range(grid_size):
            if is_terminal((i, j)):
                continue

            action_values = []
            for action in actions:
                next_i, next_j = i + action[0], j + action[1]
                if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                    r = rewards[next_i, next_j] + gamma * V[next_i, next_j]
                else:
                    r = rewards[i, j] + gamma * V[i, j]
                action_values.append(r)

            best_action = np.argmax(action_values)
            policy[i, j] = np.eye(len(actions))[best_action]

    return policy

# Policy iteration
def policy_iteration(grid_size, rewards, gamma, th, actions):
    V = np.zeros((grid_size, grid_size))
    policy = np.ones((grid_size, grid_size, len(actions))) / len(actions)

    while True:
        V = policy_evaluation(V, rewards, gamma, th)
        new_policy = policy_improvement(V, rewards, gamma, actions)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return policy, V

# Run policy iteration
optimal_policy, optimal_value_function = policy_iteration(grid_size, rewards, gamma, th, actions)
print("Optimal Policy:\n", optimal_policy)
print("Optimal Value Function:\n", optimal_value_function)



# Simulate the agent's path using the optimal policy
def simulate_policy(policy, start_state):
    state = start_state
    path = [state]

    while not is_terminal(state):
        i, j = state
        action_index = np.argmax(policy[i, j])
        action = actions[action_index]
        next_state = (i + action[0], j + action[1])

        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
            state = next_state
        else:
            break  # If next state is out of bounds, break the loop
        
        path.append(state)

    return path

# Define the start state
start_state = (0, 0)

# Get the path
path = simulate_policy(optimal_policy, start_state)
print("Path taken by the agent:\n", path)

def get_action(path,path_next):
	i,j, next_i,next_j = path[0],path[1], path_next[0],path_next[1]
	if i == next_i:
		if next_j > j:
			return 'right'
		else:
			return 'left'
	if j == next_j:
		if next_i > i:
			return 'down'
		else:
			return 'up'

# Visualize the gridworld and the path
def plot_path(grid_size, path,states = ['s_0','s_4','s_8','s_9','s_{10}','s_{14}','s_{15}']):
    fig, ax = plt.subplots()

    # Draw the grid
    for i in range(grid_size + 1):
        ax.plot([i, i], [0, grid_size], color='black')
        ax.plot([0, grid_size], [i, i], color='black')

    # Highlight the path
    for i, state in enumerate(path):
        if not is_terminal(state):
            txt = f'${states[i]}$\n$\\pi({states[i]}) = {get_action(path[i], path[i+1])}$'
        else:
            txt = f'${states[i]}$'
            
        ax.text(state[1] + 0.5, grid_size - state[0] - 0.5, txt, 
                ha='center', va='center', fontsize=12, 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        
        rect = plt.Rectangle((state[1], grid_size - state[0] - 1), 1, 1, fill=True, color='blue', alpha=0.3)
        ax.add_patch(rect)

    # Highlight the terminal state
    rect = plt.Rectangle((terminal_state[1], grid_size - terminal_state[0] - 1), 1, 1, fill=True, color='red', alpha=0.3)
    ax.add_patch(rect)

    # Set the limits and remove the ticks
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add labels
    ax.set_xlabel('Gridworld')
    ax.set_title('Agent Path Using Optimal Policy')

    plt.show()

# Plot the path
plot_path(grid_size, path)
