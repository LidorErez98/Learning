import matplotlib.pyplot as plt
import numpy as np

# Grid size
grid_size = 4

# Define terminal state
terminal_state = (3, 3)

# Define grid with rewards
rewards = np.full((grid_size, grid_size), -1)
rewards[terminal_state] = 0

# Create the plot
fig, ax = plt.subplots()

# Draw the grid
for i in range(grid_size + 1):
    ax.plot([i, i], [0, grid_size], color='black')
    ax.plot([0, grid_size], [i, i], color='black')

# Add state numbers and rewards
for i in range(grid_size):
    for j in range(grid_size):
        state_number = i * grid_size + j
        reward = rewards[i, j]
        ax.text(j + 0.5, grid_size - i - 0.5, f'S{state_number}\nR={reward}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Highlight terminal state
ax.add_patch(plt.Rectangle((terminal_state[1], grid_size - terminal_state[0] - 1), 1, 1, fill=None, edgecolor='red', linewidth=2))

# Set the limits and remove the ticks
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks([])
ax.set_yticks([])

# Add labels
ax.set_xlabel('Gridworld Example', fontsize=15)
ax.set_title('4x4 Gridworld with Rewards', fontsize=15)

# Save the figure
plt.show()
