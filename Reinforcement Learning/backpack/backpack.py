import numpy as np
import matplotlib.pyplot as plt


# Initial values
values = np.array([60,100,120])
weights = np.array([10,20,30])
num_items = len(values)
capacity = 50

th = 1e-5 # threshold
gamma = 1 # discount factor

# Reward Function
def R(action, capacity,  weight, value):
	if action == 1:
		if capacity - weight >= 0:
			return value
	return 0


def is_terminal(state, num_items):
	capacity, item_idx = state
	return capacity <= 0 or item_idx >= num_items

def knapsack_policy_evaluation(V, policy, capacity, num_items, gamma=gamma, th=th, epsilon =0.1, epsilon_decay=0.95):
	print('Starting Policy Evaluation...')

	iter = 1

	while True:
		delta  = 0
		new_V = V.copy()
		for cap in range(capacity + 1):
			items_long_run_values = []
			for idx in range(num_items):
				state = (cap,idx)

				if is_terminal(state, num_items):
					continue

				v = V[state]
				new_v = 0

				# Explore
				if np.random.uniform(0,1) < epsilon:

					action = np.random.choice([0,1])
					prob = 1.0
					r = R(action, state[0], weights[state[1]], values[state[1]])

					if r > 0:
						next_state = (state[0] - weights[state[1]], state[1] + 1)
					else:
						next_state = (state[0], state[1] + 1)

					next_v = V[next_state] if next_state[1] < num_items else 0

					new_v += prob * (r+gamma*next_v)

				# Exploit
				else:
					for action, prob in enumerate(policy[cap,idx]):
						r = R(action, state[0], weights[state[1]], values[state[1]])


						if r > 0:
							next_state = (state[0] - weights[state[1]], state[1] + 1)
						else:
							next_state = (state[0], state[1] + 1)

						next_v = V[next_state] if next_state[1] < num_items else 0


						# Compute new value
						new_v += prob*(r + gamma * next_v)

				new_V[state] = new_v

				delta = max(delta, abs(v-new_v)) # convergence condition

		if iter % 10 == 0:
			print(f'Number of Iterations: {iter}, Delta: {delta}')

		V = new_V
		iter += 1
		epsilon = max(epsilon * epsilon_decay, 0.01) # decay epsilon

		if delta < th:
			break
	print(f'The policy evaluation terminated after {iter} iterations')
	return V



def policy_improvement(V,capacity, num_items, gamma=gamma, actions= [0,1]):
	print("Starting Policy Improvement...")
	policy = np.ones((capacity + 1, num_items, len(actions)))/len(actions)

	for cap in range(capacity + 1):
		for item_idx in range(num_items):

			state = (cap,item_idx)

			if is_terminal(state, num_items):
				continue

			action_values = []

			for action in actions:

				r = R(action, cap, weights[item_idx], values[item_idx])
				next_v = 0

				if r > 0:
					next_state = (cap - weights[item_idx], item_idx+1)
					next_v = V[next_state] if next_state[1] < num_items else 0
				else:
					next_state = (cap, item_idx+1)

				action_value = r + gamma * next_v
				action_values.append(action_value)

			exp_vals = np.exp(action_values - np.max(action_values))

			probs = exp_vals / np.sum(exp_vals)

			policy[state] = probs

	return policy


def policy_iteration(capacity,num_items, gamma=gamma,th=th,actions = [0,1]):
	V = np.zeros((capacity + 1, num_items)) # for each item there are two possible actions: include or exclude. 
	policy = np.ones((capacity + 1, num_items, len(actions))) / len(actions) # random policy


	while True:
		V = knapsack_policy_evaluation(V, policy, capacity, num_items)
		optimal_policy = policy_improvement(V, capacity, num_items)

		if np.array_equal(optimal_policy, policy): # check if policy is stable.
			break

		policy = optimal_policy

	return policy, V

optimal_policy, optimal_value_function = policy_iteration(capacity,num_items)
print("Optimal Policy:\n", optimal_policy)
print("Optimal Value Function:\n", optimal_value_function)




