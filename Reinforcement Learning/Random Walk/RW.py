import matplotlib.pyplot as plt
import numpy as np

from graph import Graph

policy = {'actions': ['left','right'],
          'probs': [0.5,0.5]}

states = {'Terminate Left':[], 
          'A':['Terminate Left','B'],
          'B':['A','C'],
          'C':['B','D'],
          'D':['C','E'],
          'E':['D','Terminate Right'],
          'Terminate Right':[]}

# Create random walk graph
random_walk = Graph(states)
random_walk.CreateGraph('C')

# Value function estimation
def ValueFunctionEstimation(states, episodes,random_walk, alpha=0.1,gamma=1):
    V = {state: 0 for state in states}
    for episode in range(episodes):
        current_node = random_walk.start_node
        while True:
            action = np.random.choice(policy['actions'], p=policy['probs'])

            if action == 'left':
                next_node = current_node.left_node
            if action == 'right':
                next_node = current_node.right_node

            r = random_walk.R(next_node.state)
            
            V[current_node.state]+=  alpha * (r + gamma*V.get(next_node.state) - V.get(current_node.state))
            
            current_node = next_node

            if current_node.is_terminate:
                break
    return V

# Get predictions
real_V = {'A':1/6,'B':2/6,'C':3/6,'D':4/6,'E':5/6} # analytic solution for V

def getPredictedValues(preds,V_hat,V):
    for key in V:
        preds[key] = preds.get(key,[]) + [V_hat[key]]


res = []
preds = {}
episodes_lst = [0,1,10,100,500]
seeds = [13,14,15,16,17]

# estimate v 
for episodes in episodes_lst:
    # change the seed to assess the model's robustness
    for seed in seeds:
        np.random.seed(seed)
        v_hat = ValueFunctionEstimation(states,episodes,random_walk)
        getPredictedValues(preds,v_hat, real_V)
    res.append(preds)
    preds = {}

# average results across the 5 seeds
for idx,preds in enumerate(res):
    for key in preds:
        preds[key] = np.mean(preds[key])
    res[idx] = preds

# Plot results
states = real_V.keys()
real_vals = real_V.values()

plt.plot(states, real_vals,linestyle='dotted', label='Real $V(s)$', linewidth=2)
for idx,v in enumerate(res):
    plt.plot(states, v.values(), linestyle='solid', label=f'{episodes_lst[idx]} Episodes', linewidth=2)

plt.legend()
plt.show()
