import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def R(s,a):
    """
    s: the state of the environment
    a: the action of the agent. Can be 'buy' or 'sell'
    """
    if s > 0 and a == 'sell': # selling
        return 1, s-1

    if a == 'buy':
        if s == 9:
            return 100, 10 # full stock
        if s < 9: # restocking
            return 0, s+1
    return 0, s

def Q(s,a,gamma, H,stop):
    if H > stop:
        return 0

    if gamma == 0:
        return R(s,a)[0]

    reward, next_state = R(s,a)
    H +=1

    return reward + gamma*max(Q(next_state,'sell',gamma,H,stop),Q(next_state,'buy',gamma,H,stop))



gammas = [0,0.3,0.7]
stop = 20
def episode(H,s,gamma,stop):

    rewards = []
    while H <= stop:
        q_buy = Q(s,'buy',gamma,H,stop)
        q_sell = Q(s,'sell',gamma,H,stop)
        if q_buy > q_sell:
            a = 'buy'
        else:
            a = 'sell'

        item = R(s,a)
        rewards.append(item[0])
        s = item[1]

        H += 1
    print(f'Episode is over. Gamma:{gamma} Reward:{np.sum(rewards)}')
    return np.array(rewards)

timesteps = np.arange(1,stop+1)
res = {}
immediate_res = {}
for gamma in gammas:
    s = 3 # s_3
    H = 1 # first time step
    r = episode(H,s,gamma,stop)
    immediate_res[gamma] = r
    res[gamma] = np.cumsum(r) # cumulative reward.


# Plotting cumulative rewards
plt.figure(figsize=(12, 6))
for gamma in res:
    sns.lineplot(x=timesteps, y=res[gamma], label=f'Gamma: {gamma}', linewidth=1.5)

plt.title('Cumulative Reward Over Time', fontsize=18)
plt.xlabel('Time Step')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
plt.show()

# Plotting immediate rewards
plt.figure(figsize=(12, 6))
for gamma in immediate_res:
    sns.lineplot(x=timesteps, y=immediate_res[gamma], label=f'Gamma: {gamma}', linewidth=1.5)

plt.title('Immediate Reward Over Time', fontsize=18)
plt.xlabel('Time Step')
plt.ylabel('Immediate Reward')
plt.legend()
plt.grid(True)
plt.show()









