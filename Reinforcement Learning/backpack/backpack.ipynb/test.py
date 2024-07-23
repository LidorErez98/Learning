import numpy as np

# Initial values
values = np.array([60, 100, 120])
weights = np.array([10, 20, 30])
num_items = len(values)
capacity = 50

th = 1e-5  # threshold
gamma = 1  # discount factor


class Item:
    def __init__(self, value, weight, name):
        self.value = value
        self.weight = weight
        self.name = name


items = [Item(values[i], weights[i], f'Item_{i}') for i in range(len(values))]  # create list of items
V = {}


def policy_eval(V, )


policy_evaluation(capacity, items, V=V)

print(V)
