from nodes import Node

class Graph:
    def __init__(self,states):
        """
        gets a dictionary of nodes and their directions
        """
        self.states = states
        self.start_node = None
        self.nodes = {}

    def CreateNodes(self, s0):
        is_terminate = False
        is_start = False
        for state in self.states:

            is_start = state == s0
            is_terminate = not self.states.get(state)

            node = Node(state,is_terminate, is_start)

            if is_start:
                self.start_node = node

            self.nodes[node.state] = node

    def CreateGraph(self,s0):
        if not self.nodes:
            self.CreateNodes(s0)
        for state, directions in self.states.items():
            if directions:
                self.nodes[state].left_node = self.nodes[directions[0]]
                self.nodes[state].right_node = self.nodes[directions[1]]
    
    def R(self,state):
        if state.lower() == 'terminate right':
            return 1.0
        return 0.0