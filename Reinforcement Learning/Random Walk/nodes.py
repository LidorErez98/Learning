class Node:
    def __init__(self, state, is_terminate=False,is_start=False):
        self.state = state
        self.is_terminate = is_terminate
        self.is_start = is_start
        self.left_node = None
        self.right_node = None
    
    def __repr__(self):
        if self.is_terminate:
            return f'Node:{self.state}'
        return f'Node:{self.state}, Left: {self.left_node.state}, right: {self.right_node.state}'