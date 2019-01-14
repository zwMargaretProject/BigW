from collections import Deque

class Stack(object):
    def __init__(self):
        self.deque = Deque()
    
    def push(self, value):
        self.deque.append(value)
    
    def pop(self):
        return self.deque.pop()
    
    def __len__(self):
        return len(self.deque)
    
    def is_empty(self):
        return len(self) == 0
