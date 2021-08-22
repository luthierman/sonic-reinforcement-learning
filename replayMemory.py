from itertools import count
from collections import namedtuple, deque
import random
Transition = namedtuple("T", ("s1", "a", "s2", "r"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """"Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    def __str__(self):
        return str(self.memory)