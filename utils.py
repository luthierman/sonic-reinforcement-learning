#https://github.com/deepanshut041/Reinforcement-Learning/blob/master/algos/preprocessing/stack_frame.py
import numpy as np
import random
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque, namedtuple
import cv2
   
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed, device):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", 
        "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None])).float().to(self.device)
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)



def preprocess_frame(screen, exclude, output):
    screen = np.asarray(screen.__array__())
    # print(screen.shape)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    screen = np.ascontiguousarray(screen, dtype=np.float32) /255
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)
    return screen

def stack_frame(stacked_frames, frame, is_new):
    # print("check7")
    if is_new:
        # print("check8")
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
        # print("check9")
    else:
        # print("check10")
        stacked_frames = stacked_frames.__array__()
        # print(len(stacked_frames))
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame.__array__()
        # print("check11")
    # print("check12")
    return stacked_frames

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1,-1,-1,1), 84)
    frames = stack_frame(frames, frame, is_new)
    return frames
# Apply Wrappers to environment
