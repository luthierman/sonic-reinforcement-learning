# don't forget to activate virtual environment
import retro
import math
import torch 
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple, deque
import cv2
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from utils import *
import replayMemory
import DQN
import sonic_env
import hyperParameters
from logger import *
from pathlib import Path
env = sonic_env.make_env("SonicTheHedgehog-Genesis","GreenHillZone.Act1")
env.reset()
print(retro.__path__)
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path(".\\checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
sonic = DQN.Agent( action_size=env.action_space.n,state_size=(4,84,84),batch_size=32, save_dir=save_dir)
logger = MetricLogger(save_dir)
episodes = 100
scores = []
scores_window = deque(maxlen=20)

for e in range(episodes):
    # episode_reward = 0
    state = env.reset()
    done = False
    
    prev_state = {}
    steps_stuck = 0
    timestamp = 0
    while not done:
        env.render()
     
        action = sonic.get_action(state)
        next_state, reward, done, info = env.step(action)
        # print(reward, info)
        sonic.store_transition(state,action,reward, next_state, done)
        loss, q = sonic.learn()
#         episode_reward += reward
        # timestamp+=1
        if loss!= None:
            logger.log_step(reward, loss, q)
        
        state = next_state
        if done:
            break
#         # print("check11")
    # scores_window.append(episode_reward)
#     scores.append(episode_reward)
#     print(episode_reward)
    sonic.save()
    logger.log_episode()
    logger.record(episode=e, epsilon=sonic.epsilon, step=sonic.counter)

# done = False
# while not done:
#     env.render()
#     action = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]    
#     s1, reward, done, info =env.step(action)
#     print("Action: {} Reward: {}".format(action, reward))
#     print("State: {}".format(s1.shape))
#     print("Reward: {}".format(reward))
#     print("Info: {}".format(info))
print(retro.__path__)