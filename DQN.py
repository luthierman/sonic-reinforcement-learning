import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from utils import *
import copy
from hyperParameters import *
from itertools import count
from collections import namedtuple, deque
import random
import torch.autograd as autograd
import datetime

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        self.input_shape = input_dim
        self.num_actions = output_dim
        
        # print(self.input_shape)
        if self.input_shape[1] != 84:
            raise ValueError(f"Expecting input height: 84, got: {self.input_shape[1]}")
        if self.input_shape[2] != 84:
            raise ValueError(f"Expecting input width: 84, got: {self.input_shape[2]}")
        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, 8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr)
        
    def forward(self, x):
        
        x = self.net(x)
        x = x.view(x.size(0),-1)
        actions = self.fc(x)
        return actions
    def feature_size(self):
        return self.net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1,-1).size(1)

class Agent:
    def __init__(self, action_size,  state_size , batch_size, save_dir):
        self.action_space = action_size
        self.state_space = state_size
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.seed = random.seed(2)
        self.device = "cuda"
        
        self.buff = 10000
        self.use_cuda = torch.cuda.is_available()
        
        self.memory = ReplayBuffer(self.buff, self.batch_size, self.seed, self.device)
        self.counter =0
        
        self.save_every = 5e5
        self.gamma = GAMMA
        self.tau = TAU
        self.epsilon = EPS_START
        self.epsilon_decay = EPS_DECAY
        self.epsilon_min = EPS_END
        self.lr = LR
        self.loss_fn = nn.SmoothL1Loss()
        self.policy = Model(self.state_space, self.action_space, self.lr)
        self.target = copy.deepcopy(self.policy)
        if self.use_cuda:
            self.policy = self.policy.to(device="cuda")
            self.target = self.target.to(device="cuda")
        for p in self.target.parameters():
            p.requires_grad = False
        print("Policy:", self.policy)
        print("Target:",self.target)
        self.date = datetime.datetime.today()
        self.update_every = 1000
        self.delay = 1e3
        self.model_name = "CDQN-{date}_ADAM_lr{lr}_bs{bs}_g{g}_eps{eps}_epsmin{epsmin}_epsd{epsd}".format(
            date=self.date,
            g=self.gamma,
            bs=self.batch_size,
            lr=self.lr,
            eps=self.epsilon,
            epsmin=self.epsilon_min,
            epsd=self.epsilon_decay
        )
        
        self.sync_every = UPDATE_TARGET
        self.learn_every = LEARN_EVERY

    def store_transition(self,state,action,reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.counter+=1
  
    def get_action(self,state):
        state = torch.from_numpy(state.__array__()).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            
            action_values = self.policy.forward(state)
        if random.random() > self.epsilon:
            
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            
            action = np.random.choice(np.arange(self.action_space))
     

        return action
    def save(self):

        save_path = (
            "{}\\sonic_model_{}.chkpt".format(self.save_dir,int(self.counter//self.save_every))
        )
        torch.save(dict(model=self.policy.state_dict(),epsilon=self.epsilon), 
        save_path)
        print("SonicModel saved to {} at step {}".format(save_path, self.counter))
    
    def learn(self):
        
        if self.counter < self.delay:
            return None, None
        if self.counter %self.update_every ==0 :
            self.target.load_state_dict(self.policy.state_dict())
        
        if self.counter % self.learn_every ==0:
            self.policy.optimizer.zero_grad()
            minibatch = self.memory.sample()
            states, actions, rewards, next_states, dones = minibatch
            
            Q_eval = self.td_estimate(states, actions)
            Q_target = self.td_target( rewards, next_states, dones) 
            loss = self.loss_fn(Q_target, Q_eval)
            
            loss.backward()
            
            self.policy.optimizer.step()
            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
            return (loss.item(), Q_target.mean().item())
        else:
            return None, None
       

    def td_estimate(self,state, action):
        Q_current = self.policy(state)[np.arange(0,self.batch_size), action]
        return Q_current
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        Q_next_state = self.policy(next_state)
        best_action = torch.argmax(Q_next_state, axis=1)
        Q_next = self.target(next_state)[np.arange(0,self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma *Q_next).float()

    def sync_Q_target(self):
        self.target.load_state_dict(self.policy.state_dict())
    
