import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch.nn as nn
from collections import deque
import random

class Agent():
    def __init__(self, env, num_agents, save_path=None, log_dir=None, log=False, args=None):
        self.env = env
        self.num_agents = num_agents
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.save_path = save_path
        self.log_dir = log_dir
        self.log = log
        self.args = args
        
        if log:
            log_dir = self.log_dir

            if "SLURM_TMPDIR" in os.environ:
                log_dir = os.path.join(os.environ["SLURM_TMPDIR"], log_dir)

            os.makedirs(log_dir, exist_ok=True)

            self.log_dir = log_dir
            self.summary_writer = SummaryWriter(log_dir=log_dir, flush_secs=120)
        else:
            self.summary_writer = None
            
    def act(self, obs, state=None, training=True):
        raise NotImplementedError("Not implemented")
    
    def update(self, next_obs):
        raise NotImplementedError("Not implemented")
    
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs=None, values=None):
        raise NotImplementedError("Not implemented")
    
    def save_model(self):
        raise NotImplementedError("Not implemented")
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        # self.network = nn.Linear(obs_dim, action_dim)
    
    def forward(self, obs):
        return self.network(obs)
    
class Buffer:
    """Experience replay buffer"""
    def __init__(self, capacity, num_agents, obs_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, actions, rewards, next_obs, dones):
        """
        Args:
            obs: [num_agents, obs_dim]
            actions: [num_agents]
            rewards: [num_agents]
            next_obs: [num_agents, obs_dim]
            dones: [num_agents]
        """
        experience = (obs, actions, rewards, next_obs, dones)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        # Convert to tensors
        obs = torch.stack(obs)  # [batch_size, num_agents, obs_dim]
        actions = torch.stack(actions)  # [batch_size, num_agents]
        rewards = torch.stack(rewards)  # [batch_size, num_agents]
        next_obs = torch.stack(next_obs)  # [batch_size, num_agents, obs_dim]
        dones = torch.stack(dones)  # [batch_size, num_agents]
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)