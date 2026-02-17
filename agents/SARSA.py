import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import random
from .agent import Agent

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim))
        )
    
    def forward(self, obs):
        return self.network(obs)
    
class SARSA(Agent):
    def __init__(
        self, 
        env, 
        num_agents,
        obs_dim,
        action_dim,
        #  state_dim,
        hidden_dim=256,
        lr=0.0005, 
        gamma=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=200,
        save_path=None, 
        log_dir=None, 
        log=False, 
        args=None
    ):
        super().__init__(env, num_agents, save_path, log_dir, log, args)
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        
        self.gamma = gamma
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        # self.state_dim = state_dim
        
        self.curr_states = None
        self.curr_actions = None
        self.curr_rewards = None
        self.curr_dones = None
        self.next_states = None
        self.next_actions = None
        
        # use target network for stability
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_update_freq = target_update_freq
        
        self.update_count = 0
        self.episode_count = 0
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def act(self, obs, state=None, training=True):
        if self.next_actions is None or not training:
            obs = obs.to(self.device)
            actions = self._get_actions(obs)
        else:
            actions = self.next_actions
        
        return actions, None, None, None # match MAPPO interface
    
    def _get_actions(self, obs, training=True):
        with torch.no_grad():
            # Process all agents at once with shared network
            q_values = self.q_network(obs)  # [num_agents, action_dim]
            
            if training:
                # Vectorized epsilon-greedy
                random_mask = torch.rand(self.num_agents, device=self.device) < self.epsilon
                random_actions = torch.randint(0, self.action_dim, (self.num_agents,), device=self.device)
                greedy_actions = q_values.argmax(dim=1)
                actions = torch.where(random_mask, random_actions, greedy_actions)
            else:
                actions = q_values.argmax(dim=1)
            
            return actions
            
            # return torch.tensor(actions, dtype=torch.long, device=self.device)
    
    def update(self, next_obs):
        self.next_states = next_obs.to(self.device)
        self._update_network()
            
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.update_count += 1
        
    def _update_network(self):
        self.next_actions = self._get_actions(self.next_states, training=True)
        curr_q_vals = self.q_network(self.curr_states)
        curr_q_sa = curr_q_vals.gather(1, self.curr_actions.unsqueeze(1)).squeeze(1)
    
        with torch.no_grad():
            next_q_vals = self.target_q_network(self.next_states)
            next_q_sa = next_q_vals.gather(
                1, self.next_actions.unsqueeze(1)
            ).squeeze(1)

            target = self.curr_rewards + self.gamma * (1 - self.curr_dones) * next_q_sa
        
        loss = F.mse_loss(curr_q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Logging
        if self.log and self.summary_writer:
            self.summary_writer.add_scalar("losses/sarsa_loss", loss.item(), self.update_count)
            self.summary_writer.add_scalar("charts/epsilon", self.epsilon, self.update_count)
            self.summary_writer.add_scalar("charts/q_values_mean", curr_q_sa.mean().item(), self.update_count)
            
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs=None, values=None):
        self.curr_states = obs.to(self.device)
        self.curr_actions = actions.to(self.device)
        self.curr_rewards = rewards.to(self.device)
        self.curr_dones = dones.to(self.device)
        
    def save_model(self):
        """Save the model"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Save shared Q-network
            torch.save(self.q_network.state_dict(), f"{self.save_path}_q_net.pth")
            
            print(f"SARSA model saved to {self.save_path}")
            
    def load_model(self, path):
        """Load the model"""
        self.q_network.load_state_dict(torch.load(f"{path}_q_net.pth"))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        print(f"SARSA model loaded from {path}")