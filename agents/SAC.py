import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from .agent import Agent
from collections import deque
import math

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
            layer_init(nn.Linear(hidden_dim, action_dim))
        )
    
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
    
class SAC(Agent):
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
        tau=0.005,
        buffer_size=5000,
        batch_size=32,
        save_path=None, 
        log_dir=None, 
        log=False, 
        args=None
    ):
        super().__init__(env, num_agents, save_path, log_dir, log, args)
        self.gamma = gamma
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.tau = tau
        self.batch_size = batch_size
        self.update_count = 0
        self.episode_count = 0
        
        self.critic1 = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        
        self.target_critic1 = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.actor = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Experience replay buffer
        self.buffer = Buffer(buffer_size, num_agents, obs_dim)
        
        # tempature
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_alpha = -math.log(self.action_dim)
        
    def act(self, obs, state=None, training=True):
        with torch.no_grad():
            obs_batch = obs.to(self.device) # [num_agents, obs_dim]
            logits = self.actor(obs_batch) # [num_agents, action_dim]
            if training:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
        return actions, None, None, None # match MAPPO interface
        
    def update(self, next_obs):
        obs = self.current_obs  # [num_agents, obs_dim]
        actions = self.current_actions  # [num_agents]
        rewards = self.current_rewards  # [num_agents]
        dones = self.current_dones  # [num_agents]
        next_obs_reshaped = next_obs  # [num_agents, obs_dim]
        
        # Add experience to buffer (state is computed from obs in buffer.sample())
        self.buffer.add(
            obs.cpu(),
            actions.cpu(),
            rewards.cpu(),
            next_obs_reshaped.cpu(),
            dones.cpu()
        )
        
        # Update networks if we have enough samples
        if len(self.buffer) > self.batch_size:
            self._update_ac_networks()
            self._update_target_networks()
        

        self.update_count += 1
        
    def _update_ac_networks(self):
        """update both critics, actor, and tempature"""
        # Sample batch from buffer
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        # distribution is used for actor and tempature
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        # q-values used for critics and actor update
        q1_vals_all = self.critic1(obs)
        q2_vals_all = self.critic2(obs)
        
        self._critic_update(q1_vals_all, q2_vals_all, actions, rewards, next_obs, dones)
        self._actor_update(q1_vals_all, q2_vals_all, dist)
        self._alpha_update(dist)
        
    def _critic_update(self, q1_vals_all, q2_vals_all, actions, rewards, next_obs, dones):
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_dist = torch.distributions.Categorical(logits=next_logits)
            next_probs = next_dist.probs
            next_log_probs = next_dist.logits - torch.logsumexp(next_dist.logits, dim=-1, keepdim=True)
        
            next_q1_vals = self.critic1(next_obs)
            next_q2_vals = self.critic2(next_obs)
            next_min_q = torch.min(next_q1_vals, next_q2_vals)
            
            next_v = (next_probs * (next_min_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q_vals = rewards + self.gamma * (1 - dones) * next_v
            
        q1_vals = q1_vals_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_vals = q2_vals_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        critic1_loss = F.mse_loss(q1_vals, target_q_vals)
        critic2_loss = F.mse_loss(q2_vals, target_q_vals)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        if self.log and self.summary_writer:
            self.summary_writer.add_scalar("losses/critic1_loss", critic1_loss.item(), self.update_count)
            self.summary_writer.add_scalar("losses/critic2_loss", critic2_loss.item(), self.update_count)
            self.summary_writer.add_scalar("charts/q1_values_mean", q1_vals.mean().item(), self.update_count)
            self.summary_writer.add_scalar("charts/q1_values_mean", q1_vals.mean().item(), self.update_count)
    
    def _actor_update(self, q1_vals_all, q2_vals_all, dist):  
        probs = dist.probs
        log_probs = dist.logits - torch.logsumexp(dist.logits, dim=-1, keepdim=True)
        
        min_q = torch.min(q1_vals_all, q2_vals_all).detach()
        
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.log and self.summary_writer:
            self.summary_writer.add_scalar("losses/actor_loss", actor_loss.item(), self.update_count)
    
    def _alpha_update(self, dist):
        entropy = dist.entropy().mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_alpha).detach())
        self.alpha_optimizer.zero_grad()
        
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        if self.log and self.summary_writer:
            self.summary_writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.update_count)
            self.summary_writer.add_scalar("charts/alpha", self.alpha, self.update_count)
        
    def _update_target_networks(self):
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
    def _soft_update(self, target, source):
        for target_param, source_param, in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
        
 
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs=None, values=None):
        """Add experience to buffer (compatibility with existing interface)"""
        # Assuming num_envs = 1, so we store the experience directly
        self.current_obs = obs  # [num_agents, obs_dim]
        self.current_actions = actions  # [num_agents]
        self.current_rewards = rewards  # [num_agents]
        self.current_dones = dones  # [num_agents]
    
    def save_model(self):
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'target_critic1_state_dict': self.target_critic1.state_dict(),
                'target_critic2_state_dict': self.target_critic2.state_dict(),
                'log_alpha': self.log_alpha,
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic1_optimizer': self.critic1_optimizer.state_dict(),
                'critic2_optimizer': self.critic2_optimizer.state_dict(),
                'alpha_optimizer': self.alpha_optimizer.state_dict(),
            },  f"{self.save_path}_sac_full.pth")
            print(f"SAC model saved to {self.save_path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load actor and critics
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        # Load temperature
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        print(f"SAC model loaded from {path}")