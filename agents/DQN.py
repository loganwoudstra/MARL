import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from .agent import Agent, Buffer, Network, LFA, layer_init

T_MAX = 1_000_000
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_INTERVAL = 1000

class DuelingNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # shared feature layers (same as your original network)
        self.feature = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )
        # value stream
        self.value_stream = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        # advantage stream
        self.advantage_stream = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs):
        # Handle multi-agent batch: [batch, num_agents, obs_dim]
        is_multi_agent = obs.dim() == 3
        if is_multi_agent:
            B, N, D = obs.shape
            obs = obs.view(B * N, D)  # merge batch and agents

        features = self.feature(obs)            # [B*N, hidden_dim]
        V = self.value_stream(features)         # [B*N, 1]
        A = self.advantage_stream(features)     # [B*N, action_dim]

        Q = V + (A - A.mean(dim=-1, keepdim=True))  # [B*N, action_dim]

        if is_multi_agent:
            Q = Q.view(B, N, -1)  # reshape back to [B, N, action_dim]

        return Q
    
class DQN(Agent):
    def __init__(
        self, 
        env, 
        num_agents,
        obs_dim,
        action_dim,
        hidden_dim=256,
        lr=1e-4, 
        gamma=0.99, 
        tau=0.005,
        buffer_size=5000,
        batch_size=32,
        grad_norm_clip=5.0,
        anneal_scale=1.0, # default no annealing
        start_updating_steps=10_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
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
        self.grad_norm_clip = grad_norm_clip
        
        # # non-dueling
        # self.q_net = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        # self.target_q_net = Network(obs_dim, action_dim, hidden_dim).to(self.device)
        
        # # linear function approximation
        # self.q_net = LFA(obs_dim, action_dim).to(self.device)
        # self.target_q_net = LFA(obs_dim, action_dim).to(self.device).to(self.device)
        
        # dueling
        self.q_net = DuelingNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = DuelingNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_MAX, eta_min=lr*anneal_scale)
        
        # Experience replay buffer
        self.buffer = Buffer(buffer_size, num_agents, obs_dim)
        self.start_updating_steps = start_updating_steps
        
        # epsilon greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        
    def act(self, obs, state=None, training=True):
        if self.update_count <= self.start_updating_steps:
            actions = torch.randint(0, self.action_dim, (self.num_agents,))
        elif training and np.random.rand() < self.epsilon:
            actions = torch.randint(0, self.action_dim, (self.num_agents,))
        else:
            with torch.no_grad():
                q_vals = self.q_net(obs.to(self.device))
                actions = torch.argmax(q_vals, dim=-1)

        return actions, None, None, None
        
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
        
        # Update networks once buffer is big enough
        if self.update_count > self.start_updating_steps:
            self._update_q_network()
            self._update_target_network()
            
            # Decay epsilon
            steps = max(0, self.update_count - self.start_updating_steps)
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1.0 * steps / 100_000)

        self.update_count += 1
        
        if self.save_path is not None and self.update_count % 100 == 0:
            self.save_model()
            
    def _update_q_network(self):
        # Sample batch from buffer
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        q_vals = self.q_net(obs).gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # double dqn
            next_actions = self.q_net(next_obs).argmax(dim=-1, keepdim=True)
            next_q_target = self.target_q_net(next_obs)
            max_next_q = next_q_target.gather(2, next_actions).squeeze(-1)

            target_q = rewards + self.gamma * (1 - dones) * max_next_q

        loss = F.mse_loss(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        if self.log and self.summary_writer and self.update_count % LOG_INTERVAL == 0:
            self.summary_writer.add_scalar("losses/loss", loss.item(), self.update_count)
            self.summary_writer.add_scalar("charts/q_values_mean", q_vals.mean().item(), self.update_count)
        
    def _update_target_network(self):
        for target_param, source_param, in zip(self.target_q_net.parameters(), self.q_net.parameters()):
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
            final_path = os.path.join(PROJECT_ROOT, self.save_path, f"dqn_{self.args.layout}_seed{self.args.seed}.pth")
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            torch.save({
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },  final_path)
            # print(f"DQN model saved to {final_path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"DQN model loaded from {path}")
