import torch
from torch.utils.tensorboard import SummaryWriter

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
            self.summary_writer = SummaryWriter(log_dir=log_dir)
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