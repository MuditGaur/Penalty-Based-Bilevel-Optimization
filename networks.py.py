import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Policy Network (Actor-Critic)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, upper_param_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # Layers that process state and upper level parameters
        self.fc1 = nn.Linear(state_dim + upper_param_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Value function head
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, upper_params):
        # Concatenate state and upper level parameters
        x = torch.cat([state, upper_params], dim=1)
        
        # Forward pass through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy (mean and standard deviation)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        
        # Value function
        value = self.value(x)
        
        return mean, std, value
    
    def get_action(self, state, upper_params, deterministic=False):
        mean, std, _ = self.forward(state, upper_params)
        
        if deterministic:
            return mean, None
        
        # Sample from Normal distribution
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, action, upper_params):
        mean, std, value = self.forward(state, upper_params)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value


# Upper Level Parameter Model
class UpperLevelModel(nn.Module):
    def __init__(self, param_dim, hidden_dim=64):
        super(UpperLevelModel, self).__init__()
        
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, param_dim)
        
    def forward(self, x=None):
        if x is None:
            x = torch.ones(1, 1)  # Placeholder input
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
