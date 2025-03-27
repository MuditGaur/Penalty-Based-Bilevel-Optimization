import numpy as np
import torch

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lambda_gae=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_gae * (1 - dones[t]) * last_gae
        
    returns = advantages + values
    return advantages, returns


def sync_networks(source_network, target_network):
    """Synchronize weights between networks."""
    with torch.no_grad():
        for param_source, param_target in zip(
            source_network.parameters(), target_network.parameters()
        ):
            param_target.data.copy_(param_source.data)


def calculate_episode_rewards(rewards, dones):
    """Calculate rewards per episode from trajectory data."""
    episode_ends = np.where(np.append(dones, True))[0]
    episode_lengths = np.diff(np.append(-1, episode_ends))
    episode_rewards = [np.sum(rewards[i:i+l]) for i, l in zip(
        np.cumsum(np.append(0, episode_lengths[:-1])), episode_lengths)]
    
    return episode_rewards


def select_subset(states, actions, max_size=300):
    """Select a random subset of state-action pairs."""
    subset_size = min(max_size, len(states))
    subset_indices = np.random.choice(len(states), subset_size, replace=False)
    return states[subset_indices], actions[subset_indices]
