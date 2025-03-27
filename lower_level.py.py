import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LowerLevelOptimizer:
    def __init__(self, policy, lr=3e-4, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """Base class for lower level optimizers."""
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def prepare_batch_data(self, data):
        """Convert trajectory data to tensors and prepare for batch processing."""
        states = torch.FloatTensor(data['states'])
        actions = torch.FloatTensor(data['actions'])
        old_log_probs = torch.FloatTensor(data['log_probs']).unsqueeze(1)
        advantages = torch.FloatTensor(data['advantages']).unsqueeze(1)
        returns = torch.FloatTensor(data['returns']).unsqueeze(1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, advantages, returns


class StandardPolicyOptimizer(LowerLevelOptimizer):
    def update(self, upper_params, data, epochs=10, batch_size=64):
        """Update the standard lower level policy using PPO without coupling."""
        # Prepare batch data
        states, actions, old_log_probs, advantages, returns = self.prepare_batch_data(data)
        
        # PPO update epochs
        for _ in range(epochs):
            # Generate random indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + batch_size, len(states))
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Expand upper_params for the batch
                batch_size = len(mb_indices)
                mb_upper_params = upper_params.expand(batch_size, -1)
                
                # Forward pass
                new_log_probs, entropy, values = self.policy.evaluate(mb_states, mb_actions, mb_upper_params)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss1 = mb_advantages * ratio
                policy_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss (standard PPO loss)
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Calculate average episode reward
        episode_ends = np.where(np.append(data['dones'], True))[0]
        episode_lengths = np.diff(np.append(-1, episode_ends))
        episode_rewards = [np.sum(data['rewards'][i:i+l]) for i, l in zip(
            np.cumsum(np.append(0, episode_lengths[:-1])), episode_lengths)]
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        return -avg_reward  # Negate for minimization


class CoupledPolicyOptimizer(LowerLevelOptimizer):
    def __init__(self, policy, lr=3e-4, clip_ratio=0.2, value_coef=0.5, 
                 entropy_coef=0.01, gradient_coupling_coef=0.1):
        """Lower level optimizer with gradient coupling from upper level."""
        super().__init__(policy, lr, clip_ratio, value_coef, entropy_coef)
        self.gradient_coupling_coef = gradient_coupling_coef
    
    def compute_upper_loss_gradient(self, upper_params, states, actions):
        """Compute gradient of upper loss with respect to policy parameters."""
        batch_size = states.size(0)
        mb_upper_params = upper_params.expand(batch_size, -1)
        
        # Create a small test trajectory to compute upper loss
        with torch.enable_grad():
            # Make sure we track gradients
            new_log_probs, entropy, values = self.policy.evaluate(states, actions, mb_upper_params)
            
            # Simple proxy for upper loss: negative expected returns
            dummy_upper_loss = -values.mean()
            
            # Compute gradients of upper loss w.r.t policy parameters
            upper_loss_gradients = []
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            dummy_upper_loss.backward(retain_graph=True)
            
            for param in self.policy.parameters():
                if param.grad is not None:
                    upper_loss_gradients.append(param.grad.clone())
                else:
                    upper_loss_gradients.append(torch.zeros_like(param))
            
            # Zero out gradients for clean slate
            self.optimizer.zero_grad()
            
        return upper_loss_gradients
    
    def update(self, upper_params, data, epochs=10, batch_size=64):
        """Update policy using PPO with coupling to upper level."""
        # Prepare batch data
        states, actions, old_log_probs, advantages, returns = self.prepare_batch_data(data)
        
        # Pre-compute upper loss gradients using a subset of data
        subset_size = min(500, len(states))
        subset_indices = np.random.choice(len(states), subset_size, replace=False)
        subset_states = states[subset_indices]
        subset_actions = actions[subset_indices]
        upper_loss_gradients = self.compute_upper_loss_gradient(upper_params, subset_states, subset_actions)
        
        # PPO update epochs
        for _ in range(epochs):
            # Generate random indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + batch_size, len(states))
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Expand upper_params for the batch
                batch_size = len(mb_indices)
                mb_upper_params = upper_params.expand(batch_size, -1)
                
                # Forward pass
                new_log_probs, entropy, values = self.policy.evaluate(mb_states, mb_actions, mb_upper_params)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss1 = mb_advantages * ratio
                policy_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss (standard PPO loss)
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Compute gradients for standard loss
                self.optimizer.zero_grad()
                loss.backward()
                
                # Add gradient from upper level loss to lower level gradients
                with torch.no_grad():
                    for i, param in enumerate(self.policy.parameters()):
                        if param.grad is not None and i < len(upper_loss_gradients):
                            # Add a constant multiple of the upper loss gradient
                            param.grad += self.gradient_coupling_coef * upper_loss_gradients[i]
                
                # Apply clipping and parameter update
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Calculate average episode reward
        episode_ends = np.where(np.append(data['dones'], True))[0]
        episode_lengths = np.diff(np.append(-1, episode_ends))
        episode_rewards = [np.sum(data['rewards'][i:i+l]) for i, l in zip(
            np.cumsum(np.append(0, episode_lengths[:-1])), episode_lengths)]
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        return -avg_reward  # Negate for minimization
