import torch
import torch.optim as optim
import numpy as np

class UpperLevelOptimizer:
    def __init__(self, upper_model, env_wrapper, standard_policy, coupled_policy,
                 standard_optimizer, coupled_optimizer, 
                 lr=1e-4, gradient_diff_coef=0.05, gamma=0.99):
        """
        Upper level optimizer that manages the bilevel optimization process.
        
        Args:
            upper_model: The upper level parameter model
            env_wrapper: Environment wrapper for collecting trajectories
            standard_policy: Standard policy network
            coupled_policy: Coupled policy network
            standard_optimizer: Optimizer for standard policy
            coupled_optimizer: Optimizer for coupled policy
            lr: Learning rate for upper level
            gradient_diff_coef: Coefficient for gradient difference regularization
            gamma: Discount factor for rewards
        """
        self.upper_model = upper_model
        self.env_wrapper = env_wrapper
        self.standard_policy = standard_policy
        self.coupled_policy = coupled_policy
        self.standard_optimizer = standard_optimizer
        self.coupled_optimizer = coupled_optimizer
        self.optimizer = optim.Adam(upper_model.parameters(), lr=lr)
        self.gradient_diff_coef = gradient_diff_coef
        self.gamma = gamma
    
    def compute_trajectory_gradient(self, policy, upper_params, num_trajectories=3, max_steps=1000):
        """
        Compute gradient of lower level loss w.r.t upper level parameters using 
        a discounted sum of reward gradients along trajectories.
        """
        # Sample trajectories
        trajectories = self.env_wrapper.sample_full_trajectories(
            policy, upper_params, num_trajectories, max_steps)
        
        cumulative_gradient = torch.zeros_like(upper_params)
        
        for trajectory in trajectories:
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            log_probs = trajectory['log_probs']
            upper_param_grads = trajectory['upper_param_grads']
            
            # Calculate discounted returns (G_t)
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            
            # Normalize returns for stability
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute gradients for each step in the trajectory
            for t in range(len(states)):
                # Extract stored log_prob and its corresponding upper_params
                log_prob = log_probs[t]
                step_upper_params = upper_param_grads[t]
                
                # Compute gradient of log_prob with respect to upper_params
                (-log_prob * returns[t]).backward()
                
                # Extract and accumulate the gradient
                if step_upper_params.grad is not None:
                    # Weight the gradient by the discounted return
                    cumulative_gradient += step_upper_params.grad.clone() * (self.gamma ** t)
            
        # Average the gradient over all trajectories
        cumulative_gradient /= num_trajectories
        
        return cumulative_gradient
    
    def update(self):
        """Update the upper level parameters with gradient difference regularization."""
        # Get upper level parameters
        upper_params = self.upper_model()
        
        # Collect data with current upper parameters
        data = self.env_wrapper.collect_trajectory(self.standard_policy, upper_params)
        
        # Update both policies using different methods
        self.standard_optimizer.update(upper_params, data)
        upper_loss = self.coupled_optimizer.update(upper_params, data)
        
        # Compute gradients of lower level loss w.r.t upper params using trajectories
        grad_standard = self.compute_trajectory_gradient(
            self.standard_policy, upper_params, num_trajectories=3)
        grad_coupled = self.compute_trajectory_gradient(
            self.coupled_policy, upper_params, num_trajectories=3)
        
        # Compute absolute difference between the gradients
        grad_diff = torch.abs(grad_standard - grad_coupled)
        
        # Prepare for upper level update
        self.optimizer.zero_grad()
        
        # Create a tensor that requires grad
        upper_loss_tensor = torch.tensor(upper_loss, requires_grad=True)
        upper_loss_tensor.backward()
        
        # Add gradient difference term to upper level gradients
        with torch.no_grad():
            for param in self.upper_model.parameters():
                if param.grad is not None:
                    # Add scalar multiple of gradient difference
                    param.grad += self.gradient_diff_coef * grad_diff
        
        # Apply the update
        self.optimizer.step()
        
        return upper_loss
