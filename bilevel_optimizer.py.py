import torch
import numpy as np
from networks import PolicyNetwork, UpperLevelModel
from lower_level import StandardPolicyOptimizer, CoupledPolicyOptimizer
from upper_level import UpperLevelOptimizer
from env_wrapper import EnvironmentWrapper
from utils import sync_networks

class BilevelOptimizer:
    def __init__(self, config):
        """Initialize the bilevel optimization system."""
        self.config = config
        
        # Create environment wrapper
        self.env_wrapper = EnvironmentWrapper(config.env_name)
        
        # Create policy networks
        self.policy_standard = PolicyNetwork(
            self.env_wrapper.state_dim, 
            self.env_wrapper.action_dim, 
            config.upper_param_dim,
            config.hidden_dim_policy
        )
        
        self.policy_coupled = PolicyNetwork(
            self.env_wrapper.state_dim, 
            self.env_wrapper.action_dim, 
            config.upper_param_dim,
            config.hidden_dim_policy
        )
        
        # Create upper level model
        self.upper_model = UpperLevelModel(
            config.upper_param_dim,
            config.hidden_dim_upper
        )
        
        # Initialize policy optimizers
        self.standard_optimizer = StandardPolicyOptimizer(
            self.policy_standard,
            lr=config.lower_lr,
            clip_ratio=config.clip_ratio,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef
        )
        
        self.coupled_optimizer = CoupledPolicyOptimizer(
            self.policy_coupled,
            lr=config.lower_lr,
            clip_ratio=config.clip_ratio,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            gradient_coupling_coef=config.gradient_coupling_coef
        )
        
        # Initialize upper level optimizer
        self.upper_optimizer = UpperLevelOptimizer(
            self.upper_model,
            self.env_wrapper,
            self.policy_standard,
            self.policy_coupled,
            self.standard_optimizer,
            self.coupled_optimizer,
            lr=config.upper_lr,
            gradient_diff_coef=config.gradient_diff_coef,
            gamma=config.gamma
        )
        
        # Sync policy networks initially
        self.sync_policy_networks()
    
    def sync_policy_networks(self):
        """Synchronize weights between policy networks."""
        sync_networks(self.policy_standard, self.policy_coupled)
    
    def train(self, iterations=None, eval_freq=None):
        """Train the bilevel optimization system."""
        if iterations is None:
            iterations = self.config.iterations
        
        if eval_freq is None:
            eval_freq = self.config.eval_freq
        
        for iteration in range(iterations):
            # Upper level update
            loss = self.upper_optimizer.update()
            
            print(f"Iteration {iteration+1}/{iterations}, Upper Loss: {loss:.4f}")
            
            # Evaluate occasionally
            if (iteration + 1) % eval_freq == 0:
                avg_reward = self.evaluate(self.config.eval_episodes)
                print(f"Evaluation: Average Reward = {avg_reward:.2f}")
                
                # Re-sync networks occasionally to prevent divergence
                if iteration < iterations - 20:  # Don't sync in final iterations
                    if (iteration + 1) % self.config.sync_frequency == 0:
                        self.sync_policy_networks()
    
    def evaluate(self, episodes=10, render=False):
        """Evaluate the current policy."""
        upper_params = self.upper_model()
        return self.env_wrapper.evaluate_policy(
            self.policy_coupled, upper_params, episodes, render)
