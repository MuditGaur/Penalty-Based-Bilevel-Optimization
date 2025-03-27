import numpy as np
import torch
import gym
from utils import compute_gae

class EnvironmentWrapper:
    def __init__(self, env_name):
        """Initialize the environment wrapper."""
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
    def collect_trajectory(self, policy, upper_params, num_steps=2048, gamma=0.99, lambda_gae=0.95):
        """Collect experience using the specified policy."""
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = self.env.reset()
        done = False
        
        for _ in range(num_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                action, log_prob = policy.get_action(state_tensor, upper_params)
                _, _, value = policy.evaluate(state_tensor, action, upper_params)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action.squeeze(0).numpy())
            
            # Store data
            states.append(state)
            actions.append(action.squeeze(0).numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            # Update state
            state = next_state
            
            # Reset if episode ends
            if done:
                state = self.env.reset()
                done = False
        
        # Compute final value for incomplete episode
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = policy.evaluate(state_tensor, 
                                          torch.zeros((1, self.action_dim)), 
                                          upper_params)
        last_value = last_value.item()
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        log_probs = np.array(log_probs)
        
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, lambda_gae)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns
        }
    
    def sample_full_trajectories(self, policy, upper_params, num_trajectories=3, max_steps=1000):
        """Sample complete trajectories for gradient estimation."""
        all_trajectories = []
        
        for _ in range(num_trajectories):
            # Initialize trajectory data
            states = []
            actions = []
            rewards = []
            log_probs = []
            upper_param_grads = []
            
            # Reset environment
            state = self.env.reset()
            done = False
            step_count = 0
            
            # Collect a complete trajectory
            while not done and step_count < max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Create a fresh copy of upper_params with gradient tracking
                current_upper_params = upper_params.clone().detach().requires_grad_(True)
                
                # Get action with gradient tracking
                with torch.enable_grad():
                    action, log_prob = policy.get_action(state_tensor, current_upper_params)
                
                # Execute action in environment
                next_state, reward, done, _ = self.env.step(action.squeeze(0).detach().numpy())
                
                # Store trajectory data
                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                upper_param_grads.append(current_upper_params)
                
                # Update state for next step
                state = next_state
                step_count += 1
            
            trajectory = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'log_probs': log_probs,
                'upper_param_grads': upper_param_grads,
                'length': step_count
            }
            
            all_trajectories.append(trajectory)
        
        return all_trajectories
    
    def evaluate_policy(self, policy, upper_params, episodes=10, render=False):
        """Evaluate the current policy."""
        total_rewards = []
        
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # Get action (deterministic)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _ = policy.get_action(state_tensor, upper_params, deterministic=True)
                
                action_np = action.squeeze(0).numpy()
                next_state, reward, done, _ = self.env.step(action_np)
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        return sum(total_rewards) / len(total_rewards)
