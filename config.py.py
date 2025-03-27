class Config:
    """Configuration class for bilevel optimization."""
    
    # Environment settings
    env_name = "Pendulum-v1"
    
    # Network architecture
    hidden_dim_policy = 128
    hidden_dim_upper = 64
    
    # Upper level settings
    upper_param_dim = 8
    upper_lr = 1e-4
    gradient_diff_coef = 0.05
    
    # Lower level settings
    lower_lr = 3e-4
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    gradient_coupling_coef = 0.1
    
    # PPO settings
    ppo_epochs = 10
    batch_size = 64
    
    # General RL settings
    gamma = 0.99
    lambda_gae = 0.95
    
    # Training settings
    iterations = 50
    eval_freq = 10
    eval_episodes = 5
    trajectory_length = 2048
    max_steps_per_episode = 1000
    num_trajectories = 3  # For gradient estimation
    
    # Sync policy networks every N iterations
    sync_frequency = 10


def get_config():
    """Get default configuration."""
    return Config()


def make_config(**kwargs):
    """Create a custom configuration by overriding defaults."""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Config has no attribute '{key}'")
    
    return config
