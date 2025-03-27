import torch
import numpy as np
import argparse
from config import make_config
from bilevel_optimizer import BilevelOptimizer

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bilevel Optimization with Policy Gradients")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="Pendulum-v1", 
                        help="OpenAI Gym environment name")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of training iterations")
    parser.add_argument("--eval_freq", type=int, default=10,
                       help="Evaluation frequency (iterations)")
    parser.add_argument("--eval_episodes", type=int, default=5,
                       help="Number of episodes for evaluation")
    
    # Algorithm parameters
    parser.add_argument("--upper_lr", type=float, default=1e-4,
                       help="Upper level learning rate")
    parser.add_argument("--lower_lr", type=float, default=3e-4,
                       help="Lower level learning rate")
    parser.add_argument("--upper_param_dim", type=int, default=8,
                       help="Upper level parameter dimension")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gradient_coupling", type=float, default=0.1,
                       help="Gradient coupling coefficient")
    parser.add_argument("--gradient_diff", type=float, default=0.05,
                       help="Gradient difference coefficient")
    
    # Other settings
    parser.add_argument("--render", action="store_true", 
                       help="Render final evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create configuration
    config = make_config(
        env_name=args.env,
        upper_lr=args.upper_lr,
        lower_lr=args.lower_lr,
        upper_param_dim=args.upper_param_dim,
        gamma=args.gamma,
        gradient_coupling_coef=args.gradient_coupling,
        gradient_diff_coef=args.gradient_diff,
        iterations=args.iterations,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes
    )
    
    # Initialize bilevel optimizer
    bilevel_opt = BilevelOptimizer(config)
    
    # Train the bilevel system
    print(f"Starting training for {args.iterations} iterations...")
    bilevel_opt.train()
    
    # Evaluate final performance
    print("\nEvaluating final policy...")
    final_reward = bilevel_opt.evaluate(episodes=10, render=args.render)
    print(f"Final Average Reward: {final_reward:.2f}")

if __name__ == "__main__":
    main()
