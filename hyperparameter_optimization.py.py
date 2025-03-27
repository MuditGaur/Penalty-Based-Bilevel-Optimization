import optuna
import numpy as np
import torch
from config import make_config
from bilevel_optimizer import BilevelOptimizer
import os
import json
from datetime import datetime

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

class EarlyStoppingCallback:
    """Early stopping callback for Optuna."""
    
    def __init__(self, patience=5):
        self.patience = patience
        self.best_value = float('-inf')
        self.no_improvement_count = 0
    
    def __call__(self, study, trial):
        current_value = trial.value
        
        if current_value > self.best_value:
            self.best_value = current_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        if self.no_improvement_count >= self.patience:
            study.stop()


def objective(trial):
    """Objective function for hyperparameter optimization."""
    # Basic environment settings
    env_name = "Pendulum-v1"
    
    # Define hyperparameter search spaces
    upper_param_dim = trial.suggest_categorical("upper_param_dim", [4, 8, 16, 32])
    upper_lr = trial.suggest_float("upper_lr", 1e-5, 1e-3, log=True)
    lower_lr = trial.suggest_float("lower_lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    lambda_gae = trial.suggest_float("lambda_gae", 0.9, 0.99)
    clip_ratio = trial.suggest_float("clip_ratio", 0.1, 0.3)
    value_coef = trial.suggest_float("value_coef", 0.3, 1.0)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True)
    gradient_coupling_coef = trial.suggest_float("gradient_coupling_coef", 0.01, 0.5, log=True)
    gradient_diff_coef = trial.suggest_float("gradient_diff_coef", 0.01, 0.3, log=True)
    
    # Create a custom configuration
    config = make_config(
        env_name=env_name,
        upper_param_dim=upper_param_dim,
        upper_lr=upper_lr,
        lower_lr=lower_lr,
        gamma=gamma,
        lambda_gae=lambda_gae,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        gradient_coupling_coef=gradient_coupling_coef,
        gradient_diff_coef=gradient_diff_coef,
        # Use shorter training for faster evaluation
        iterations=20,
        eval_freq=10,
        eval_episodes=3
    )
    
    # Set a unique seed for each trial
    seed = trial.number
    set_random_seeds(seed)
    
    # Initialize and train
    bilevel_opt = BilevelOptimizer(config)
    
    try:
        # Train with early stopping
        best_reward = float('-inf')
        patience_counter = 0
        max_patience = 3
        
        for i in range(config.iterations // config.eval_freq):
            # Train for eval_freq iterations
            bilevel_opt.train(iterations=config.eval_freq, eval_freq=config.eval_freq)
            
            # Evaluate
            reward = bilevel_opt.evaluate(episodes=config.eval_episodes)
            
            # Report intermediate values
            trial.report(reward, i)
            
            # Check if we should stop
            if reward > best_reward:
                best_reward = reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Early stopping after {(i+1) * config.eval_freq} iterations")
                break
                
            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        # Final evaluation with more episodes
        final_reward = bilevel_opt.evaluate(episodes=10)
        
        return final_reward
            
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf')


def save_best_params(study, output_dir="hyperopt_results"):
    """Save the best parameters from the study."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Add timestamp
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "best_reward": best_value,
        "params": best_params
    }
    
    # Save to file
    filename = os.path.join(output_dir, f"best_params_{result['timestamp']}.json")
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Best parameters saved to {filename}")
    return filename


def run_hyperparameter_optimization(n_trials=100, study_name="bilevel_optimization", 
                                  storage=None, n_jobs=1, pruner=None):
    """Run hyperparameter optimization."""
    # Create a study
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
    # Create a study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # We want to maximize the reward
        pruner=pruner,
        load_if_exists=True
    )
    
    # Add early stopping callback
    early_stopping = EarlyStoppingCallback(patience=10)
    
    # Start optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[early_stopping])
    
    # Print results
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    save_path = save_best_params(study)
    
    # Return the best parameters
    return trial.params


def run_optimization_with_population_based_training(population_size=10, generations=5):
    """Implement population-based training for hyperparameter search."""
    # This is a simplified version of PBT
    # For a full implementation, you might want to use ray.tune or a custom implementation
    
    # Initialize population with random hyperparameters
    population = []
    for i in range(population_size):
        params = {
            "upper_param_dim": np.random.choice([4, 8, 16, 32]),
            "upper_lr": 10 ** np.random.uniform(-5, -3),
            "lower_lr": 10 ** np.random.uniform(-5, -3),
            "gamma": np.random.uniform(0.97, 0.999),
            "lambda_gae": np.random.uniform(0.9, 0.99),
            "clip_ratio": np.random.uniform(0.1, 0.3),
            "value_coef": np.random.uniform(0.3, 1.0),
            "entropy_coef": 10 ** np.random.uniform(-3, -1.3),
            "gradient_coupling_coef": 10 ** np.random.uniform(-2, -0.3),
            "gradient_diff_coef": 10 ** np.random.uniform(-2, -0.5),
        }
        population.append({
            "params": params,
            "reward": None,
            "agent": None
        })
    
    # PBT iterations
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations}")
        
        # Evaluate all agents
        for i, individual in enumerate(population):
            # Create config
            config = make_config(
                env_name="Pendulum-v1",
                **individual["params"],
                iterations=15,  # Shorter training for faster evaluation
                eval_freq=5,
                eval_episodes=3
            )
            
            # Set seed for reproducibility within the generation
            set_random_seeds(i + generation * population_size)
            
            # Initialize agent
            agent = BilevelOptimizer(config)
            individual["agent"] = agent
            
            # Train
            agent.train()
            
            # Evaluate
            reward = agent.evaluate(episodes=5)
            individual["reward"] = reward
            
            print(f"  Individual {i+1}/{population_size} - Reward: {reward:.2f}")
        
        # Sort population by reward
        population.sort(key=lambda x: x["reward"], reverse=True)
        
        # Early stopping for the last generation
        if generation == generations - 1:
            break
        
        # Replace bottom half with modified versions of top half
        for i in range(population_size // 2, population_size):
            # Choose a model from the top half to replicate
            source_idx = np.random.randint(0, population_size // 2)
            source = population[source_idx]
            
            # Copy parameters with mutations
            new_params = source["params"].copy()
            
            # Mutate some parameters (randomly selected)
            for param in new_params:
                # 30% chance to mutate each parameter
                if np.random.random() < 0.3:
                    if param == "upper_param_dim":
                        new_params[param] = np.random.choice([4, 8, 16, 32])
                    elif "lr" in param:
                        # Perturb learning rates
                        new_params[param] *= np.random.uniform(0.5, 2.0)
                    elif param in ["gamma", "lambda_gae"]:
                        # Small perturbations for discount factors
                        new_params[param] += np.random.uniform(-0.01, 0.01)
                        new_params[param] = min(max(new_params[param], 0.9), 0.999)
                    else:
                        # Other parameters, perturb by up to 50%
                        new_params[param] *= np.random.uniform(0.5, 1.5)
            
            # Update the individual
            population[i]["params"] = new_params
            population[i]["reward"] = None
            population[i]["agent"] = None
    
    # Get the best parameters
    best_individual = max(population, key=lambda x: x["reward"])
    best_params = best_individual["params"]
    best_reward = best_individual["reward"]
    
    print("\nBest parameters from PBT:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best reward: {best_reward:.2f}")
    
    # Save the best parameters
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "best_reward": best_reward,
        "params": best_params,
        "method": "population_based_training"
    }
    
    os.makedirs("hyperopt_results", exist_ok=True)
    filename = os.path.join("hyperopt_results", f"pbt_best_params_{result['timestamp']}.json")
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Best parameters saved to {filename}")
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--method", type=str, default="optuna", choices=["optuna", "pbt"],
                       help="Optimization method: optuna or population-based training (pbt)")
    parser.add_argument("--trials", type=int, default=50,
                       help="Number of trials for Optuna")
    parser.add_argument("--population", type=int, default=10,
                       help="Population size for PBT")
    parser.add_argument("--generations", type=int, default=5,
                       help="Number of generations for PBT")
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs for Optuna")
    
    args = parser.parse_args()
    
    if args.method == "optuna":
        print(f"Running Optuna with {args.trials} trials and {args.jobs} parallel jobs")
        run_hyperparameter_optimization(n_trials=args.trials, n_jobs=args.jobs)
    else:
        print(f"Running Population-Based Training with population={args.population}, generations={args.generations}")
        run_optimization_with_population_based_training(
            population_size=args.population,
            generations=args.generations
        )
