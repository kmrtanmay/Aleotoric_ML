"""
Advanced Uncertainty and Adversarial Methods for Reinforcement Learning

This module implements:
1. Sophisticated adversarial attack methods for RL
2. Advanced uncertainty quantification and calibration techniques
3. Metrics for evaluating robustness under uncertainty
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import entropy, pearsonr

# Import base framework (assuming it's in the same directory)
# from uncertainty_rl_framework import *


# ==========================
# ADVANCED ADVERSARIAL ATTACKS
# ==========================

class PGDAttack:
    """
    Projected Gradient Descent attack for RL agents.
    More sophisticated than FGSM by taking multiple gradient steps.
    """
    
    def __init__(self, epsilon=0.1, alpha=0.01, num_steps=10, random_start=True):
        """
        Initialize the PGD attack.
        
        Args:
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of iterations
            random_start: Whether to start with a random perturbation
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        
    def attack(self, model, state, criterion, observation_space):
        """
        Perform PGD attack on the state.
        
        Args:
            model: The agent's model
            state: Current state
            criterion: Loss function
            observation_space: Environment's observation space
            
        Returns:
            Perturbed state
        """
        # Initialize perturbed state
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.clone().detach()
            
        # Get bounds from observation space
        if hasattr(observation_space, 'low') and hasattr(observation_space, 'high'):
            low = torch.FloatTensor(observation_space.low)
            high = torch.FloatTensor(observation_space.high)
        else:
            # Default bounds if not specified
            low = -torch.ones_like(state_tensor) * float('inf')
            high = torch.ones_like(state_tensor) * float('inf')
            
        # Random start
        if self.random_start:
            random_noise = torch.FloatTensor(*state_tensor.shape).uniform_(-self.epsilon, self.epsilon)
            perturbed_state = state_tensor + random_noise
            perturbed_state = torch.max(torch.min(perturbed_state, high), low)
        else:
            perturbed_state = state_tensor.clone()
            
        for _ in range(self.num_steps):
            # Ensure we can compute gradients
            perturbed_state.requires_grad = True
            
            # Forward pass
            output = model(perturbed_state)
            
            # Compute loss and gradients
            # For Q-networks, the loss aims to minimize the max Q-value
            if isinstance(output, torch.Tensor) and output.size(-1) > 1:
                # For Q-values in DQN, minimize the max Q-value
                target = -torch.argmax(output, dim=1)
                loss = criterion(output, target)
            else:
                # For value functions, minimize the value directly
                loss = -output.mean()
                
            loss.backward()
            
            # Update with gradient step
            with torch.no_grad():
                gradients = perturbed_state.grad.sign()
                perturbed_state = torch.max(torch.min(perturbed_state, high), low)
                
                # Reset gradients
                perturbed_state.grad.zero_()
            
        return perturbed_state.squeeze(0).detach().numpy()


class CriticBasedAttack:
    """
    Critic-based attack for actor-critic methods.
    Uses the critic's value function to guide adversarial perturbations.
    """
    
    def __init__(self, epsilon=0.1, alpha=0.01, num_steps=10):
        """
        Initialize the critic-based attack.
        
        Args:
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of iterations
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
    def attack(self, actor_critic, state, observation_space):
        """
        Perform critic-based attack on the state.
        
        Args:
            actor_critic: The agent's actor-critic model
            state: Current state
            observation_space: Environment's observation space
            
        Returns:
            Perturbed state
        """
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.clone().detach()
            
        # Get bounds from observation space
        if hasattr(observation_space, 'low') and hasattr(observation_space, 'high'):
            low = torch.FloatTensor(observation_space.low)
            high = torch.FloatTensor(observation_space.high)
        else:
            # Default bounds if not specified
            low = -torch.ones_like(state_tensor) * float('inf')
            high = torch.ones_like(state_tensor) * float('inf')
            
        perturbed_state = state_tensor.clone()
        
        for _ in range(self.num_steps):
            # Ensure we can compute gradients
            perturbed_state.requires_grad = True
            
            # For actor-critic, we want to minimize the value function
            # This depends on the specific actor-critic implementation
            if hasattr(actor_critic, 'critic'):
                # Specific case for PPO-style actor-critic
                _, value = actor_critic(perturbed_state)
                loss = -value.mean()  # Minimize value function
            elif hasattr(actor_critic, 'forward') and callable(actor_critic.forward):
                # Generic case, assuming last output is value
                outputs = actor_critic(perturbed_state)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    value = outputs[-1]  # Assume last output is value
                    loss = -value.mean()
                else:
                    # Fallback to direct minimization
                    loss = -outputs.mean()
            else:
                # Fallback for other architectures
                # Assuming actor_critic is callable and returns value
                loss = -actor_critic(perturbed_state).mean()
                
            loss.backward()
            
            # Update with gradient step
            with torch.no_grad():
                gradients = perturbed_state.grad.sign()
                perturbed_state = perturbed_state + self.alpha * gradients
                
                # Project back to epsilon ball
                delta = torch.clamp(perturbed_state - state_tensor, -self.epsilon, self.epsilon)
                perturbed_state = state_tensor + delta
                
                # Project back to valid state space
                perturbed_state = torch.max(torch.min(perturbed_state, high), low)
                
                # Reset gradients
                if perturbed_state.grad is not None:
                    perturbed_state.grad.zero_()
            
        return perturbed_state.squeeze(0).detach().numpy()


class AdversarialStateWrapper(gym.Wrapper):
    """
    Advanced wrapper for state-based adversarial attacks.
    Supports PGD and other iterative attacks.
    """
    
    def __init__(self, env, attack_type='pgd', model=None, epsilon=0.1, 
                alpha=0.01, num_steps=5, attack_freq=1.0):
        """
        Initialize the adversarial state wrapper.
        
        Args:
            env: Environment to wrap
            attack_type: Type of attack ('pgd', 'critic_based')
            model: Agent's model for gradient-based attacks
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of iterations
            attack_freq: Frequency of attacks (probability per step)
        """
        super().__init__(env)
        self.attack_type = attack_type
        self.model = model
        self.epsilon = epsilon
        self.attack_freq = attack_freq
        
        # Initialize attack method
        if attack_type == 'pgd':
            self.attack = PGDAttack(epsilon, alpha, num_steps)
        elif attack_type == 'critic_based':
            self.attack = CriticBasedAttack(epsilon, alpha, num_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # For loss computation in PGD
        self.criterion = nn.CrossEntropyLoss()
        
    def step(self, action):
        """Step the environment and potentially apply adversarial attack."""
        obs, reward, done, info = self.env.step(action)
        
        # Apply attack with probability attack_freq
        if self.model is not None and np.random.random() < self.attack_freq:
            if self.attack_type == 'pgd':
                perturbed_obs = self.attack.attack(
                    self.model, obs, self.criterion, self.observation_space
                )
            elif self.attack_type == 'critic_based':
                perturbed_obs = self.attack.attack(
                    self.model, obs, self.observation_space
                )
            else:
                perturbed_obs = obs
                
            return perturbed_obs, reward, done, info
        
        return obs, reward, done, info


# ==========================
# ADVANCED UNCERTAINTY QUANTIFICATION
# ==========================

class BayesianEnsemble:
    """
    Bayesian ensemble method for uncertainty quantification.
    Combines multiple models with different initializations.
    """
    
    def __init__(self, model_class, state_dim, action_dim, hidden_dim, num_models=5, **kwargs):
        """
        Initialize the Bayesian ensemble.
        
        Args:
            model_class: Class for the base model
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension
            num_models: Number of models in the ensemble
            **kwargs: Additional arguments for the model
        """
        self.num_models = num_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ensemble
        self.models = [
            model_class(state_dim, action_dim, hidden_dim, **kwargs).to(self.device)
            for _ in range(num_models)
        ]
        
        # Initialize optimizers
        self.optimizers = [
            optim.Adam(model.parameters(), lr=kwargs.get('lr', 3e-4))
            for model in self.models
        ]
        
    def predict(self, state, return_std=False):
        """
        Make prediction with uncertainty quantification.
        
        Args:
            state: Input state
            return_std: Whether to return standard deviation
            
        Returns:
            Mean prediction and (optionally) standard deviation
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                prediction = model(state)
                predictions.append(prediction)
                
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute mean and std
        mean_pred = predictions.mean(dim=0)
        
        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred
        
    def train_step(self, states, targets, weights=None):
        """
        Perform a training step for each model in the ensemble.
        
        Args:
            states: Input states
            targets: Target values
            weights: Optional weights for each sample (for bootstrapping)
            
        Returns:
            Mean loss across the ensemble
        """
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.FloatTensor(targets).to(self.device)
            
        # Get batch size
        batch_size = states.shape[0]
        
        ensemble_losses = []
        
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            
            # Generate bootstrap weights if not provided
            if weights is None:
                # Sample with replacement: some samples may be repeated, some not used
                bootstrap_idx = np.random.choice(batch_size, batch_size, replace=True)
                bootstrap_states = states[bootstrap_idx]
                bootstrap_targets = targets[bootstrap_idx]
            else:
                bootstrap_states = states
                bootstrap_targets = targets
                
            # Forward pass
            predictions = model(bootstrap_states)
            
            # Compute loss
            loss = F.mse_loss(predictions, bootstrap_targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ensemble_losses.append(loss.item())
            
        return np.mean(ensemble_losses)


class DropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty quantification.
    Keeps dropout active during inference to obtain epistemic uncertainty.
    """
    
    def __init__(self, model, num_samples=10):
        """
        Initialize the Dropout uncertainty wrapper.
        
        Args:
            model: Model with dropout layers
            num_samples: Number of forward passes for uncertainty estimation
        """
        self.model = model
        self.num_samples = num_samples
        
    def predict(self, state, return_std=False):
        """
        Make prediction with uncertainty quantification.
        
        Args:
            state: Input state
            return_std: Whether to return standard deviation
            
        Returns:
            Mean prediction and (optionally) standard deviation
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
            
        # Set model to train mode to enable dropout
        self.model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                prediction = self.model(state)
                predictions.append(prediction)
                
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute mean and std
        mean_pred = predictions.mean(dim=0)
        
        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred


class UncertaintyCalibration:
    """
    Tools for evaluating and calibrating uncertainty estimates.
    """
    
    @staticmethod
    def expected_calibration_error(uncertainties, errors, num_bins=10):
        """
        Compute the Expected Calibration Error (ECE).
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors (e.g. |y_true - y_pred|)
            num_bins: Number of bins for calibration
            
        Returns:
            ECE value and bin information for plotting
        """
        # Ensure inputs are numpy arrays
        uncertainties = np.array(uncertainties)
        errors = np.array(errors)
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = errors[sorted_indices]
        
        # Create bins
        bin_size = len(sorted_uncertainties) // num_bins
        bins = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_uncertainties)
            
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            bin_errors = sorted_errors[start_idx:end_idx]
            
            avg_uncertainty = np.mean(bin_uncertainties)
            avg_error = np.mean(bin_errors)
            bin_size = len(bin_uncertainties)
            
            bins.append({
                'avg_uncertainty': avg_uncertainty,
                'avg_error': avg_error,
                'size': bin_size
            })
        
        # Compute ECE
        total_samples = len(uncertainties)
        ece = 0
        
        for bin_info in bins:
            bin_weight = bin_info['size'] / total_samples
            ece += bin_weight * abs(bin_info['avg_uncertainty'] - bin_info['avg_error'])
            
        return ece, bins
    
    @staticmethod
    def plot_calibration(uncertainties, errors, num_bins=10, title="Uncertainty Calibration"):
        """
        Plot uncertainty calibration.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            num_bins: Number of bins for calibration
            title: Plot title
        """
        ece, bins = UncertaintyCalibration.expected_calibration_error(
            uncertainties, errors, num_bins
        )
        
        # Extract data for plotting
        avg_uncertainties = [bin_info['avg_uncertainty'] for bin_info in bins]
        avg_errors = [bin_info['avg_error'] for bin_info in bins]
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Perfect calibration line
        max_val = max(max(avg_uncertainties), max(avg_errors))
        plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Calibration')
        
        # Actual calibration
        plt.scatter(avg_uncertainties, avg_errors, s=100, alpha=0.7)
        plt.plot(avg_uncertainties, avg_errors, 'o-', label=f'Calibration (ECE={ece:.4f})')
        
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Observed Error')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure
    
    @staticmethod
    def recalibrate_uncertainty(model, calibration_data, alpha=0.05, method='isotonic'):
        """
        Recalibrate uncertainty estimates.
        
        Args:
            model: Model to calibrate
            calibration_data: Data for calibration (states, targets)
            alpha: Significance level for conformal prediction
            method: Calibration method ('isotonic' or 'temperature')
            
        Returns:
            Calibrated model
        """
        # Implementation depends on the specific model type
        # This is a placeholder for the general approach
        
        if method == 'isotonic':
            # Isotonic regression for calibration
            from sklearn.isotonic import IsotonicRegression
            
            # Get states and targets
            states, targets = calibration_data
            
            # Get uncertainty estimates
            if hasattr(model, 'predict') and callable(model.predict):
                _, uncertainties = model.predict(states, return_std=True)
                uncertainties = uncertainties.detach().cpu().numpy()
            else:
                # Fallback for other models
                uncertainties = np.ones(len(states))
                
            # Get predictions
            if hasattr(model, 'predict') and callable(model.predict):
                predictions = model.predict(states)
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.detach().cpu().numpy()
            else:
                # Fallback for other models
                predictions = np.zeros(len(states))
                
            # Compute errors
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
            errors = np.abs(predictions - targets)
            
            # Train isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(uncertainties.flatten(), errors.flatten())
            
            # Return calibrated model (as a wrapper)
            return lambda x: (model(x), torch.tensor(iso_reg.predict(model(x).detach().cpu().numpy())))
            
        elif method == 'temperature':
            # Temperature scaling for calibration
            # Simplified version, would need proper implementation for specific models
            print("Temperature scaling calibration not implemented yet.")
            return model
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")


# ==========================
# ROBUSTNESS METRICS
# ==========================

class RobustnessMetrics:
    """
    Metrics for evaluating robustness under uncertainty.
    """
    
    @staticmethod
    def robustness_gap(clean_rewards, noisy_rewards):
        """
        Compute robustness gap.
        
        Args:
            clean_rewards: Rewards in clean environment
            noisy_rewards: Rewards in noisy environment
            
        Returns:
            Robustness gap value
        """
        return np.mean(clean_rewards) - np.mean(noisy_rewards)
    
    @staticmethod
    def relative_performance(clean_rewards, noisy_rewards):
        """
        Compute relative performance.
        
        Args:
            clean_rewards: Rewards in clean environment
            noisy_rewards: Rewards in noisy environment
            
        Returns:
            Relative performance value
        """
        return np.mean(noisy_rewards) / np.mean(clean_rewards) if np.mean(clean_rewards) != 0 else float('inf')
    
    @staticmethod
    def kl_divergence(clean_policy, noisy_policy):
        """
        Compute KL divergence between clean and noisy policies.
        
        Args:
            clean_policy: Action probabilities in clean environment
            noisy_policy: Action probabilities in noisy environment
            
        Returns:
            KL divergence value
        """
        return entropy(clean_policy, noisy_policy)
    
    @staticmethod
    def uncertainty_reward_correlation(uncertainties, rewards):
        """
        Compute correlation between uncertainty and reward.
        
        Args:
            uncertainties: Uncertainty estimates
            rewards: Corresponding rewards
            
        Returns:
            Correlation coefficient and p-value
        """
        return pearsonr(uncertainties, rewards)
    
    @staticmethod
    def critical_state_identification(uncertainties, rewards, threshold=0.9):
        """
        Identify critical states based on uncertainty and reward.
        
        Args:
            uncertainties: Uncertainty estimates
            rewards: Corresponding rewards
            threshold: Percentile threshold for high uncertainty
            
        Returns:
            Indices of critical states
        """
        # Normalize uncertainties
        norm_uncertainties = (uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties) + 1e-8)
        
        # Normalize rewards (inverted: lower is worse)
        norm_rewards = 1 - (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
        
        # Compute criticality score (high uncertainty + low reward)
        criticality = norm_uncertainties * norm_rewards
        
        # Find states above threshold
        threshold_value = np.percentile(criticality, threshold * 100)
        critical_indices = np.where(criticality >= threshold_value)[0]
        
        return critical_indices
    
    @staticmethod
    def compute_all_metrics(clean_data, noisy_data):
        """
        Compute all robustness metrics.
        
        Args:
            clean_data: Data from clean environment (rewards, uncertainties, policies)
            noisy_data: Data from noisy environment (rewards, uncertainties, policies)
            
        Returns:
            Dictionary of metrics
        """
        clean_rewards = clean_data.get('rewards', [])
        noisy_rewards = noisy_data.get('rewards', [])
        clean_uncertainties = clean_data.get('uncertainties', [])
        noisy_uncertainties = noisy_data.get('uncertainties', [])
        clean_policies = clean_data.get('policies', [])
        noisy_policies = noisy_data.get('policies', [])
        
        metrics = {}
        
        # Basic metrics
        metrics['robustness_gap'] = RobustnessMetrics.robustness_gap(clean_rewards, noisy_rewards)
        metrics['relative_performance'] = RobustnessMetrics.relative_performance(clean_rewards, noisy_rewards)
        
        # Uncertainty metrics
        if clean_uncertainties and noisy_uncertainties:
            metrics['uncertainty_increase'] = np.mean(noisy_uncertainties) / np.mean(clean_uncertainties)
            
            if len(clean_uncertainties) == len(clean_rewards):
                corr, p_value = RobustnessMetrics.uncertainty_reward_correlation(
                    clean_uncertainties, clean_rewards
                )
                metrics['clean_uncertainty_reward_corr'] = corr
                metrics['clean_uncertainty_reward_p_value'] = p_value
                
            if len(noisy_uncertainties) == len(noisy_rewards):
                corr, p_value = RobustnessMetrics.uncertainty_reward_correlation(
                    noisy_uncertainties, noisy_rewards
                )
                metrics['noisy_uncertainty_reward_corr'] = corr
                metrics['noisy_uncertainty_reward_p_value'] = p_value
        
        # Policy metrics
        if clean_policies and noisy_policies and len(clean_policies) == len(noisy_policies):
            kl_divs = [RobustnessMetrics.kl_divergence(cp, np) for cp, np in zip(clean_policies, noisy_policies)]
            metrics['avg_kl_divergence'] = np.mean(kl_divs)
            metrics['max_kl_divergence'] = np.max(kl_divs)
            
        return metrics


# ==========================
# EXAMPLE USAGE
# ==========================

def demonstrate_pgd_attack():
    """Demonstrate PGD attack on CartPole."""
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create a simple DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    
    # Define a simple Q-network
    q_network = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
    )
    
    # Create PGD attack
    pgd_attack = PGDAttack(epsilon=0.1, alpha=0.01, num_steps=10)
    
    # Demonstrate attack on a random state
    state = env.reset()
    
    # Get clean Q-values
    with torch.no_grad():
        q_values = q_network(torch.FloatTensor(state).unsqueeze(0))
        clean_action = torch.argmax(q_values, dim=1).item()
    
    # Apply attack
    criterion = nn.CrossEntropyLoss()
    perturbed_state = pgd_attack.attack(q_network, state, criterion, env.observation_space)
    
    # Get perturbed Q-values
    with torch.no_grad():
        q_values = q_network(torch.FloatTensor(perturbed_state).unsqueeze(0))
        perturbed_action = torch.argmax(q_values, dim=1).item()
    
    print("Clean state:", state)
    print("Perturbed state:", perturbed_state)
    print("Clean action:", clean_action)
    print("Perturbed action:", perturbed_action)
    
    return clean_action != perturbed_action  # Return whether attack changed action


def demonstrate_uncertainty_calibration():
    """Demonstrate uncertainty calibration."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True uncertainties
    true_uncertainties = np.random.uniform(0, 1, n_samples)
    
    # Generated errors (with some noise)
    errors = true_uncertainties + 0.2 * np.random.normal(0, 1, n_samples)
    errors = np.abs(errors)
    
    # Predicted uncertainties (miscalibrated)
    miscalibrated = true_uncertainties ** 2  # Underestimate uncertainty
    
    # Compute and plot calibration
    fig = UncertaintyCalibration.plot_calibration(
        miscalibrated, errors, num_bins=10, 
        title="Uncertainty Calibration (Before)"
    )
    plt.show()
    
    # Apply isotonic regression for calibration
    from sklearn.isotonic import IsotonicRegression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(miscalibrated, errors)
    
    # Calibrated uncertainties
    calibrated = iso_reg.predict(miscalibrated)
    
    # Compute and plot calibration after recalibration
    fig = UncertaintyCalibration.plot_calibration(
        calibrated, errors, num_bins=10,
        title="Uncertainty Calibration (After)"
    )
    plt.show()
    
    # Compute ECE before and after
    ece_before, _ = UncertaintyCalibration.expected_calibration_error(
        miscalibrated, errors, num_bins=10
    )
    ece_after, _ = UncertaintyCalibration.expected_calibration_error(
        calibrated, errors, num_bins=10
    )
    
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration: {ece_after:.4f}")
    
    return ece_before, ece_after


def main():
    """Run demonstrations."""
    print("==== Demonstrating PGD Attack ====")
    attack_success = demonstrate_pgd_attack()
    print(f"Attack {'succeeded' if attack_success else 'failed'} in changing action\n")
    
    print("==== Demonstrating Uncertainty Calibration ====")
    ece_before, ece_after = demonstrate_uncertainty_calibration()
    print(f"ECE improvement: {ece_before - ece_after:.4f}\n")


if __name__ == "__main__":
    main()
state = perturbed_state + self.alpha * gradients
                
                # Project back to epsilon ball
                delta = torch.clamp(perturbed_state - state_tensor, -self.epsilon, self.epsilon)
                perturbed_state = state_tensor + delta
                
                # Project back to valid state space
                perturbed_