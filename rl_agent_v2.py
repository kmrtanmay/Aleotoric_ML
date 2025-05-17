"""
Reinforcement Learning Under Uncertainty Benchmarking Framework
===============================================================

This framework allows benchmarking standard RL algorithms from Stable-Baselines3 
under different types of uncertainty:
- Epistemic (Model/Parameter) Uncertainty
- Aleatoric (Environmental) Uncertainty 
- Adversarial Attacks

Compatible with MuJoCo and classic control environments from Gymnasium.
"""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Type, Union, Optional, Callable, Any
import random
import time
import pandas as pd
from enum import Enum
from dataclasses import dataclass

# Stable-Baselines3 imports
from stable_baselines3 import PPO, DQN, SAC, TD3, A2C
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

os.environ['MUJOCO_GL'] = 'egl'

# Check if CUDA is available and set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create log directory
LOG_DIR = "./logs_v4"
os.makedirs(LOG_DIR, exist_ok=True)

# Create models directory
MODELS_DIR = "./models_v4"
os.makedirs(MODELS_DIR, exist_ok=True)

# Create videos directory 
VIDEOS_DIR = "./videos_v4"
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Create directory for episodic rewards
EPISODE_REWARDS_DIR = "./episode_rewards_v4"
os.makedirs(EPISODE_REWARDS_DIR, exist_ok=True)

#-------------------------------------------------------------------------
# CONFIGURATION CLASSES
#-------------------------------------------------------------------------

class UncertaintyType(str, Enum):
    """Types of uncertainty for experimentation."""
    NONE = "none"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    ADVERSARIAL = "adversarial"

@dataclass
class ExperimentConfig:
    """Configuration parameters for running experiments."""
    env_id: str
    algorithm: str
    uncertainty_type: Optional[str] = None
    uncertainty_level: float = 0.0
    seed: int = 42
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    log_dir: str = "./logs_v4"
    save_model: bool = True
    
    def __str__(self):
        return (f"{self.env_id}_{self.algorithm}_"
                f"{self.uncertainty_type or 'baseline'}_{self.uncertainty_level}_"
                f"seed{self.seed}")
    
    def get_log_path(self):
        """Get the logging path for this experiment."""
        return os.path.join(self.log_dir, str(self))
    
    def get_model_path(self):
        """Get the model saving path for this experiment."""
        return os.path.join(MODELS_DIR, str(self))
    
    def get_episode_rewards_path(self):
        """Get the path for saving episodic rewards."""
        return os.path.join(EPISODE_REWARDS_DIR, f"{str(self)}_episodic_rewards.csv")


#-------------------------------------------------------------------------
# 1. UNCERTAINTY WRAPPERS
#-------------------------------------------------------------------------

class EpistemicUncertaintyWrapper(gym.Wrapper):
    """
    Apply epistemic uncertainty to the environment by introducing
    model uncertainty through limited or modified state information.
    
    This wrapper simulates epistemic uncertainty by:
    1. Dropping state dimensions randomly (masking)
    2. Adding dropout-like noise to state information
    3. Restricting information in OOD scenarios
    """
    def __init__(self, env, uncertainty_level=0.2, dropout_prob=0.2, seed=None):
        super().__init__(env)
        self.uncertainty_level = uncertainty_level
        self.dropout_prob = dropout_prob
        self.rng = np.random.RandomState(seed)
        
        # For observation space masks
        if isinstance(self.observation_space, spaces.Box):
            self.obs_shape = self.observation_space.shape
            self.obs_dim = np.prod(self.obs_shape)
            self.mask = np.ones(self.obs_dim, dtype=bool)
        
    def _apply_epistemic_uncertainty(self, obs):
        """Apply epistemic uncertainty to an observation."""
        if isinstance(self.observation_space, spaces.Box):
            # Flatten observation
            flat_obs = obs.flatten()
            
            # 1. Randomly mask out some dimensions
            if self.rng.random() < self.uncertainty_level:
                mask = self.rng.choice(
                    [0, 1], 
                    size=self.obs_dim, 
                    p=[self.dropout_prob, 1 - self.dropout_prob]
                ).astype(bool)
                # Always keep at least one dimension
                if not np.any(mask):
                    mask[self.rng.randint(0, self.obs_dim)] = True
                
                flat_obs = flat_obs * mask
            
            # 2. Add dropout-like noise
            dropout_mask = self.rng.binomial(1, 1-self.dropout_prob, self.obs_dim)
            flat_obs = flat_obs * dropout_mask * (1.0 / (1.0 - self.dropout_prob))
            
            # Reshape back
            return flat_obs.reshape(self.obs_shape)
        
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._apply_epistemic_uncertainty(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        modified_obs = self._apply_epistemic_uncertainty(obs)
        return modified_obs, reward, terminated, truncated, info


class AleatoricUncertaintyWrapper(gym.Wrapper):
    """
    Apply aleatoric uncertainty to the environment by adding environmental stochasticity.
    
    This wrapper simulates aleatoric uncertainty by:
    1. Adding Gaussian noise to observations
    2. Adding randomness to action execution
    3. Adding stochasticity to transition dynamics
    """
    def __init__(self, env, uncertainty_level=0.2, noise_type='gaussian', seed=None):
        super().__init__(env)
        self.uncertainty_level = uncertainty_level
        self.noise_type = noise_type
        self.rng = np.random.RandomState(seed)
        
        # Different noise distributions
        self.noise_functions = {
            'gaussian': lambda shape: self.rng.normal(0, self.uncertainty_level, shape),
            'uniform': lambda shape: self.rng.uniform(-self.uncertainty_level, self.uncertainty_level, shape),
            'laplace': lambda shape: self.rng.laplace(0, self.uncertainty_level/np.sqrt(2), shape),
        }
        
        # Get a reference to the selected noise function
        if noise_type in self.noise_functions:
            self.noise_func = self.noise_functions[noise_type]
        else:
            self.noise_func = self.noise_functions['gaussian']
            
    def _apply_aleatoric_uncertainty(self, obs):
        """Apply aleatoric uncertainty to an observation."""
        if isinstance(self.observation_space, spaces.Box):
            # Add noise depending on observation range
            noise = self.noise_func(obs.shape)
            
            # Scale noise based on observation space
            low, high = self.observation_space.low, self.observation_space.high
            scale = (high - low) / 2.0 if not np.any(np.isinf(high)) else 1.0
            
            # Add noise and clip to valid range
            noisy_obs = obs + noise * scale
            noisy_obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
            return noisy_obs
            
        return obs
    
    def _apply_action_noise(self, action):
        """Apply noise to actions to simulate actuation errors."""
        if isinstance(self.action_space, spaces.Box):
            # Add noise to continuous actions
            noise = self.noise_func(action.shape)
            
            # Scale noise based on action space
            low, high = self.action_space.low, self.action_space.high
            scale = (high - low) / 2.0
            
            # Add noise and clip to valid range
            noisy_action = action + noise * scale
            noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
            return noisy_action
        
        elif isinstance(self.action_space, spaces.Discrete):
            # For discrete actions, randomly change action with small probability
            if self.rng.random() < self.uncertainty_level:
                return self.rng.randint(0, self.action_space.n)
        
        return action
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._apply_aleatoric_uncertainty(obs), info
    
    def step(self, action):
        # Apply noise to action (simulating actuator noise)
        noisy_action = self._apply_action_noise(action)
        
        # Take step with modified action
        obs, reward, terminated, truncated, info = self.env.step(noisy_action)
        
        # Apply noise to observation
        noisy_obs = self._apply_aleatoric_uncertainty(obs)
        
        # Apply randomness to rewards (optional)
        if self.uncertainty_level > 0:
            reward_noise = self.noise_func((1,))[0] * abs(reward) * 0.5
            reward = reward + reward_noise
            
        return noisy_obs, reward, terminated, truncated, info


class AdversarialAttackWrapper(gym.Wrapper):
    """
    Apply adversarial attacks to the environment's observations.
    
    This wrapper implements different attack methods:
    1. FGSM (Fast Gradient Sign Method)
    2. PGD (Projected Gradient Descent)
    3. Random noise attacks
    """
    def __init__(self, env, agent_model=None, attack_type='random', epsilon=0.2, attack_freq=0.5, seed=None):
        super().__init__(env)
        self.agent_model = agent_model  # The agent's model to use for gradient-based attacks
        self.attack_type = attack_type
        self.epsilon = epsilon  # Attack strength
        self.attack_freq = attack_freq  # Frequency of attacks (probability)
        self.rng = np.random.RandomState(seed)
        
    def _fgsm_attack(self, obs, model):
        """
        Fast Gradient Sign Method attack.
        
        Args:
            obs: The current observation
            model: The agent's model
            
        Returns:
            Perturbed observation
        """
        if isinstance(model, torch.nn.Module) and hasattr(model, 'policy'):
            # Convert to tensor if not already
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.FloatTensor(obs).to(device)
                if len(obs.shape) == 1:  # Add batch dimension if needed
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = obs
                
            obs_tensor.requires_grad = True
            
            try:
                # Try to get action
                with torch.enable_grad():
                    actions, _ = model.predict(obs_tensor, deterministic=True)
                    
                    if isinstance(actions, np.ndarray):
                        actions = torch.FloatTensor(actions).to(device)
                        
                    # For models with critic, use policy loss
                    if hasattr(model, 'critic'):
                        # Use critic to get value
                        values = model.critic(obs_tensor)
                        loss = -values.mean()  # Maximize negative value
                    else:
                        # Fallback: use action norm as loss
                        loss = -torch.norm(actions)
                        
                    # Calculate gradients
                    loss.backward()
                    
                    # Create perturbation
                    if obs_tensor.grad is not None:
                        perturbation = self.epsilon * torch.sign(obs_tensor.grad)
                        perturbed_obs = obs_tensor.detach() + perturbation
                        
                        # Clip to observation space bounds
                        if isinstance(self.observation_space, spaces.Box):
                            perturbed_obs = torch.clamp(
                                perturbed_obs, 
                                torch.FloatTensor(self.observation_space.low).to(device),
                                torch.FloatTensor(self.observation_space.high).to(device)
                            )
                        
                        return perturbed_obs.cpu().detach().numpy().squeeze()
            except Exception as e:
                print(f"Error in FGSM attack: {e}")
        
        # Fallback to random attack if FGSM fails
        return self._random_attack(obs)
    
    def _random_attack(self, obs):
        """Apply random noise attack as a baseline."""
        if isinstance(self.observation_space, spaces.Box):
            noise = self.rng.uniform(-self.epsilon, self.epsilon, obs.shape)
            perturbed_obs = obs + noise
            perturbed_obs = np.clip(
                perturbed_obs,
                self.observation_space.low,
                self.observation_space.high
            )
            return perturbed_obs
        return obs
    
    def _apply_attack(self, obs):
        """Choose and apply attack based on configuration."""
        if self.rng.random() > self.attack_freq:
            return obs
            
        if self.attack_type == 'fgsm' and self.agent_model is not None:
            return self._fgsm_attack(obs, self.agent_model)
        else:
            return self._random_attack(obs)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info  # No attack on reset
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply adversarial attack
        perturbed_obs = self._apply_attack(obs)
        
        # Store original observation for analysis if needed
        info['original_obs'] = obs
        
        return perturbed_obs, reward, terminated, truncated, info


# Helper function to create environment with uncertainty
def make_env_with_uncertainty(
    env_id: str,
    uncertainty_type: Optional[UncertaintyType] = None,
    uncertainty_level: float = 0.0,
    agent_model=None,
    seed: int = None,
    render_mode: Optional[str] = None,
):
    """Create an environment with the specified uncertainty type and level."""
    
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        
        # Add uncertainty based on type
        if uncertainty_type == UncertaintyType.EPISTEMIC:
            env = EpistemicUncertaintyWrapper(
                env, uncertainty_level=uncertainty_level, seed=seed
            )
        elif uncertainty_type == UncertaintyType.ALEATORIC:
            env = AleatoricUncertaintyWrapper(
                env, uncertainty_level=uncertainty_level, seed=seed
            )
        elif uncertainty_type == UncertaintyType.ADVERSARIAL:
            env = AdversarialAttackWrapper(
                env, agent_model, epsilon=uncertainty_level, seed=seed
            )
                
        env = Monitor(env)  # For tracking statistics
        
        # Set seeds for reproducibility
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
        return env
        
    return _init


#-------------------------------------------------------------------------
# 2. EVALUATION METRICS AND CALLBACKS
#-------------------------------------------------------------------------

class EpisodeWeightedRewards:
    """
    Tracks episode rewards and computes weighted moving averages.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.episode_rewards = []
        self.weighted_averages = []
        
    def add_reward(self, reward):
        """Add a new episode reward and update weighted average."""
        self.episode_rewards.append(reward)
        
        # Calculate weighted average with sliding window
        if len(self.episode_rewards) >= self.window_size:
            # Get last window_size rewards
            window = self.episode_rewards[-self.window_size:]
            
            # Calculate weighted average (more weight to recent episodes)
            weights = np.linspace(0.5, 1.0, self.window_size)
            weights = weights / weights.sum()  # Normalize weights
            weighted_avg = sum(w * r for w, r in zip(weights, window))
            
            self.weighted_averages.append(weighted_avg)
        elif len(self.episode_rewards) > 0:
            # If we don't have enough episodes yet, use simple average
            self.weighted_averages.append(np.mean(self.episode_rewards))
    
    def plot(self, save_path):
        """Plot weighted average rewards by episode. [COMMENTED OUT]"""
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(len(self.weighted_averages)), self.weighted_averages)
        # plt.xlabel('Episodes')
        # plt.ylabel(f'Weighted Average Reward (Window Size: {self.window_size})')
        # plt.title('Training Performance By Episode')
        # plt.grid(alpha=0.3)
        # plt.savefig(save_path)
        # plt.close()
        
        # Save the episode rewards as CSV for further analysis
        data = {
            'episode': list(range(len(self.episode_rewards))),
            'reward': self.episode_rewards
        }
        df = pd.DataFrame(data)
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)

class EpisodeRewardCallback(BaseCallback):
    """
    Callback for tracking episode rewards.
    """
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Add reward from current step
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if episode terminated or truncated
        if self.locals["dones"][0]:
            # Store the episode reward
            self.episode_rewards.append(self.current_episode_reward)
            
            # Save episodic rewards to CSV periodically
            if len(self.episode_rewards) % 10 == 0:  # Save every 10 episodes
                self._save_rewards()
                
            # Reset for next episode
            self.current_episode_reward = 0
            
        return True
    
    def _save_rewards(self):
        """Save episode rewards to CSV file."""
        data = {
            'episode': list(range(len(self.episode_rewards))),
            'reward': self.episode_rewards
        }
        df = pd.DataFrame(data)
        df.to_csv(self.save_path, index=False)
        
    def on_training_end(self):
        """Save final rewards when training ends."""
        self._save_rewards()

class WeightedRewardCallback(BaseCallback):
    """
    Callback for tracking weighted average rewards over episodes.
    
    Args:
        window_size: Number of episodes to include in weighted average
    """
    def __init__(self, window_size=20, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.rewards = []
        self.weighted_averages = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Add reward from current step
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if episode terminated or truncated
        if self.locals["dones"][0]:
            # Store the episode reward
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Calculate weighted average with last window_size episodes
            if len(self.episode_rewards) > 0:
                # Use the last window_size episodes (or all if fewer)
                window = self.episode_rewards[-self.window_size:]
                weighted_avg = sum(window) / len(window)
                self.weighted_averages.append(weighted_avg)
                
                # Log the weighted average
                self.logger.record("rollout/weighted_avg_reward", weighted_avg)
                
            # Reset for next episode
            self.current_episode_reward = 0
            
        return True
    
    def plot_weighted_rewards(self, save_path):
        """Plot weighted average rewards and save to file. [COMMENTED OUT]"""
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(self.weighted_averages)), self.weighted_averages)
        # plt.xlabel('Episodes')
        # plt.ylabel('Weighted Average Reward (Last 20 Episodes)')
        # plt.title('Training Performance: Weighted Average Reward')
        # plt.grid(alpha=0.3)
        # plt.savefig(save_path)
        # plt.close()
        
        # Save the episode rewards as CSV
        data = {
            'episode': list(range(len(self.episode_rewards))),
            'reward': self.episode_rewards
        }
        df = pd.DataFrame(data)
        csv_path = save_path.replace(".png", ".csv")
        df.to_csv(csv_path, index=False)

class UncertaintyEvaluationCallback(BaseCallback):
    """
    Callback for evaluating agent performance under different uncertainties.
    
    Records metrics such as:
    - Reward with/without uncertainty
    - Performance degradation curves
    - Recovery time after perturbations
    """
    def __init__(
        self,
        eval_env,
        agent_model,
        uncertainty_types,
        uncertainty_levels,
        eval_freq=10000,
        n_eval_episodes=5,
        log_path="./logs_v4",
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.agent_model = agent_model
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.uncertainty_types = uncertainty_types
        self.uncertainty_levels = uncertainty_levels
        self.log_path = log_path
        
        # Results storage
        self.results = {
            'timesteps': [],
            'mean_reward': [],
            'std_reward': [],
            'uncertainty_type': [],
            'uncertainty_level': [],
            'robustness_gap': [],
            'recovery_time': []
        }
        
        # Create log directory
        os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        """
        Evaluate the agent periodically at different uncertainty levels.
        """
        if self.num_timesteps % self.eval_freq != 0:
            return True
            
        # Log current timestep
        self.results['timesteps'].append(self.num_timesteps)
        
        # Baseline evaluation (no uncertainty)
        clean_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            clean_rewards.append(episode_reward)
            
        mean_clean_reward = np.mean(clean_rewards)
        std_clean_reward = np.std(clean_rewards)
        
        # Log baseline performance
        self.results['mean_reward'].append(mean_clean_reward)
        self.results['std_reward'].append(std_clean_reward)
        self.results['uncertainty_type'].append(UncertaintyType.NONE.value)
        self.results['uncertainty_level'].append(0.0)
        self.results['robustness_gap'].append(0.0)
        self.results['recovery_time'].append(0.0)
        
        self.logger.record("eval/mean_reward", mean_clean_reward)
        self.logger.record("eval/std_reward", std_clean_reward)
        
        # Evaluate under different uncertainties
        for uncertainty_type in self.uncertainty_types:
            for level in self.uncertainty_levels:
                if level == 0.0:
                    continue  # Skip baseline case (already evaluated)
                    
                # Create environment with uncertainty
                env_fn = make_env_with_uncertainty(
                    env_id=self.eval_env.unwrapped.spec.id,
                    uncertainty_type=uncertainty_type,
                    uncertainty_level=level,
                    agent_model=self.agent_model if uncertainty_type == UncertaintyType.ADVERSARIAL else None
                )
                
                env = env_fn()
                
                # Evaluate agent
                uncertain_rewards = []
                recovery_times = []
                
                for _ in range(self.n_eval_episodes):
                    obs, _ = env.reset()
                    done = False
                    episode_reward = 0
                    recovery_time = 0
                    recovering = False
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        
                        # Measure recovery time after negative reward
                        if reward < 0:
                            recovering = True
                            recovery_time += 1
                        elif recovering:
                            recovering = False
                            recovery_times.append(recovery_time)
                            recovery_time = 0
                            
                        done = terminated or truncated
                        
                    uncertain_rewards.append(episode_reward)
                    
                # Calculate metrics
                mean_uncertain_reward = np.mean(uncertain_rewards)
                std_uncertain_reward = np.std(uncertain_rewards)
                robustness_gap = mean_clean_reward - mean_uncertain_reward
                mean_recovery_time = np.mean(recovery_times) if recovery_times else 0
                
                # Log results
                self.results['mean_reward'].append(mean_uncertain_reward)
                self.results['std_reward'].append(std_uncertain_reward)
                self.results['uncertainty_type'].append(uncertainty_type.value)
                self.results['uncertainty_level'].append(level)
                self.results['robustness_gap'].append(robustness_gap)
                self.results['recovery_time'].append(mean_recovery_time)
                
                # Log to tensorboard
                self.logger.record(f"eval/{uncertainty_type.value}/mean_reward_{level}", mean_uncertain_reward)
                self.logger.record(f"eval/{uncertainty_type.value}/robustness_gap_{level}", robustness_gap)
                if mean_recovery_time > 0:
                    self.logger.record(f"eval/{uncertainty_type.value}/recovery_time_{level}", mean_recovery_time)
                
                # Clean up environment
                env.close()

        lengths = [len(arr) for arr in self.results.values()]
        if len(set(lengths)) > 1:
            # If arrays have different lengths, find the minimum length
            min_length = min(lengths)
            # Truncate all arrays to the minimum length
            for key in self.results:
                self.results[key] = self.results[key][:min_length]

        # Save results as CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.log_path}/uncertainty_evaluation_{self.num_timesteps}.csv", index=False)
        
        # Plot robustness curves [COMMENTED OUT]
        # self._plot_robustness_curves()
        
        return True
    
    def _plot_final_curves(self):
        """Plot final robustness curves using all collected data. [COMMENTED OUT]"""
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        
        # Save comprehensive final results as CSV
        df.to_csv(f"{self.log_path}/final_evaluation_results.csv", index=False)   
        
        # All plotting code is commented out as requested
        # [Plotting code was here]
        
    def _plot_robustness_curves(self):
        """Plot robustness curves for each uncertainty type. [COMMENTED OUT]"""
        # if len(self.results['timesteps']) == 0:
        #     return
            
        # # Convert results to DataFrame for easier plotting
        # df = pd.DataFrame(self.results)
        
        # # Get the latest timestep results
        # latest_timestep = df['timesteps'].max()
        # latest_df = df[df['timesteps'] == latest_timestep]
        
        # # Plot robustness gap vs uncertainty level for each uncertainty type
        # plt.figure(figsize=(12, 8))
        
        # for uncertainty_type in [ut.value for ut in self.uncertainty_types]:
        #     type_df = latest_df[latest_df['uncertainty_type'] == uncertainty_type]
        #     if not type_df.empty:
        #         plt.plot(
        #             type_df['uncertainty_level'], 
        #             type_df['robustness_gap'],
        #             marker='o',
        #             label=f"{uncertainty_type}"
        #         )
                
        # plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        # plt.xlabel('Uncertainty Level')
        # plt.ylabel('Robustness Gap (Clean Reward - Uncertain Reward)')
        # plt.title(f'Robustness Curves at {latest_timestep} Steps')
        # plt.legend()
        # plt.grid(alpha=0.3)
        
        # # Save plot
        # plt.savefig(f"{self.log_path}/robustness_curves_{latest_timestep}.png")
        # plt.close()


#-------------------------------------------------------------------------
# 3. EXPERIMENT RUNNER
#-------------------------------------------------------------------------

class UncertaintyExperiment:
    """
    Experiment runner for training and evaluating RL agents under uncertainty.
    
    Handles:
    - Agent creation and configuration
    - Environment setup with different uncertainty types
    - Training and evaluation loops
    - Results visualization and analysis
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agent = None
        self.env = None
        self.eval_env = None
        
        # Create directories
        self.logs_path = config.get_log_path()
        os.makedirs(self.logs_path, exist_ok=True)
        
        # Set random seeds
        set_random_seed(config.seed)
        
    def setup(self):
        """Setup environments and agent for the experiment."""
        # Create environment with uncertainty
        env_id = self.config.env_id
        uncertainty_type = UncertaintyType(self.config.uncertainty_type) if self.config.uncertainty_type else None
        uncertainty_level = self.config.uncertainty_level
        
        # Create base environment
        env_fn = make_env_with_uncertainty(
            env_id=env_id,
            uncertainty_type=uncertainty_type,
            uncertainty_level=uncertainty_level,
            seed=self.config.seed
        )
        
        self.env = DummyVecEnv([env_fn])
        
        # Create evaluation environment (without uncertainty for baseline comparisons)
        eval_env_fn = make_env_with_uncertainty(
            env_id=env_id,
            uncertainty_type=None,
            uncertainty_level=0.0,
            seed=self.config.seed + 1000  # Different seed for eval
        )
        
        self.eval_env = eval_env_fn()
        
        # Configure agent based on algorithm
        algorithm = self.config.algorithm.upper()
        
        if algorithm == "PPO":
            self.agent = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.logs_path,
                device=device
            )
        elif algorithm == "DQN":
            self.agent = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.logs_path,
                device=device
            )
        elif algorithm == "SAC":
            self.agent = SAC(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.logs_path,
                device=device
            )
        elif algorithm == "TD3":
            self.agent = TD3(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.logs_path,
                device=device
            )
        elif algorithm == "A2C":
            self.agent = A2C(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.logs_path,
                device=device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Set up logger
        new_logger = configure(
            self.logs_path, 
            ["stdout", "csv", "tensorboard"]
        )
        self.agent.set_logger(new_logger)
        
        return self
        
    def train(self):
        """Train the agent and evaluate under different uncertainties."""
        # Uncertainty types for evaluation
        uncertainty_types = [
            UncertaintyType.NONE,
            UncertaintyType.EPISTEMIC, 
            UncertaintyType.ALEATORIC
        ]
        
        # Add adversarial type only if we have a trained model
        if self.agent is not None:
            uncertainty_types.append(UncertaintyType.ADVERSARIAL)
            
        # Uncertainty levels for evaluation
        uncertainty_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        # Create episode reward callback to save episodic rewards
        episode_reward_callback = EpisodeRewardCallback(
            save_path=self.config.get_episode_rewards_path()
        )
        
        # Create other callbacks
        uncertainty_callback = UncertaintyEvaluationCallback(
            eval_env=self.eval_env,
            agent_model=self.agent,
            uncertainty_types=uncertainty_types,
            uncertainty_levels=uncertainty_levels,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            log_path=self.logs_path
        )
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.logs_path,
            log_path=self.logs_path,
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False
        )

        # Train the agent with the episodic reward tracking callback
        self.agent.learn(
            total_timesteps=self.config.total_timesteps,
            callback=[uncertainty_callback, eval_callback, episode_reward_callback]
        )

        # Generate final evaluation plots [COMMENTED OUT]
        # uncertainty_callback._plot_final_curves()

        # Save the trained agent
        if self.config.save_model:
            model_path = self.config.get_model_path()
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.agent.save(model_path)
            print(f"Model saved to {model_path}")
        
        return self
        
    def evaluate(self, uncertainty_types=None, uncertainty_levels=None):
        """
        Evaluate a trained agent under different uncertainty conditions.
        
        Args:
            uncertainty_types: List of uncertainty types to evaluate
            uncertainty_levels: List of uncertainty levels to evaluate
        """
        if self.agent is None:
            raise ValueError("Agent not trained. Call train() first or load a trained agent.")
            
        # Default uncertainty types and levels
        if uncertainty_types is None:
            uncertainty_types = [
                UncertaintyType.NONE,
                UncertaintyType.EPISTEMIC,
                UncertaintyType.ALEATORIC,
                UncertaintyType.ADVERSARIAL
            ]
            
        if uncertainty_levels is None:
            uncertainty_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
            
        # Results storage
        results = {
            'uncertainty_type': [],
            'uncertainty_level': [],
            'mean_reward': [],
            'std_reward': [],
            'success_rate': [],
            'failure_rate': [],
            'avg_episode_length': []
        }
        
        # Evaluate for each uncertainty type and level
        for u_type in uncertainty_types:
            for level in uncertainty_levels:
                # Skip adversarial with 0 level
                if u_type == UncertaintyType.ADVERSARIAL and level == 0.0:
                    continue
                    
                # Create evaluation environment with the specific uncertainty
                if u_type == UncertaintyType.NONE:
                    eval_env = gym.make(self.config.env_id)
                else:
                    env_fn = make_env_with_uncertainty(
                        env_id=self.config.env_id,
                        uncertainty_type=u_type,
                        uncertainty_level=level,
                        agent_model=self.agent if u_type == UncertaintyType.ADVERSARIAL else None,
                        seed=self.config.seed + 2000
                    )
                    eval_env = env_fn()
                
                # Evaluate agent
                episode_rewards = []
                episode_lengths = []
                success_count = 0
                failure_count = 0
                
                for i in range(self.config.n_eval_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    episode_reward = 0
                    episode_length = 0
                    
                    while not done:
                        action, _ = self.agent.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        done = terminated or truncated
                        
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    # Count successes/failures based on environment-specific criteria
                    # Customize these thresholds based on the environment
                    if hasattr(eval_env, 'is_success') and 'is_success' in info and info['is_success']:
                        success_count += 1
                    elif episode_reward > self._get_success_threshold(self.config.env_id):
                        success_count += 1
                    elif episode_reward < self._get_failure_threshold(self.config.env_id):
                        failure_count += 1
                
                # Calculate metrics
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                success_rate = success_count / self.config.n_eval_episodes
                failure_rate = failure_count / self.config.n_eval_episodes
                avg_episode_length = np.mean(episode_lengths)
                
                # Store results
                results['uncertainty_type'].append(u_type.value)
                results['uncertainty_level'].append(level)
                results['mean_reward'].append(mean_reward)
                results['std_reward'].append(std_reward)
                results['success_rate'].append(success_rate)
                results['failure_rate'].append(failure_rate)
                results['avg_episode_length'].append(avg_episode_length)
                
                print(f"Uncertainty: {u_type.value}, Level: {level}, "
                      f"Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2f}")
                
                # Close environment
                eval_env.close()
                
        # Save evaluation results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.logs_path, "evaluation_results.csv"), index=False)
        
        # Create plots [COMMENTED OUT]
        # self._plot_evaluation_results(results_df)
        
        return results_df
    
    def _get_success_threshold(self, env_id):
        """Get success threshold for a specific environment."""
        # These thresholds can be customized based on the environment
        thresholds = {
            "CartPole-v1": 475,              # Near maximum for CartPole
            "LunarLander-v3": 200,           # Good landing
            "HalfCheetah-v5": 4000,          # Good performance
            "Hopper-v5": 2000,               # Good hopping
            "Walker2d-v4": 3000,             # Good walking
            "Ant-v5": 3000,                  # Good performance
            "Humanoid-v5": 5000,             # Good performance
            "HumanoidStandup-v4": 100000,    # Good standup
            "Pendulum-v1": -200,             # Good balancing (negative rewards)
            "BipedalWalker-v3": 300,         # Successful walking
            # Add more environments as needed
        }
        
        # Return threshold or default value if environment not in dictionary
        return thresholds.get(env_id, 100)
    
    def _get_failure_threshold(self, env_id):
        """Get failure threshold for a specific environment."""
        # These thresholds can be customized based on the environment
        thresholds = {
            "CartPole-v1": 50,               # Quick failure
            "LunarLander-v3": -100,          # Crash
            "HalfCheetah-v5": 0,             # No progress
            "Hopper-v5": 0,                  # Fall
            "Walker2d-v4": 0,                # Fall
            "Ant-v5": 0,                     # Poor performance
            "Humanoid-v5": 0,                # Fall
            "HumanoidStandup-v4": 0,         # No progress
            "Pendulum-v1": -1000,            # Poor balancing
            "BipedalWalker-v3": -100,        # Fall
            # Add more environments as needed
        }
        
        # Return threshold or default value if environment not in dictionary
        return thresholds.get(env_id, -100)
    
    def _plot_evaluation_results(self, results_df):
        """Create visualizations of evaluation results. [COMMENTED OUT]"""
        # Plotting code is commented out as requested
        # All plotting functions are now commented out
        
        # Save the episodic rewards during evaluation
        for u_type in results_df['uncertainty_type'].unique():
            for level in results_df['uncertainty_level'].unique():
                type_df = results_df[(results_df['uncertainty_type'] == u_type) & 
                                     (results_df['uncertainty_level'] == level)]
                if not type_df.empty:
                    filename = f"{self.logs_path}/eval_{u_type}_level_{level}_rewards.csv"
                    type_df.to_csv(filename, index=False)
    
    def record_video(self, uncertainty_type=None, uncertainty_level=0.0):
        """
        Record a video of the agent's performance under uncertainty.
        
        Args:
            uncertainty_type: Type of uncertainty to apply
            uncertainty_level: Level of uncertainty to apply
        """
        if self.agent is None:
            raise ValueError("Agent not trained. Call train() first or load a trained agent.")
            
        # Create video directory
        video_path = os.path.join(VIDEOS_DIR, str(self.config))
        os.makedirs(video_path, exist_ok=True)
        
        # Create environment with video recording
        env_fn = make_env_with_uncertainty(
            env_id=self.config.env_id,
            uncertainty_type=uncertainty_type,
            uncertainty_level=uncertainty_level,
            agent_model=self.agent if uncertainty_type == UncertaintyType.ADVERSARIAL else None,
            render_mode="rgb_array"
        )
        
        env = env_fn()
        env = RecordVideo(
            env,
            video_path,
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix=f"{uncertainty_type.value if uncertainty_type else 'baseline'}_{uncertainty_level}"
        )
        
        # Run agent in environment
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        env.close()
        
        print(f"Recorded video with total reward: {total_reward}")
        return total_reward


#-------------------------------------------------------------------------
# 4. COMPARATIVE EXPERIMENTS
#-------------------------------------------------------------------------

def run_comparative_experiments(configs, uncertainty_types, uncertainty_levels):
    """
    Run and compare multiple experiments with different configurations.
    
    Args:
        configs: List of ExperimentConfig objects
        uncertainty_types: List of uncertainty types to evaluate
        uncertainty_levels: List of uncertainty levels to evaluate
        
    Returns:
        Dictionary of results
    """
    results = {}
    experiments = {}
    
    # Run experiments for each configuration
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config}")
        print(f"{'='*50}\n")
        
        # Create and run experiment
        experiment = UncertaintyExperiment(config)
        experiment.setup()
        experiment.train()
        
        # Store experiment and results
        experiments[str(config)] = experiment
        results[str(config)] = experiment.evaluate(
            uncertainty_types=uncertainty_types,
            uncertainty_levels=uncertainty_levels
        )
        
        # Record videos for visualization
        for u_type in uncertainty_types:
            # Record baseline (no uncertainty)
            if u_type == UncertaintyType.NONE:
                experiment.record_video()
            else:
                # Record with medium and high uncertainty
                for level in [0.2, 0.5]:
                    if u_type != UncertaintyType.ADVERSARIAL or level > 0:
                        experiment.record_video(u_type, level)
    
    # Create comparative visualizations [COMMENTED OUT]
    # compare_results(results, configs, uncertainty_types, uncertainty_levels)
    
    # Save combined episodic rewards from all experiments
    combined_rewards = {}
    for config_str, experiment in experiments.items():
        try:
            rewards_path = [c for c in configs if str(c) == config_str][0].get_episode_rewards_path()
            if os.path.exists(rewards_path):
                rewards_df = pd.read_csv(rewards_path)
                combined_rewards[config_str] = rewards_df
        except Exception as e:
            print(f"Error loading rewards for {config_str}: {e}")
    
    # Save combined rewards
    if combined_rewards:
        combined_dir = os.path.join(EPISODE_REWARDS_DIR, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        for config_str, df in combined_rewards.items():
            df.to_csv(os.path.join(combined_dir, f"{config_str}_rewards.csv"), index=False)
    
    return experiments, results


def compare_results(results, configs, uncertainty_types, uncertainty_levels):
    """
    Create comparative visualizations across multiple experiments. [COMMENTED OUT]
    
    Args:
        results: Dictionary of evaluation results
        configs: List of ExperimentConfig objects
        uncertainty_types: List of uncertainty types
        uncertainty_levels: List of uncertainty levels
    """
    # Output directory
    output_dir = os.path.join(LOG_DIR, "comparative_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrames
    result_dfs = {k: v for k, v in results.items()}
    
    # All plotting code is commented out as requested
    # [Plotting code was here]
    
    # Save combined results as CSV
    combined_results = []
    for config_str, df in result_dfs.items():
        config = [c for c in configs if str(c) == config_str][0]
        df_copy = df.copy()
        df_copy['algorithm'] = config.algorithm
        df_copy['env_id'] = config.env_id
        combined_results.append(df_copy)
    
    if combined_results:
        combined_df = pd.concat(combined_results)
        combined_df.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)


#-------------------------------------------------------------------------
# 5. UTILITY FUNCTIONS
#-------------------------------------------------------------------------

def setup_experiment_configs(
    env_ids,
    algorithms,
    uncertainty_types=None,
    uncertainty_levels=None,
    seeds=None,
    total_timesteps=None
):
    """
    Generate experiment configurations from the provided parameters.
    
    Args:
        env_ids: List of environment IDs to test
        algorithms: List of algorithms to use
        uncertainty_types: List of uncertainty types to apply during training
        uncertainty_levels: List of uncertainty levels to apply during training
        seeds: List of random seeds to use
        total_timesteps: Dictionary mapping environment to training steps
        
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    
    # Default parameters
    if uncertainty_types is None:
        uncertainty_types = [None]  # No uncertainty by default
        
    if uncertainty_levels is None:
        uncertainty_levels = [0.0]  # No uncertainty by default
        
    if seeds is None:
        seeds = [42]  # Default seed
        
    # Default training steps for different environments
    if total_timesteps is None:
        # total_timesteps = {
        #     "CartPole-v1": 50_000,
        #     "LunarLander-v3": 500_000,
        #     "Pendulum-v1": 200_000,
        #     "HalfCheetah-v5": 1_000_000,
        #     "Hopper-v5": 1_000_000,
        #     "Walker2d-v4": 1_000_000,
        #     "Ant-v5": 1_000_000,
        #     "Humanoid-v5": 2_000_000,
        #     "HumanoidStandup-v4": 2_000_000,
        #     "BipedalWalker-v3": 1_000_000
        # }

        total_timesteps = {
            "CartPole-v1": 50_0,
            "LunarLander-v3": 500_0,
            "Pendulum-v1": 200_0,
            "HalfCheetah-v5": 1_000,
            "Hopper-v5": 1_000,
            "Walker2d-v4": 1_000,
            "Ant-v5": 1_000,
            "Humanoid-v5": 5_000,
            "HumanoidStandup-v4": 2_000,
            "BipedalWalker-v3": 1_000
        }
    
    # Generate all combinations
    for env_id in env_ids:
        # Get total timesteps for this environment
        steps = total_timesteps.get(env_id, 500_000)
        
        for algorithm in algorithms:
            for uncertainty_type in uncertainty_types:
                for uncertainty_level in uncertainty_levels:
                    for seed in seeds:
                        config = ExperimentConfig(
                            env_id=env_id,
                            algorithm=algorithm,
                            uncertainty_type=uncertainty_type,
                            uncertainty_level=uncertainty_level,
                            seed=seed,
                            total_timesteps=steps
                        )
                        configs.append(config)
                        
    return configs


def load_trained_agent(config):
    """
    Load a trained agent from disk.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Loaded agent
    """
    model_path = config.get_model_path()
    
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"No trained model found at {model_path}")
        
    # Load based on algorithm
    algorithm = config.algorithm.upper()
    
    if algorithm == "PPO":
        return PPO.load(model_path)
    elif algorithm == "DQN":
        return DQN.load(model_path)
    elif algorithm == "SAC":
        return SAC.load(model_path)
    elif algorithm == "TD3":
        return TD3.load(model_path)
    elif algorithm == "A2C":
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


#-------------------------------------------------------------------------
# 6. EXAMPLE USAGE
#-------------------------------------------------------------------------

def run_full_benchmark():
    """Run a comprehensive benchmark on various environments and algorithms."""
    # Classic control environments
    classic_envs = [
        "CartPole-v1",      # Discrete actions
        "Pendulum-v1",      # Continuous actions
        "LunarLander-v3"    # More complex
    ]
    
    # MuJoCo environments
    mujoco_envs = [
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v4",
        "Ant-v5"
    ]
    
    # Algorithms for different action spaces
    discrete_algorithms = ["PPO", "DQN", "A2C"]
    continuous_algorithms = ["PPO", "SAC", "TD3"]
    
    # Uncertainty types and levels for evaluation
    uncertainty_types = [
        UncertaintyType.NONE,
        UncertaintyType.EPISTEMIC,
        UncertaintyType.ALEATORIC,
        UncertaintyType.ADVERSARIAL
    ]
    
    uncertainty_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    # Generate configurations for classic control environments
    classic_configs = setup_experiment_configs(
        env_ids=classic_envs[:1],        # Start with simpler environments
        algorithms=discrete_algorithms,  # Use discrete algorithms for cartpole
        uncertainty_types=[None],        # Train without uncertainty first
        uncertainty_levels=[0.0],        # No uncertainty during training
        seeds=[42]
    )
    
    # Generate configurations for MuJoCo environments
    mujoco_configs = setup_experiment_configs(
        env_ids=mujoco_envs[:1],          # Start with half-cheetah
        algorithms=continuous_algorithms,  # Use continuous algorithms
        uncertainty_types=[None],          # Train without uncertainty first
        uncertainty_levels=[0.0],          # No uncertainty during training
        seeds=[42]
    )
    
    # Run experiments
    print("\nRunning classic control experiments:")
    classic_experiments, classic_results = run_comparative_experiments(
        classic_configs, 
        uncertainty_types,
        uncertainty_levels
    )
    
    print("\nRunning MuJoCo experiments:")
    mujoco_experiments, mujoco_results = run_comparative_experiments(
        mujoco_configs, 
        uncertainty_types,
        uncertainty_levels
    )
    
    return {
        'classic_experiments': classic_experiments,
        'classic_results': classic_results,
        'mujoco_experiments': mujoco_experiments,
        'mujoco_results': mujoco_results
    }


def run_targeted_experiment(env_id, algorithm, uncertainty_type, uncertainty_level):
    """
    Run a single targeted experiment.
    
    Args:
        env_id: Environment ID
        algorithm: Algorithm to use
        uncertainty_type: Type of uncertainty to apply during training
        uncertainty_level: Level of uncertainty to apply during training
    """
    config = ExperimentConfig(
        env_id=env_id,
        algorithm=algorithm,
        uncertainty_type=uncertainty_type,
        uncertainty_level=uncertainty_level
    )
    
    experiment = UncertaintyExperiment(config)
    experiment.setup()
    experiment.train()
    
    # Evaluate with different uncertainty types and levels
    results = experiment.evaluate(
        uncertainty_types=[
            UncertaintyType.NONE,
            UncertaintyType.EPISTEMIC,
            UncertaintyType.ALEATORIC,
            UncertaintyType.ADVERSARIAL
        ],
        uncertainty_levels=[0.0, 0.1, 0.2, 0.3, 0.5]
    )
    
    # Record videos
    experiment.record_video()  # Baseline
    experiment.record_video(UncertaintyType.EPISTEMIC, 0.3)  # Medium epistemic
    experiment.record_video(UncertaintyType.ALEATORIC, 0.3)  # Medium aleatoric
    experiment.record_video(UncertaintyType.ADVERSARIAL, 0.3)  # Medium adversarial
    
    return experiment, results


def main():
    """Main function for running experiments."""
    print("RL Benchmarking Under Uncertainty")
    print("=================================")
    
    # Set up experiments
    print("\nSetup: Choose an option:")
    print("1. Run targeted experiment (single environment/algorithm)")
    print("2. Run limited benchmark (fewer envs/algorithms for quicker results)")
    print("3. Run full benchmark (comprehensive testing)")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        # Run targeted experiment
        env_options = {
            "1": "CartPole-v1",
            "2": "LunarLander-v3",
            "3": "Pendulum-v1",
            "4": "HalfCheetah-v5",
            "5": "Hopper-v5",
            "6": "Walker2d-v4",
            "7": "Ant-v5",
            "8": "Humanoid-v5",
            "9": "HumanoidStandup-v4",
            "10": "BipedalWalker-v3"
        }
        
        algo_options = {
            "1": "PPO",
            "2": "DQN",
            "3": "A2C",
            "4": "SAC",
            "5": "TD3"
        }
        
        print("\nChoose environment:")
        for key, env in env_options.items():
            print(f"{key}. {env}")
            
        env_choice = input("\nEnvironment choice: ")
        env_id = env_options.get(env_choice, "CartPole-v1")
        
        print("\nChoose algorithm:")
        for key, algo in algo_options.items():
            print(f"{key}. {algo}")
            
        algo_choice = input("\nAlgorithm choice: ")
        algorithm = algo_options.get(algo_choice, "PPO")
        
        print("\nRunning targeted experiment...")
        # experiment, results = run_targeted_experiment(
        #     env_id=env_id,
        #     algorithm=algorithm,
        #     uncertainty_type=None,  # No uncertainty during training
        #     uncertainty_level=0.0
        # )

        # experiment, results = run_targeted_experiment(
        #     env_id=env_id,
        #     algorithm=algorithm,
        #     uncertainty_type=UncertaintyType.ALEATORIC,  # No uncertainty during training
        #     uncertainty_level=0.3
        # )

        experiment, results = run_targeted_experiment(
            env_id=env_id,
            algorithm=algorithm,
            uncertainty_type=UncertaintyType.ADVERSARIAL,  # No uncertainty during training
            uncertainty_level=1
        )
        
    elif choice == "2":
        # Run limited benchmark
        print("\nRunning limited benchmark...")
        
        # Classic control environments
        classic_envs = ["CartPole-v1"]
        classic_algorithms = ["PPO", "DQN"]
        
        # MuJoCo environments
        mujoco_envs = ["HalfCheetah-v5"]
        mujoco_algorithms = ["PPO", "SAC"]
        
        # Uncertainty types and levels for evaluation
        uncertainty_types = [
            UncertaintyType.NONE,
            UncertaintyType.EPISTEMIC,
            UncertaintyType.ALEATORIC
        ]
        
        uncertainty_levels = [0.0, 0.2, 0.5]
        
        # Generate configurations
        classic_configs = setup_experiment_configs(
            env_ids=classic_envs,
            algorithms=classic_algorithms,
            uncertainty_types=[None],
            uncertainty_levels=[0.0],
            seeds=[42]
        )
        
        mujoco_configs = setup_experiment_configs(
            env_ids=mujoco_envs,
            algorithms=mujoco_algorithms,
            uncertainty_types=[None],
            uncertainty_levels=[0.0],
            seeds=[42]
        )
        
        # Run experiments
        print("\nRunning classic control experiments:")
        run_comparative_experiments(
            classic_configs, 
            uncertainty_types,
            uncertainty_levels
        )
        
        print("\nRunning MuJoCo experiments:")
        run_comparative_experiments(
            mujoco_configs, 
            uncertainty_types,
            uncertainty_levels
        )
        
    elif choice == "3":
        # Run full benchmark
        print("\nRunning full benchmark...")
        results = run_full_benchmark()
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\nExperiments completed. Results are saved in:")
    print(f"- Logs: {LOG_DIR}")
    print(f"- Models: {MODELS_DIR}")
    print(f"- Videos: {VIDEOS_DIR}")
    print(f"- Episode Rewards: {EPISODE_REWARDS_DIR}")


if __name__ == "__main__":
    main()