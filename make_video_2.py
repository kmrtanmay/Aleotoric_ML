import os
import argparse
import numpy as np
import gymnasium as gym
from enum import Enum
from stable_baselines3 import PPO, DQN, SAC, TD3, A2C
from gymnasium.wrappers import RecordVideo, TimeLimit
from gymnasium import spaces

# Set environment variable for headless rendering
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL backend for headless rendering

class UncertaintyType(str, Enum):
    """Types of uncertainty for experimentation."""
    NONE = "none"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    ADVERSARIAL = "adversarial"

class EpistemicUncertaintyWrapper(gym.Wrapper):
    """Apply epistemic uncertainty to the environment."""
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
    """Apply aleatoric uncertainty to the environment."""
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
    """Apply adversarial attacks to the environment's observations."""
    def __init__(self, env, agent_model=None, attack_type='random', epsilon=0.2, attack_freq=0.5, seed=None):
        super().__init__(env)
        self.agent_model = agent_model  # The agent's model to use for gradient-based attacks
        self.attack_type = attack_type
        self.epsilon = epsilon  # Attack strength
        self.attack_freq = attack_freq  # Frequency of attacks (probability)
        self.rng = np.random.RandomState(seed)
        
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
            
        # For simplicity, just use random attack in this script
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


def make_env_with_uncertainty(
    env_id,
    uncertainty_type=None,
    uncertainty_level=0.0,
    agent_model=None,
    seed=None,
    render_mode="rgb_array",
    max_episode_steps=1000
):
    """Create an environment with the specified uncertainty type and level."""
    # Create base environment
    env = gym.make(env_id, render_mode=render_mode)
    
    # Check if the environment already has a time limit
    if hasattr(env, '_max_episode_steps'):
        print(f"Environment already has a time limit of {env._max_episode_steps} steps")
        # We'll work with the existing time limit
        
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
    
    # Set seeds for reproducibility
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    return env

def load_model(model_path, algorithm):
    """Load trained model based on algorithm."""
    try:
        if algorithm.upper() == "PPO":
            return PPO.load(model_path)
        elif algorithm.upper() == "DQN":
            return DQN.load(model_path)
        elif algorithm.upper() == "SAC":
            return SAC.load(model_path)
        elif algorithm.upper() == "TD3":
            return TD3.load(model_path)
        elif algorithm.upper() == "A2C":
            return A2C.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def record_video(env_id, model_path, algorithm, uncertainty_type=None, uncertainty_level=0.0, 
                video_dir="./videos", max_steps=1000, fps=30):
    """Record a video of agent performance using EGL backend for headless rendering."""
    print(f"Recording video for {env_id} with {algorithm}...")
    print(f"Uncertainty type: {uncertainty_type}, level: {uncertainty_level}")
    print(f"Using MuJoCo GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    print(f"Max steps: {max_steps}, FPS: {fps}")
    
    try:
        # Create output directory
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(video_dir, f"{env_id}_{algorithm}")
        if uncertainty_type and uncertainty_type != "none":
            output_dir += f"_{uncertainty_type}_{uncertainty_level}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Video will be saved to: {output_dir}")
        
        # Load the model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, algorithm)
        print("Model loaded successfully!")
        
        # Create environment
        print("Creating environment...")
        uncertainty_type_enum = None if uncertainty_type is None or uncertainty_type == "none" else UncertaintyType(uncertainty_type)
        env = make_env_with_uncertainty(
            env_id=env_id,
            uncertainty_type=uncertainty_type_enum,
            uncertainty_level=uncertainty_level,
            agent_model=model if uncertainty_type == "adversarial" else None,
            seed=42,
            render_mode="rgb_array",
            max_episode_steps=max_steps
        )
        
        print("Setting up video recorder...")
        video_name = f"{uncertainty_type or 'baseline'}_{uncertainty_level}"
        env = RecordVideo(
            env, 
            output_dir,
            episode_trigger=lambda x: True,
            name_prefix=video_name,
            step_trigger=None,  # Only record at episode start
            video_length=max_steps * 2,  # Make sure enough frames are recorded
            fps=fps,  # Set frame rate
            disable_logger=False  # Enable logging for debugging
        )
        
        # Run and record episode
        print("Running episode...")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            
            # Print debugging info periodically
            if step_count % 100 == 0:
                print(f"Step {step_count}/{max_steps}: Action shape {action.shape}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print detailed status periodically
            if step_count % 100 == 0 or terminated or truncated:
                print(f"Step {step_count}/{max_steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                if info.get('TimeLimit.truncated', False):
                    print(f"TimeLimit truncated: {info['TimeLimit.truncated']}")
            
            done = terminated or truncated
            
        # Make sure to close the environment to finish video writing
        env.close()
        
        print(f"Episode finished after {step_count} steps with total reward: {total_reward}")
        print(f"Video saved to {output_dir}")
        
    except Exception as e:
        print(f"Error recording video: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Record videos of trained agents")
    parser.add_argument("--env", type=str, required=True, help="Environment ID")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--algorithm", type=str, required=True, choices=["PPO", "DQN", "SAC", "TD3", "A2C"])
    parser.add_argument("--uncertainty", type=str, default=None, choices=["none", "epistemic", "aleatoric", "adversarial"])
    parser.add_argument("--level", type=float, default=0.0, help="Uncertainty level")
    parser.add_argument("--video-dir", type=str, default="./videos", help="Directory to save videos")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    
    args = parser.parse_args()
    
    # Setting MUJOCO_GL one more time to ensure it's set
    os.environ['MUJOCO_GL'] = 'egl'
    
    record_video(
        args.env, 
        args.model, 
        args.algorithm, 
        args.uncertainty, 
        args.level, 
        args.video_dir, 
        args.max_steps,
        args.fps
    )

if __name__ == "__main__":
    main()