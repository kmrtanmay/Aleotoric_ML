import os
import sys
import numpy as np
import pygame
import gymnasium as gym
import torch
from enum import Enum
from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import time

# Add your UncertaintyType class and wrappers
class UncertaintyType(str, Enum):
    """Types of uncertainty for experimentation."""
    NONE = "none"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    ADVERSARIAL = "adversarial"

# Import gym spaces
from gym import spaces

class EpistemicUncertaintyWrapper(gym.Wrapper):
    """
    Apply epistemic uncertainty to the environment by introducing
    model uncertainty through limited or modified state information.
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
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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


# Define device for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants for UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 120, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
YELLOW = (255, 255, 80)

BUTTON_HEIGHT = 40
BUTTON_WIDTH = 200
SLIDER_HEIGHT = 30
SLIDER_WIDTH = 300
SLIDER_KNOB_RADIUS = 10
PADDING = 20

# Function to make environment with uncertainty
def make_env_with_uncertainty(
    env_id: str,
    uncertainty_type: Optional[UncertaintyType] = None,
    uncertainty_level: float = 0.0,
    agent_model=None,
    seed: int = None,
    render_mode: Optional[str] = 'rgb_array',
):
    """Create an environment with the specified uncertainty type and level."""
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
            
    # Set seeds for reproducibility
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    return env

# Helper class for UI buttons
class Button:
    def __init__(self, x, y, width, height, text, action=None, selected=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.selected = selected
        self.hover = False
        
    def draw(self, surface, font):
        # Determine colors based on state
        if self.selected:
            bg_color = BLUE
            text_color = WHITE
        elif self.hover:
            bg_color = LIGHT_GRAY
            text_color = BLACK
        else:
            bg_color = GRAY
            text_color = BLACK
            
        # Draw button
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=5)  # Border
        
        # Draw text
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
            
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.action:
                return self.action()
        return None

# Helper class for UI sliders
class Slider:
    def __init__(self, x, y, width, height, min_value, max_value, initial_value, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.text = text
        self.action = action
        self.dragging = False
        self.knob_pos = self._value_to_pos(initial_value)
        
    def _value_to_pos(self, value):
        # Convert value to knob position
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return int(self.rect.x + ratio * self.rect.width)
    
    def _pos_to_value(self, pos):
        # Convert knob position to value
        ratio = (pos - self.rect.x) / self.rect.width
        return self.min_value + ratio * (self.max_value - self.min_value)
    
    def draw(self, surface, font):
        # Draw slider track
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect, border_radius=5)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=5)  # Border
        
        # Draw knob
        knob_rect = pygame.Rect(
            self.knob_pos - SLIDER_KNOB_RADIUS, 
            self.rect.centery - SLIDER_KNOB_RADIUS,
            2 * SLIDER_KNOB_RADIUS, 
            2 * SLIDER_KNOB_RADIUS
        )
        pygame.draw.circle(surface, BLUE, (self.knob_pos, self.rect.centery), SLIDER_KNOB_RADIUS)
        pygame.draw.circle(surface, BLACK, (self.knob_pos, self.rect.centery), SLIDER_KNOB_RADIUS, 2)
        
        # Draw text and value
        text_surf = font.render(f"{self.text}: {self.value:.2f}", True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.x, self.rect.y - 15))
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            knob_rect = pygame.Rect(
                self.knob_pos - SLIDER_KNOB_RADIUS, 
                self.rect.centery - SLIDER_KNOB_RADIUS,
                2 * SLIDER_KNOB_RADIUS, 
                2 * SLIDER_KNOB_RADIUS
            )
            if knob_rect.collidepoint(event.pos):
                self.dragging = True
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                if self.action:
                    return self.action(self.value)
                
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = event.pos[0]
            # Constrain to slider bounds
            mouse_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
            self.knob_pos = mouse_x
            self.value = self._pos_to_value(mouse_x)
            if self.action:
                return self.action(self.value)
                
        return None

# Class for handling metrics and plots
class MetricsDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont('Arial', 14)
        
        # Metrics to track
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
        self.avg_reward = 0
        self.avg_length = 0
        self.uncertainty_type = "none"
        self.uncertainty_level = 0.0
        
        # Metrics history for different uncertainty configurations
        self.history = {}
        
    def update_metrics(self, reward, done, uncertainty_type, uncertainty_level):
        """Update metrics with latest step info."""
        self.current_reward += reward
        self.current_length += 1
        
        # Handle episode completion
        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            # Calculate averages
            self.avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            self.avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
            
            # Store metrics by uncertainty configuration
            config_key = f"{uncertainty_type}_{uncertainty_level:.2f}"
            if config_key not in self.history:
                self.history[config_key] = {
                    'rewards': [],
                    'lengths': []
                }
                
            self.history[config_key]['rewards'].append(self.current_reward)
            self.history[config_key]['lengths'].append(self.current_length)
            
            # Reset current episode metrics
            self.current_reward = 0
            self.current_length = 0
            
        # Update current uncertainty settings
        self.uncertainty_type = uncertainty_type
        self.uncertainty_level = uncertainty_level
    
    def generate_performance_plot(self):
        """Generate performance comparison plot for different uncertainty settings."""
        fig, ax = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100), plt.gca()
        
        # Plot rewards for each uncertainty configuration
        for config, data in self.history.items():
            if not data['rewards']:
                continue
                
            uncertainty_type, level = config.split('_')
            label = f"{uncertainty_type} ({level})"
            
            # Calculate moving average if enough data points
            rewards = data['rewards']
            if len(rewards) > 5:
                # Simple moving average
                window_size = min(5, len(rewards))
                avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes = range(window_size-1, len(rewards))
                ax.plot(episodes, avg_rewards, label=label)
            else:
                # Just plot raw data if not enough for moving average
                ax.plot(range(len(rewards)), rewards, label=label)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Performance Under Different Uncertainty Settings')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Convert plot to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        
        return plot_surface
        
    def draw(self, surface, x, y):
        """Draw metrics display."""
        self.surface.fill(WHITE)
        
        # Draw text metrics
        metrics_text = [
            f"Uncertainty Type: {self.uncertainty_type}",
            f"Uncertainty Level: {self.uncertainty_level:.2f}",
            f"Current Episode Reward: {self.current_reward:.2f}",
            f"Current Episode Length: {self.current_length}",
            f"Avg Reward (last 10 episodes): {self.avg_reward:.2f}",
            f"Avg Episode Length (last 10): {self.avg_length:.2f}",
            f"Total Episodes: {len(self.episode_rewards)}"
        ]
        
        for i, text in enumerate(metrics_text):
            text_surf = self.font.render(text, True, BLACK)
            self.surface.blit(text_surf, (10, 10 + i * 25))
            
        # Generate and display performance plot if we have data
        if self.history and any(len(data['rewards']) > 0 for data in self.history.values()):
            plot_y_offset = 10 + len(metrics_text) * 25 + 20  # Position below text metrics
            plot_surface = self.generate_performance_plot()
            self.surface.blit(plot_surface, (10, plot_y_offset))
            
        # Blit to main surface
        surface.blit(self.surface, (x, y))


class UncertaintyDemoApp:
    def __init__(self, env_id, model_path, model_type='PPO'):
        pygame.init()
        
        # Set up display
        self.info_width = 600
        self.window_width = 1280
        self.window_height = 720
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"RL Agent Under Uncertainty: {env_id}")
        
        # Load fonts
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Load RL model
        self.model_type = model_type
        self.load_model(model_path)
        
        # Set up environment
        self.env_id = env_id
        self.uncertainty_type = UncertaintyType.NONE
        self.uncertainty_level = 0.0
        self.env = make_env_with_uncertainty(
            env_id=env_id,
            uncertainty_type=self.uncertainty_type,
            uncertainty_level=self.uncertainty_level,
            agent_model=self.model,
            seed=42,
        )
        
        self.observation, self.info = self.env.reset()
        self.frame = None
        self.current_reward = 0
        self.done = False
        self.paused = False
        
        # Set up UI elements
        self.setup_ui()
        
        # Set up metrics
        self.metrics = MetricsDisplay(self.info_width - 20, 500)
        
        # Timing
        self.clock = pygame.time.Clock()
        self.fps = 30
        
    def load_model(self, model_path):
        """Load the RL model based on specified type."""
        if self.model_type == 'PPO':
            self.model = PPO.load(model_path)
        elif self.model_type == 'SAC':
            self.model = SAC.load(model_path)
        elif self.model_type == 'TD3':
            self.model = TD3.load(model_path)
        elif self.model_type == 'A2C':
            self.model = A2C.load(model_path)
        elif self.model_type == 'DQN':
            self.model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        print(f"Loaded {self.model_type} model from {model_path}")
    
    def setup_ui(self):
        """Set up UI elements for controlling the demo."""
        self.ui_elements = []
        
        # Uncertainty type buttons
        button_y = 50
        self.uncertainty_buttons = []
        
        for i, u_type in enumerate(UncertaintyType):
            selected = (u_type == self.uncertainty_type)
            button = Button(
                self.window_width - self.info_width + PADDING,
                button_y + i * (BUTTON_HEIGHT + 10),
                BUTTON_WIDTH,
                BUTTON_HEIGHT,
                f"Type: {u_type.value}",
                action=lambda t=u_type: self.set_uncertainty_type(t),
                selected=selected
            )
            self.uncertainty_buttons.append(button)
            self.ui_elements.append(button)
            
        # Uncertainty level slider
        slider_y = button_y + 4 * (BUTTON_HEIGHT + 10) + 20
        self.uncertainty_slider = Slider(
            self.window_width - self.info_width + PADDING,
            slider_y,
            SLIDER_WIDTH,
            SLIDER_HEIGHT,
            0.0,
            1.0,
            self.uncertainty_level,
            "Uncertainty Level",
            action=self.set_uncertainty_level
        )
        self.ui_elements.append(self.uncertainty_slider)
        
        # Replay speed slider
        speed_slider_y = slider_y + SLIDER_HEIGHT + 40
        self.speed_slider = Slider(
            self.window_width - self.info_width + PADDING,
            speed_slider_y,
            SLIDER_WIDTH,
            SLIDER_HEIGHT,
            1,
            60,
            self.fps,
            "Simulation Speed (FPS)",
            action=self.set_simulation_speed
        )
        self.ui_elements.append(self.speed_slider)
        
        # Control buttons
        control_button_y = speed_slider_y + SLIDER_HEIGHT + 40
        
        # Reset button
        self.reset_button = Button(
            self.window_width - self.info_width + PADDING,
            control_button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Reset Environment",
            action=self.reset_environment
        )
        self.ui_elements.append(self.reset_button)
        
        # Pause button
        self.pause_button = Button(
            self.window_width - self.info_width + PADDING + BUTTON_WIDTH + 20,
            control_button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Pause/Resume",
            action=self.toggle_pause
        )
        self.ui_elements.append(self.pause_button)
        
    def set_uncertainty_type(self, uncertainty_type):
        """Set the uncertainty type and update environment."""
        self.uncertainty_type = uncertainty_type
        
        # Update button selection
        for button in self.uncertainty_buttons:
            button.selected = (button.text == f"Type: {uncertainty_type.value}")
            
        # Recreate environment with new uncertainty
        self.env.close()
        self.env = make_env_with_uncertainty(
            env_id=self.env_id,
            uncertainty_type=self.uncertainty_type,
            uncertainty_level=self.uncertainty_level,
            agent_model=self.model,
            seed=42,
        )
        self.observation, self.info = self.env.reset()
        self.done = False
        
        print(f"Set uncertainty type to {uncertainty_type.value}")
        return uncertainty_type
    
    def set_uncertainty_level(self, level):
        """Set the uncertainty level and update environment."""
        self.uncertainty_level = level
        
        # Recreate environment with new uncertainty level
        self.env.close()
        self.env = make_env_with_uncertainty(
            env_id=self.env_id,
            uncertainty_type=self.uncertainty_type,
            uncertainty_level=self.uncertainty_level,
            agent_model=self.model,
            seed=42,
        )
        self.observation, self.info = self.env.reset()
        self.done = False
        
        print(f"Set uncertainty level to {level:.2f}")
        return level
    
    def set_simulation_speed(self, fps):
        """Set the simulation speed (FPS)."""
        self.fps = int(fps)
        print(f"Set simulation speed to {self.fps} FPS")
        return fps
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        print(f"{'Paused' if self.paused else 'Resumed'} simulation")
        return self.paused
    
    def reset_environment(self):
        """Reset the environment."""
        self.env.close()
        self.env = make_env_with_uncertainty(
            env_id=self.env_id,
            uncertainty_type=self.uncertainty_type,
            uncertainty_level=self.uncertainty_level,
            agent_model=self.model,
            seed=42,
        )
        self.observation, self.info = self.env.reset()
        self.done = False
        self.current_reward = 0
        print("Reset environment")
        return True
    
    def update(self):
        """Update the environment and agent."""
        if self.paused or self.done:
            return
            
        # Get action from model
        action, _states = self.model.predict(self.observation, deterministic=True)
        
        # Step the environment
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.current_reward += reward
        
        # Update metrics
        self.metrics.update_metrics(
            reward, 
            self.done, 
            self.uncertainty_type.value, 
            self.uncertainty_level
        )
        
        # Get rendered frame
        self.frame = self.env.render()
        
        # Reset if episode is done
        if self.done:
            print(f"Episode ended with reward: {self.current_reward:.2f}")
            self.observation, self.info = self.env.reset()
            self.current_reward = 0
    
    def run(self):
        """Main loop for the demo."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                # Handle UI events
                for ui_element in self.ui_elements:
                    ui_element.handle_event(event)
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Update simulation
            self.update()
            
            # Draw environment frame
            if self.frame is not None:
                # Convert numpy array to pygame surface
                frame_surface = pygame.surfarray.make_surface(
                    np.transpose(self.frame, (1, 0, 2))
                )
                
                # Calculate position and size for display
                env_width = self.window_width - self.info_width
                env_height = self.window_height
                frame_ratio = frame_surface.get_width() / frame_surface.get_height()
                
                if env_width / env_height > frame_ratio:
                    # Constrained by height
                    display_height = env_height
                    display_width = int(display_height * frame_ratio)
                else:
                    # Constrained by width
                    display_width = env_width
                    display_height = int(display_width / frame_ratio)
                
                # Center in available space
                x_offset = (env_width - display_width) // 2
                y_offset = (env_height - display_height) // 2
                
                # Scale and display
                frame_surface = pygame.transform.scale(frame_surface, (display_width, display_height))
                self.screen.blit(frame_surface, (x_offset, y_offset))
            
            # Draw UI panel background
            ui_panel = pygame.Rect(self.window_width - self.info_width, 0, self.info_width, self.window_height)
            pygame.draw.rect(self.screen, LIGHT_GRAY, ui_panel)
            pygame.draw.line(self.screen, BLACK, (self.window_width - self.info_width, 0), 
                            (self.window_width - self.info_width, self.window_height), 2)
            
            # Draw title
            title_text = self.title_font.render(f"RL Agent Under Uncertainty", True, BLACK)
            subtitle_text = self.font.render(f"Environment: {self.env_id}, Model: {self.model_type}", True, DARK_GRAY)
            self.screen.blit(title_text, (self.window_width - self.info_width + PADDING, 10))
            self.screen.blit(subtitle_text, (self.window_width - self.info_width + PADDING, 35))
            
            # Draw UI elements
            for ui_element in self.ui_elements:
                ui_element.draw(self.screen, self.font)
                
            # Draw metrics
            metrics_y = 300  # Adjust based on your UI layout
            self.metrics.draw(
                self.screen, 
                self.window_width - self.info_width + 10, 
                metrics_y
            )
            
            # Update display
            pygame.display.flip()
            
            # Cap the framerate
            self.clock.tick(self.fps)
            
        # Clean up
        pygame.quit()
        self.env.close()


def main():
    """Run the uncertainty demo with specified environment and model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive RL Agent Under Uncertainty Demo')
    parser.add_argument('--env', type=str, default='Humanoid-v5', help='Gym environment ID')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3', 'A2C', 'DQN'], 
                        help='Type of RL algorithm')
    
    args = parser.parse_args()
    
    app = UncertaintyDemoApp(
        env_id=args.env,
        model_path=args.model_path,
        model_type=args.model_type
    )
    app.run()


if __name__ == "__main__":
    main()