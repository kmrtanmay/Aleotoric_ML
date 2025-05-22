# Reinforcement Learning Under Uncertainty Benchmarking Framework

A comprehensive framework for training and evaluating reinforcement learning agents under different types of uncertainty. This framework supports epistemic, aleatoric, and adversarial uncertainties across various RL algorithms and environments.

## 🎯 Overview

This benchmarking framework allows you to:
- Train RL agents using standard algorithms (PPO, DQN, SAC, TD3, A2C)
- Apply different types of uncertainty during training and evaluation
- Evaluate robustness across uncertainty types and levels
- Generate videos of agent performance under uncertainty
- Create comprehensive visualizations and analysis

## 📋 Features

### Uncertainty Types
- **Epistemic Uncertainty**: Model/parameter uncertainty through state masking and dropout
- **Aleatoric Uncertainty**: Environmental stochasticity via observation and action noise
- **Adversarial Attacks**: Perturbations to observations (FGSM and random attacks)
- **Baseline**: Clean environment without uncertainty

### Supported Algorithms
- **PPO** (Proximal Policy Optimization)
- **DQN** (Deep Q-Network)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **A2C** (Advantage Actor-Critic)

### Supported Environments
- **Classic Control**: CartPole-v1, Pendulum-v1, LunarLander-v3
- **MuJoCo**: HalfCheetah-v5, Hopper-v5, Walker2d-v4, Ant-v5, Humanoid-v5

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install stable-baselines3[extra]
pip install gymnasium[mujoco]
pip install matplotlib pandas seaborn tensorboard
```

### Basic Usage

1. **Train an agent** with uncertainty:
```python
from rl_agent_v2 import UncertaintyExperiment, ExperimentConfig, UncertaintyType

# Create configuration
config = ExperimentConfig(
    env_id="CartPole-v1",
    algorithm="PPO",
    uncertainty_type=UncertaintyType.ALEATORIC,
    uncertainty_level=0.2,
    total_timesteps=50_000
)

# Run experiment
experiment = UncertaintyExperiment(config)
experiment.setup().train()
```

2. **Generate videos** of trained agents:
```bash
python make_video_v2.py --env CartPole-v1 \
                        --model models_v4/CartPole-v1_PPO_baseline_0.0_seed42 \
                        --algorithm PPO \
                        --uncertainty aleatoric \
                        --level 0.2
```

3. **Create visualizations** from TensorBoard logs:
```python
from create_tensorboard_plots import extract_tensorboard_data

log_path = "logs_v4/experiment_name/events.out.tfevents.*"
data, tags = extract_tensorboard_data(log_path, "./exported_data")
```

## 📁 Project Structure

```
├── rl_agent_v2.py              # Main framework code
├── make_video_v2.py            # Video generation script
├── create_tensorboard_plots.py # Visualization utilities
├── logs_v4/                    # Training logs and TensorBoard data
├── models_v4/                  # Saved trained models
├── videos/                     # Generated videos
├── episode_rewards_v4/         # Episode reward data
└── tb_exported_data/           # Exported TensorBoard data
```

## 🔧 Main Components

### 1. Uncertainty Wrappers

#### EpistemicUncertaintyWrapper
Simulates model uncertainty through:
- Random state dimension masking
- Dropout-like noise application
- Information restriction scenarios

#### AleatoricUncertaintyWrapper
Adds environmental stochasticity via:
- Gaussian/uniform/Laplace noise to observations
- Action execution noise
- Reward randomness

#### AdversarialAttackWrapper
Applies adversarial perturbations:
- FGSM (Fast Gradient Sign Method) attacks
- Random noise attacks
- Configurable attack frequency and strength

### 2. Experiment Configuration

```python
@dataclass
class ExperimentConfig:
    env_id: str                    # Environment ID
    algorithm: str                 # RL algorithm
    uncertainty_type: str          # Type of uncertainty
    uncertainty_level: float       # Uncertainty strength (0.0-1.0)
    seed: int                      # Random seed
    total_timesteps: int          # Training duration
    eval_freq: int                # Evaluation frequency
    n_eval_episodes: int          # Episodes per evaluation
```

### 3. Evaluation Metrics

The framework tracks:
- **Mean episode rewards** under different uncertainty conditions
- **Robustness gap**: Performance difference between clean and uncertain environments
- **Recovery time**: Time to recover after perturbations
- **Success/failure rates** based on environment-specific thresholds

## 📊 Usage Examples

### Example 1: Single Environment Benchmark

```python
# Train PPO on CartPole with aleatoric uncertainty
config = ExperimentConfig(
    env_id="CartPole-v1",
    algorithm="PPO",
    uncertainty_type="aleatoric",
    uncertainty_level=0.3,
    total_timesteps=50_000
)

experiment = UncertaintyExperiment(config)
experiment.setup().train()

# Evaluate across uncertainty types and levels
results = experiment.evaluate(
    uncertainty_types=[UncertaintyType.NONE, UncertaintyType.EPISTEMIC, 
                      UncertaintyType.ALEATORIC, UncertaintyType.ADVERSARIAL],
    uncertainty_levels=[0.0, 0.1, 0.2, 0.3, 0.5]
)
```

### Example 2: Comparative Study

```python
from rl_agent_v2 import setup_experiment_configs, run_comparative_experiments

# Generate configurations for multiple algorithms
configs = setup_experiment_configs(
    env_ids=["CartPole-v1", "LunarLander-v3"],
    algorithms=["PPO", "DQN"],
    uncertainty_types=[None],  # Train without uncertainty
    uncertainty_levels=[0.0],
    seeds=[42, 123, 456]
)

# Run comparative experiments
experiments, results = run_comparative_experiments(
    configs=configs,
    uncertainty_types=[UncertaintyType.NONE, UncertaintyType.ALEATORIC],
    uncertainty_levels=[0.0, 0.2, 0.5]
)
```

### Example 3: Video Generation

```bash
# Generate baseline video
python make_video_v2.py --env HalfCheetah-v5 \
                        --model models_v4/HalfCheetah-v5_SAC_baseline_0.0_seed42 \
                        --algorithm SAC

# Generate video with epistemic uncertainty
python make_video_v2.py --env HalfCheetah-v5 \
                        --model models_v4/HalfCheetah-v5_SAC_baseline_0.0_seed42 \
                        --algorithm SAC \
                        --uncertainty epistemic \
                        --level 0.3 \
                        --max-steps 1000
```

## 🎮 Interactive Usage

Run the main script for an interactive experience:

```bash
python rl_agent_v2.py
```

Choose from:
1. **Targeted experiment**: Single environment/algorithm combination
2. **Limited benchmark**: Quick comparison across select environments
3. **Full benchmark**: Comprehensive testing across all supported environments

## 📈 Data Analysis

### TensorBoard Integration
The framework automatically logs metrics to TensorBoard:
```bash
tensorboard --logdir logs_v4
```

### Export and Visualization
Extract data from TensorBoard logs for custom analysis:
```python
from create_tensorboard_plots import extract_tensorboard_data, create_combined_episode_rewards_plot

# Extract all metrics
data, tags = extract_tensorboard_data("logs_v4/experiment/events.out.tfevents.*")

# Create combined episode reward plots
create_combined_episode_rewards_plot("episode_rewards_v4", "./plots")
```

### Generated Plots
The framework automatically creates:
- Mean reward comparisons across uncertainty types
- Robustness gap analysis
- Recovery time metrics
- Training progress curves
- Uncertainty heatmaps

## ⚙️ Configuration Options

### Environment-Specific Settings
The framework includes optimized settings for different environments:
- **CartPole-v1**: 50,000 timesteps, success threshold 475
- **LunarLander-v3**: 500,000 timesteps, success threshold 200
- **HalfCheetah-v5**: 1,000,000 timesteps, success threshold 4000
- **Humanoid-v5**: 2,000,000 timesteps, success threshold 5000

### Uncertainty Level Guidelines
- **0.0**: No uncertainty (baseline)
- **0.1**: Low uncertainty
- **0.2**: Medium uncertainty
- **0.3**: High uncertainty
- **0.5**: Very high uncertainty

## 🔍 Advanced Features

### Custom Uncertainty Types
Extend the framework by implementing custom uncertainty wrappers:
```python
class CustomUncertaintyWrapper(gym.Wrapper):
    def __init__(self, env, uncertainty_level=0.2):
        super().__init__(env)
        self.uncertainty_level = uncertainty_level
    
    def step(self, action):
        # Implement custom uncertainty logic
        pass
```

### Model Loading and Evaluation
Load pre-trained models for evaluation:
```python
from rl_agent_v2 import load_trained_agent

config = ExperimentConfig(...)
agent = load_trained_agent(config)

# Create evaluation environment
experiment = UncertaintyExperiment(config)
experiment.agent = agent
results = experiment.evaluate()
```

## 🤝 Contributing

To extend the framework:
1. Add new uncertainty types in the uncertainty wrappers section
2. Implement additional RL algorithms in the experiment setup
3. Add new environments with appropriate success/failure thresholds
4. Enhance visualization capabilities in the plotting utilities

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{rl_uncertainty_framework,
  title={Reinforcement Learning Under Uncertainty Benchmarking Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/rl-uncertainty-framework}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

- **stable-baselines3**: RL algorithms implementation
- **gymnasium**: Environment interface
- **mujoco**: Physics simulation for continuous control
- **tensorboard**: Logging and visualization
- **matplotlib/seaborn**: Static plotting
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the code comments
- Review the example configurations and usage patterns

---

*This framework provides a comprehensive platform for studying RL agent robustness under uncertainty. Use it to benchmark algorithms, analyze failure modes, and develop more robust RL systems.*