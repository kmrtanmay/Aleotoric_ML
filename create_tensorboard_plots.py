import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def extract_tensorboard_data(log_path, output_dir='./tb_exported_data'):
    """
    Extract data from TensorBoard logs to CSV files and create plots
    
    Args:
        log_path: Path to the TensorBoard log file
        output_dir: Directory to save extracted data and plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the event file
    print(f"Loading TensorBoard log from: {log_path}")
    ea = event_accumulator.EventAccumulator(log_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalar events
            event_accumulator.TENSORS: 0,  # Load all tensor events
            event_accumulator.HISTOGRAMS: 0,  # Load all histogram events
        })
    ea.Reload()  # Load all data
    
    # Get list of all available tags (metrics)
    tags = ea.Tags()
    print(f"Found {len(tags['scalars'])} scalar metrics")
    print(f"Available tags: {tags['scalars']}")
    
    # Create a dictionary to store all scalar data
    data = {}
    
    # Extract scalar data for each tag
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'step': [event.step for event in events],
            'value': [event.value for event in events],
            'wall_time': [event.wall_time for event in events]
        }
        
        # Convert to DataFrame for easier manipulation
        data[tag] = pd.DataFrame(data[tag])
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.csv")
        data[tag].to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
    
    # Create plots
    create_plots(data, tags, output_dir)
    
    return data, tags

def create_plots(data, tags, output_dir):
    """Create all plots based on the extracted data"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Mean Rewards by Uncertainty Type and Level
    mean_reward_tags = [tag for tag in tags['scalars'] if 'mean_reward' in tag]
    if mean_reward_tags:
        plt.figure(figsize=(12, 8))
        for tag in mean_reward_tags:
            plt.plot(data[tag]['step'], data[tag]['value'], label=tag.replace('eval/', ''))
        
        plt.xlabel('Steps')
        plt.ylabel('Mean Reward')
        plt.title('Mean Rewards by Uncertainty Type and Level')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'mean_rewards.png'), dpi=300)
        plt.close()
        print("Created mean rewards plot")
    
    # 2. Robustness Gap Analysis
    robustness_tags = [tag for tag in tags['scalars'] if 'robustness_gap' in tag]
    if robustness_tags:
        plt.figure(figsize=(12, 8))
        for tag in robustness_tags:
            plt.plot(data[tag]['step'], data[tag]['value'], label=tag.replace('eval/', ''))
        
        plt.xlabel('Steps')
        plt.ylabel('Robustness Gap')
        plt.title('Performance Degradation Under Uncertainty')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'robustness_gap.png'), dpi=300)
        plt.close()
        print("Created robustness gap plot")
    
    # 3. Recovery Time Analysis
    recovery_tags = [tag for tag in tags['scalars'] if 'recovery_time' in tag]
    if recovery_tags:
        plt.figure(figsize=(12, 8))
        for tag in recovery_tags:
            plt.plot(data[tag]['step'], data[tag]['value'], label=tag.replace('eval/', ''))
        
        plt.xlabel('Steps')
        plt.ylabel('Mean Recovery Time')
        plt.title('Recovery Time After Perturbations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'recovery_time.png'), dpi=300)
        plt.close()
        print("Created recovery time plot")
    
    # 4. Training Progress (Rollout Metrics)
    rollout_tags = [tag for tag in tags['scalars'] if 'rollout/' in tag]
    if rollout_tags:
        plt.figure(figsize=(12, 8))
        for tag in rollout_tags:
            plt.plot(data[tag]['step'], data[tag]['value'], label=tag.replace('rollout/', ''))
        
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Training Progress Metrics')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_progress.png'), dpi=300)
        plt.close()
        print("Created training progress plot")
    
    # 5. Create individual plots for each metric
    for tag in tags['scalars']:
        plt.figure(figsize=(10, 6))
        plt.plot(data[tag]['step'], data[tag]['value'])
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title(tag)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{tag.replace('/', '_')}.png"), dpi=300)
        plt.close()
    
    # 6. Try to create uncertainty comparison plots (if data exists)
    try:
        uncertainty_types = ['none', 'epistemic', 'aleatoric', 'adversarial']
        uncertainty_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        # Get the latest step where we have evaluation data
        eval_steps = []
        for tag in [t for t in tags['scalars'] if 'eval/' in t]:
            if not data[tag].empty:
                eval_steps.append(data[tag]['step'].iloc[-1])
        
        if eval_steps:
            latest_step = max(eval_steps)
            
            # Create comparative bar charts for each uncertainty level
            for level in uncertainty_levels:
                values = []
                labels = []
                
                for u_type in uncertainty_types:
                    # Skip adversarial with level 0
                    if u_type == 'adversarial' and level == 0.0:
                        continue
                        
                    # Look for tag with this uncertainty type and level
                    tag = None
                    if u_type == 'none' and level == 0.0:
                        tag = 'eval/mean_reward'
                    else:
                        tag = f'eval/{u_type}/mean_reward_{level}'
                        
                    if tag in tags['scalars'] and not data[tag].empty:
                        # Get the latest value
                        values.append(data[tag]['value'].iloc[-1])
                        labels.append(u_type)
                
                if values:
                    plt.figure(figsize=(10, 6))
                    plt.bar(labels, values)
                    plt.xlabel('Uncertainty Type')
                    plt.ylabel('Mean Reward')
                    plt.title(f'Comparison at Uncertainty Level {level}')
                    plt.grid(alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'comparison_level_{level}.png'), dpi=300)
                    plt.close()
                    print(f"Created comparison plot for level {level}")
            
            # Create uncertainty heatmap if possible
            try:
                # Create a matrix for the heatmap
                heatmap_data = np.zeros((len(uncertainty_types), len(uncertainty_levels)))
                
                # Fill the matrix with the latest reward values
                for i, u_type in enumerate(uncertainty_types):
                    for j, level in enumerate(uncertainty_levels):
                        # Skip adversarial with level 0
                        if u_type == 'adversarial' and level == 0.0:
                            heatmap_data[i, j] = np.nan
                            continue
                            
                        # Look for tag with this uncertainty type and level
                        tag = None
                        if u_type == 'none' and level == 0.0:
                            tag = 'eval/mean_reward'
                        else:
                            tag = f'eval/{u_type}/mean_reward_{level}'
                            
                        if tag in tags['scalars'] and not data[tag].empty:
                            # Get the latest value
                            heatmap_data[i, j] = data[tag]['value'].iloc[-1]
                        else:
                            heatmap_data[i, j] = np.nan
                
                # Create the heatmap
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis",
                              xticklabels=uncertainty_levels, yticklabels=uncertainty_types)
                plt.xlabel('Uncertainty Level')
                plt.ylabel('Uncertainty Type')
                plt.title('Performance Across Uncertainty Types and Levels')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'uncertainty_heatmap.png'), dpi=300)
                plt.close()
                print("Created uncertainty heatmap")
            except Exception as e:
                print(f"Could not create heatmap: {e}")
    except Exception as e:
        print(f"Could not create uncertainty comparison plots: {e}")
    
    print(f"All plots saved to {plots_dir}")

def create_combined_episode_rewards_plot(csv_dir, output_dir):
    """
    Create a plot combining episode rewards from multiple CSV files
    
    Args:
        csv_dir: Directory containing episode reward CSV files
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_episodic_rewards.csv')]
    
    if not csv_files:
        print(f"No episode reward CSV files found in {csv_dir}")
        return
    
    plt.figure(figsize=(14, 8))
    
    for csv_file in csv_files:
        try:
            # Extract experiment info from filename
            experiment_name = csv_file.replace('_episodic_rewards.csv', '')
            
            # Load data
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            
            # Plot episode rewards
            plt.plot(df['episode'], df['reward'], label=experiment_name)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards Across Experiments')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_episode_rewards.png'), dpi=300)
    plt.close()
    print(f"Created combined episode rewards plot in {output_dir}")
    
    # Also create a smoothed version
    plt.figure(figsize=(14, 8))
    
    for csv_file in csv_files:
        try:
            # Extract experiment info from filename
            experiment_name = csv_file.replace('_episodic_rewards.csv', '')
            
            # Load data
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            
            # Apply rolling average
            window_size = min(20, len(df) // 5) if len(df) > 10 else 1
            if window_size > 1:
                smoothed = df['reward'].rolling(window=window_size).mean()
                plt.plot(df['episode'][window_size-1:], smoothed[window_size-1:], label=experiment_name)
            else:
                plt.plot(df['episode'], df['reward'], label=experiment_name)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Episode Rewards Across Experiments')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothed_episode_rewards.png'), dpi=300)
    plt.close()
    print(f"Created smoothed episode rewards plot in {output_dir}")

def main():
    # Example usage
    # Replace with your actual paths
    log_path = "/n/home05/kumartanmay/main/alea_ml/Aleotoric_ML/logs_v4/Humanoid-v5_A2C_aleatoric_0.3_seed42/events.out.tfevents.1746677854.holygpu7c26201.rc.fas.harvard.edu.1002882.0"
    output_dir = "./tb_exported_data"
    episode_rewards_dir = "./episode_rewards_v4"
    
    # Extract data from TensorBoard logs
    data, tags = extract_tensorboard_data(log_path, output_dir)
    
    # Create combined episode rewards plot (if directory exists)
    if os.path.exists(episode_rewards_dir):
        create_combined_episode_rewards_plot(episode_rewards_dir, output_dir)
    else:
        print(f"Episode rewards directory not found: {episode_rewards_dir}")
    
    print(f"All data extracted and saved to {output_dir}")
    print("To use this data for custom plots:")
    print("1. Load the CSV files from the output directory")
    print("2. Create custom plots using matplotlib, seaborn, or plotly")
    print("3. Combine data from different experiments as needed")

if __name__ == "__main__":
    main()