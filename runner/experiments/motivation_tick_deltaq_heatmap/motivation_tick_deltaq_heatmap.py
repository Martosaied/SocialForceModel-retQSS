import json
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.constants import Constants

# Parameter ranges
MOTIVATION_UPDATE_DT_VALUES = [0.001,0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DELTAQ_VALUES = [-8, -7, -6, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5]

# Experiment parameters
WIDTH = 50
PEDESTRIAN_COUNT = int(50 * (50 * 0.4) * 0.3)
VOLUMES = 1

# Control whether to run simulations or just plot existing results
RUN_SIMULATIONS = False  # Set to True to run simulations, False to only plot


def motivation_tick_deltaq_heatmap():
    """
    Main function to run the motivation tick vs deltaQ heatmap experiment.
    """
    print(f"Running motivation tick vs deltaQ heatmap experiment for {PEDESTRIAN_COUNT} pedestrians...\n")
    
    if RUN_SIMULATIONS:
        for motivation_dt in MOTIVATION_UPDATE_DT_VALUES:
            for deltaq in DELTAQ_VALUES:
                print(f"Running experiment for motivation_dt: {motivation_dt}, deltaq: {deltaq}")
                run_single_experiment(motivation_dt, deltaq)
    
    # Plot the results
    print("Plotting results...")
    plot_heatmap()


def run_single_experiment(motivation_dt, deltaq):
    """
    Run a single experiment for given motivation_dt and deltaq values.
    """
    config = load_config('experiments/motivation_tick_deltaq_heatmap/config.json')
    
    # Create output directory with both parameters
    output_dir = create_output_dir(f'experiments/motivation_tick_deltaq_heatmap/results/motivation_dt_{motivation_dt}_deltaq_{deltaq}')
    print(f"Created output directory: {output_dir}")
    
    # Update configuration
    config['iterations'] = 1
    config['parameters']['N']['value'] = PEDESTRIAN_COUNT
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE
    config['parameters']['MOTIVATION_UPDATE_DT']['value'] = motivation_dt
    config['parameters']['FROM_Y']['value'] = 15
    config['parameters']['TO_Y']['value'] = 35
    
    # Save config copy
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Format tolerance values
    formatted_tolerance = np.format_float_positional(1 * 10 ** deltaq)
    formatted_abs_tolerance = np.format_float_positional(1 * 10 ** (deltaq - 3))
    
    print(f"Motivation DT: {motivation_dt}, DeltaQ: {deltaq}")
    print(f"Tolerance: {formatted_tolerance}, AbsTolerance: {formatted_abs_tolerance}")
    
    # Update model file with parameters
    model_file = '../retqss/model/helbing_not_qss.mo'
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', model_file])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', model_file])
    subprocess.run([
        'sed', '-i',
        f's/^[[:space:]]*Tolerance=[^,]*/       Tolerance={formatted_tolerance}/g',
        model_file
    ])
    subprocess.run([
        'sed', '-i',
        f's/^[[:space:]]*AbsTolerance=[^,]*/       AbsTolerance={formatted_abs_tolerance}/g',
        model_file
    ])
    
    # Compile and run
    compile_c_code()
    compile_model('helbing_not_qss')
    
    run_experiment(
        config, 
        output_dir, 
        'helbing_not_qss', 
        plot=False, 
        copy_results=False # We don't want to copy the results to the latest directory
    )
    
    copy_results_to_latest(output_dir)
    print(f"Experiment completed. Results saved in {output_dir}")


def plot_heatmap():
    """
    Create a heatmap showing clustering groups for different combinations of 
    motivation update tick (Y-axis) and deltaQ (X-axis).
    """
    print("Generating heatmap...")
    
    # Get all result directories
    results_dir = 'experiments/motivation_tick_deltaq_heatmap/results'
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        return
    
    result_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    # Initialize data matrix
    heatmap_data = np.zeros((len(MOTIVATION_UPDATE_DT_VALUES), len(DELTAQ_VALUES)))
    heatmap_std = np.zeros((len(MOTIVATION_UPDATE_DT_VALUES), len(DELTAQ_VALUES)))
    
    # Process each result directory
    for result_dir in result_dirs:
        try:
            # Parse directory name to extract parameters
            parts = result_dir.split('_')
            if len(parts) >= 4 and parts[0] == 'motivation' and parts[1] == 'dt':
                motivation_dt = float(parts[2])
                deltaq = float(parts[4])
                
                # Find indices in our parameter arrays
                if motivation_dt in MOTIVATION_UPDATE_DT_VALUES and deltaq in DELTAQ_VALUES:
                    dt_idx = MOTIVATION_UPDATE_DT_VALUES.index(motivation_dt)
                    dq_idx = DELTAQ_VALUES.index(deltaq)
                    
                    # Read metrics.csv
                    metrics_path = os.path.join(results_dir, result_dir, 'latest', 'metrics.csv')
                    if os.path.exists(metrics_path):
                        df = pd.read_csv(metrics_path)
                        clustering_groups = df['clustering_based_groups'].dropna().tolist()
                        
                        if clustering_groups:
                            mean_groups = np.mean(clustering_groups)
                            std_groups = np.std(clustering_groups, ddof=1)
                            
                            heatmap_data[dt_idx, dq_idx] = mean_groups
                            heatmap_std[dt_idx, dq_idx] = std_groups
                            
                            print(f"Processed dt={motivation_dt}, deltaq={deltaq}: {mean_groups:.2f}±{std_groups:.2f} groups")
                        else:
                            print(f"No clustering data for dt={motivation_dt}, deltaq={deltaq}")
                    else:
                        print(f"Metrics file not found for dt={motivation_dt}, deltaq={deltaq}")
                else:
                    print(f"Parameters dt={motivation_dt}, deltaq={deltaq} not in expected ranges")
        except Exception as e:
            print(f"Error processing {result_dir}: {e}")
            continue
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    
    # Create labels for axes
    deltaq_labels = [f'1e{dq}' for dq in DELTAQ_VALUES]
    motivation_labels = [f'{dt:.3f}' for dt in MOTIVATION_UPDATE_DT_VALUES]
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=deltaq_labels,
                yticklabels=motivation_labels,
                annot=True, 
                fmt='.2f',
                cmap='viridis',
                cbar_kws={'label': 'Average Clustering Groups'})
    
    plt.title('Clustering Groups: Motivation Update Tick vs DeltaQ', fontsize=16, fontweight='bold')
    plt.xlabel('DeltaQ (Tolerance)', fontsize=14)
    plt.ylabel('Motivation Update DT', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('experiments/motivation_tick_deltaq_heatmap/clustering_groups_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second heatmap showing standard deviation
    plt.figure(figsize=(14, 10))
    
    sns.heatmap(heatmap_std, 
                xticklabels=deltaq_labels,
                yticklabels=motivation_labels,
                annot=True, 
                fmt='.2f',
                cmap='plasma',
                cbar_kws={'label': 'Standard Deviation of Clustering Groups'})
    
    plt.title('Clustering Groups Standard Deviation: Motivation Update Tick vs DeltaQ', 
              fontsize=16, fontweight='bold')
    plt.xlabel('DeltaQ (Tolerance)', fontsize=14)
    plt.ylabel('Motivation Update DT', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('experiments/motivation_tick_deltaq_heatmap/clustering_groups_std_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("HEATMAP SUMMARY STATISTICS")
    print("="*80)
    
    # Find best and worst combinations
    flat_data = heatmap_data.flatten()
    valid_data = flat_data[flat_data > 0]
    
    if len(valid_data) > 0:
        min_groups = np.min(valid_data)
        max_groups = np.max(valid_data)
        
        min_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
        max_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        
        print(f"Minimum groups: {min_groups:.2f} (dt={MOTIVATION_UPDATE_DT_VALUES[min_idx[0]]}, deltaq={DELTAQ_VALUES[min_idx[1]]})")
        print(f"Maximum groups: {max_groups:.2f} (dt={MOTIVATION_UPDATE_DT_VALUES[max_idx[0]]}, deltaq={DELTAQ_VALUES[max_idx[1]]})")
        print(f"Average groups: {np.mean(valid_data):.2f}")
        print(f"Standard deviation: {np.std(valid_data):.2f}")
    
    print("="*80)
    print("Heatmaps saved as:")
    print("- clustering_groups_heatmap.png")
    print("- clustering_groups_std_heatmap.png")


if __name__ == '__main__':
    motivation_tick_deltaq_heatmap()
