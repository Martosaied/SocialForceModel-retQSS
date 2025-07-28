import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def subway_attack_rate():
    """
    Run the subway_combination model experiment and analyze attack rate evolution.
    """
    print("Running subway combination model with attack rate analysis...\n")

    # Run the experiment
    output_dir = run()

    # Analyze and plot attack rate evolution
    print("Analyzing attack rate evolution...")
    plot_attack_rate_evolution(output_dir)

    print(f"\nExperiment completed. Results saved in {output_dir}")


def run():
    """
    Run the subway_combination model experiment using config_subte_comb.json.
    """
    print("Running subway_combination model experiment...")

    # Load the subway combination configuration
    config = load_config('config_subte_comb.json')

    # Create output directory with experiment name
    output_dir = create_output_dir(
        'experiments/subway_attack_rate/results',
        'subway_combination_attack_rate'
    )
    print(f"Created output directory: {output_dir}")

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Compile the C++ code if requested
    compile_c_code()

    # Compile the model if requested
    compile_model('subway_combination')

    # Run experiment
    run_experiment(
        config,
        output_dir,
        'subway_combination',
        plot=False,
        copy_results=False
    )

    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)

    return output_dir


def plot_attack_rate_evolution(output_dir):
    """
    Plot the evolution of attack rate over time using exposed+infected and susceptible counts.

    Attack rate = (Exposed + Infected) / (Susceptible + Exposed + Infected + Recovered)
    """
    # Read the solution CSV file
    solution_file = os.path.join(output_dir, 'latest', 'solution.csv')
    if not os.path.exists(solution_file):
        print(f"Solution file not found: {solution_file}")
        return

    df = pd.read_csv(solution_file)

    # Calculate attack rate over time
    # Get state counts for each time step
    times = []
    attack_rates = []
    susceptible_counts = []
    exposed_counts = []
    infected_counts = []
    recovered_counts = []

    for index, row in df.iterrows():
        time = row['time']

        # Count particles in each state
        susceptible = 0
        exposed = 0
        infected = 0
        recovered = 0

        # Count particles by state (PS[i] columns)
        for col in df.columns:
            if col.startswith('PS[') and col.endswith(']'):
                state = row[col]
                if pd.notna(state):
                    if state == 0:  # SUSCEPTIBLE
                        susceptible += 1
                    elif state == 1:  # EXPOSED
                        exposed += 1
                    elif state in [2, 3, 4]:  # PRE_SYMPTOMATIC, SYMPTOMATIC, ASYMPTOMATIC
                        infected += 1
                    elif state == 5:  # RECOVERED
                        recovered += 1

        # Calculate attack rate
        total_population = susceptible + exposed + infected + recovered
        if total_population > 0:
            attack_rate = (exposed + infected) / total_population
        else:
            attack_rate = 0

        times.append(time)
        attack_rates.append(attack_rate)
        susceptible_counts.append(susceptible)
        exposed_counts.append(exposed)
        infected_counts.append(infected)
        recovered_counts.append(recovered)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Attack rate evolution
    ax1.plot(times, attack_rates, 'r-', linewidth=2, label='Attack Rate')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Attack Rate')
    ax1.set_title('Evolution of Attack Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot 2: Population counts by state
    ax2.plot(times, susceptible_counts, 'b-', label='Susceptible', linewidth=2)
    ax2.plot(times, exposed_counts, 'y-', label='Exposed', linewidth=2)
    ax2.plot(times, infected_counts, 'r-', label='Infected', linewidth=2)
    ax2.plot(times, recovered_counts, 'g-', label='Recovered', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Number of Individuals')
    ax2.set_title('Population Counts by State Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'latest', 'attack_rate_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Attack rate evolution plot saved to: {plot_path}")

    # Also save to the experiment results directory
    results_plot_path = os.path.join(os.path.dirname(output_dir), 'attack_rate_evolution.png')
    plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
    print(f"Attack rate evolution plot also saved to: {results_plot_path}")

    plt.close()

    # Print summary statistics
    max_attack_rate = max(attack_rates)
    max_attack_time = times[attack_rates.index(max_attack_rate)]
    final_attack_rate = attack_rates[-1]

    print(f"\nAttack Rate Analysis Summary:")
    print(f"Maximum attack rate: {max_attack_rate:.4f} at time {max_attack_time:.2f}s")
    print(f"Final attack rate: {final_attack_rate:.4f}")
    print(f"Total simulation time: {times[-1]:.2f}s")

    # Save summary to file
    summary_path = os.path.join(output_dir, 'latest', 'attack_rate_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Attack Rate Analysis Summary\n")
        f.write(f"===========================\n")
        f.write(f"Maximum attack rate: {max_attack_rate:.4f} at time {max_attack_time:.2f}s\n")
        f.write(f"Final attack rate: {final_attack_rate:.4f}\n")
        f.write(f"Total simulation time: {times[-1]:.2f}s\n")
        f.write(f"Final population counts:\n")
        f.write(f"  Susceptible: {susceptible_counts[-1]}\n")
        f.write(f"  Exposed: {exposed_counts[-1]}\n")
        f.write(f"  Infected: {infected_counts[-1]}\n")
        f.write(f"  Recovered: {recovered_counts[-1]}\n")


if __name__ == '__main__':
    subway_attack_rate_experiment()
