import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Clustering import Clustering
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

PEDESTRIAN_COUNT = int(20 * 50 * 0.3)
WIDTH = 20
VOLUMES = 50

class LanesByIterationsExperiment:
    """
    Enhanced experiment class for analyzing lane formation evolution across iterations.
    """
    
    def __init__(self):
        self.results_dir = Path('experiments/lanes_by_iterations/results')
        self.figures_dir = Path('experiments/lanes_by_iterations/figures')
        self.figures_dir.mkdir(exist_ok=True)
        
    def run(self):
        """Run the complete experiment."""
        print(f"Running iterations for {PEDESTRIAN_COUNT} pedestrians and plotting lanes by iteration...\n")
        self._run_simulation()
        print("Creating enhanced visualizations...")
        self._create_enhanced_plots()
        
    def _run_simulation(self):
        """Run the simulation experiment."""
        config = load_config('experiments/lanes_by_iterations/config.json')

        # Create output directory with experiment name if provided
        output_dir = create_output_dir('experiments/lanes_by_iterations/results')
        print(f"Created output directory: {output_dir}")

        config['parameters']['N']['value'] = PEDESTRIAN_COUNT
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC

        # Replace the map in the config
        generated_map = generate_map(VOLUMES, WIDTH)
        # config['parameters']['OBSTACLES'] = {
        #   "name": "OBSTACLES",
        #   "type": "map",
        #   "map": generated_map
        # }

        # Add from where to where pedestrians are generated
        config['parameters']['FROM_Y'] = {
          "name": "FROM_Y",
          "type": "value",
          "value": (VOLUMES/ 2) - int(WIDTH / 2)
        }
        config['parameters']['TO_Y'] = {
          "name": "TO_Y",
          "type": "value",
          "value": (VOLUMES/ 2) + int(WIDTH / 2)
        }

        # Save config copy in experiment directory
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

        subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/social_force_model.mo'])
        subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/social_force_model.mo'])

        # Compile the C++ code if requested
        compile_c_code()

        # Compile the model if requested
        compile_model('social_force_model')

        # Run experiment
        run_experiment(
            config, 
            output_dir, 
            'social_force_model', 
            plot=False, 
            copy_results=True
        )

        # Copy results from output directory to latest directory
        copy_results_to_latest(output_dir)

        print(f"\nExperiment completed. Results saved in {output_dir}")

    def _analyze_iteration_data(self) -> Dict:
        """
        Analyze the iteration data and extract comprehensive statistics.
        
        Returns:
            Dictionary containing analyzed data for plotting
        """
        print("Analyzing iteration data...")
        
        # Get the results files
        latest_dir = self.results_dir / 'latest'
        if not latest_dir.exists():
            print("No results found in latest directory!")
            return {}
            
        results_files = [f for f in latest_dir.iterdir() if f.suffix == '.csv']
        
        if not results_files:
            print("No CSV files found!")
            return {}
        
        # Initialize data structures
        analysis_data = {
            'times': [],
            'groups_per_iteration': [],
            'groups_evolution': {},
            'group_sizes': {},
            'stability_metrics': {},
            'convergence_data': {}
        }
        
        # Process each iteration file
        for i, result_file in enumerate(results_files):
            print(f"Processing iteration {i+1}/{len(results_files)}: {result_file.name}")
            
            try:
                df = pd.read_csv(result_file)
                particles = int((len(df.columns) - 1) / 5)
                
                iteration_groups = []
                iteration_times = []
                group_sizes_per_time = []
                
                # Analyze each time step
                for index, row in df.iterrows():
                    if index % 5 != 0:  # Sample every 5th time step
                        continue
                        
                    time = row['time']
                    iteration_times.append(time)
                    
                    # Calculate groups for this time step
                    groups = Clustering(row, particles).calculate_groups_by_time(
                        row, 
                        from_y=(VOLUMES/ 2) - int(WIDTH / 2),
                        to_y=(VOLUMES/ 2) + int(WIDTH / 2)
                    )
                    
                    num_groups = len(groups)
                    iteration_groups.append(num_groups)
                    
                    # Store group sizes for analysis
                    if time not in analysis_data['group_sizes']:
                        analysis_data['group_sizes'][time] = []
                    analysis_data['group_sizes'][time].extend([len(g) for g in groups])
                    
                    # Store evolution data
                    if time not in analysis_data['groups_evolution']:
                        analysis_data['groups_evolution'][time] = []
                    analysis_data['groups_evolution'][time].append(num_groups)
                
                # Store iteration data
                analysis_data['times'].extend(iteration_times)
                analysis_data['groups_per_iteration'].append({
                    'iteration': i,
                    'times': iteration_times,
                    'groups': iteration_groups,
                    'mean_groups': np.mean(iteration_groups),
                    'std_groups': np.std(iteration_groups),
                    'max_groups': np.max(iteration_groups),
                    'min_groups': np.min(iteration_groups)
                })
                
                # Calculate stability metrics
                if len(iteration_groups) > 1:
                    stability = 1 - (np.std(iteration_groups) / np.mean(iteration_groups))
                    analysis_data['stability_metrics'][i] = stability
                
            except Exception as e:
                print(f"Warning: Could not process {result_file.name}: {e}")
                continue
        
        # Calculate convergence data
        if analysis_data['groups_per_iteration']:
            mean_groups_per_iteration = [data['mean_groups'] for data in analysis_data['groups_per_iteration']]
            analysis_data['convergence_data'] = {
                'iterations': list(range(len(mean_groups_per_iteration))),
                'mean_groups': mean_groups_per_iteration,
                'convergence_rate': self._calculate_convergence_rate(mean_groups_per_iteration)
            }
        
        return analysis_data

    def _calculate_convergence_rate(self, data: List[float]) -> float:
        """Calculate the convergence rate of the lane formation process."""
        if len(data) < 2:
            return 0.0
        
        # Calculate the rate of change
        changes = np.diff(data)
        if len(changes) == 0:
            return 0.0
        
        # Convergence rate is the average absolute change
        return np.mean(np.abs(changes))

    def _create_enhanced_plots(self):
        """Create comprehensive enhanced plots showing lane evolution."""
        analysis_data = self._analyze_iteration_data()
        
        if not analysis_data:
            print("No data available for plotting!")
            return
        
        print("Creating enhanced visualizations...")
        
        # Create main comprehensive figure
        self._create_main_comprehensive_plot(analysis_data)
        
        # Create time evolution plots
        self._create_time_evolution_plots(analysis_data)
        
        # Create statistical analysis plots
        self._create_statistical_analysis_plots(analysis_data)
        
        # Create convergence analysis
        self._create_convergence_analysis(analysis_data)
        
        print(f"All enhanced plots saved to: {self.figures_dir}")

    def _create_main_comprehensive_plot(self, data: Dict):
        """Create the main comprehensive plot showing multiple aspects of lane evolution."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Groups per time (all iterations) - Top left
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_groups_per_time_all_iterations(ax1, data)
        
        # Plot 2: Groups per time (averaged) - Top center
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_groups_per_time_averaged(ax2, data)
        
        # Plot 3: Group size distribution - Top right
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_group_size_distribution(ax3, data)
        
        # Plot 4: Iteration comparison - Middle left
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_iteration_comparison(ax4, data)
        
        # Plot 5: Stability analysis - Middle center
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_stability_analysis(ax5, data)
        
        # Plot 6: Convergence analysis - Middle right
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_convergence_analysis(ax6, data)
        
        # Plot 7: Heatmap of group evolution - Bottom (spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_group_evolution_heatmap(ax7, data)
        
        # Add overall title
        fig.suptitle('Comprehensive Analysis: Lane Formation Evolution Across Iterations', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(self.figures_dir / 'comprehensive_lane_evolution_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Main comprehensive plot saved!")

    def _plot_groups_per_time_all_iterations(self, ax, data: Dict):
        """Plot groups per time for all iterations."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(data['groups_per_iteration'])))
        
        for i, iteration_data in enumerate(data['groups_per_iteration']):
            ax.plot(iteration_data['times'], iteration_data['groups'], 
                   alpha=0.6, linewidth=1, color=colors[i], 
                   label=f'Iteration {i+1}' if i < 5 else None)
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('(a) Groups per Time - All Iterations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(data['groups_per_iteration']) <= 5:
            ax.legend(fontsize=10)

    def _plot_groups_per_time_averaged(self, ax, data: Dict):
        """Plot averaged groups per time with error bars."""
        if not data['groups_evolution']:
            return
            
        times = sorted(data['groups_evolution'].keys())
        means = [np.mean(data['groups_evolution'][t]) for t in times]
        stds = [np.std(data['groups_evolution'][t]) for t in times]
        
        ax.errorbar(times, means, yerr=stds, fmt='o-', 
                   markersize=6, capsize=4, capthick=2, elinewidth=2,
                   color='#2E86AB', alpha=0.8, label='Mean ± Std')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('(b) Groups per Time - Averaged', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_group_size_distribution(self, ax, data: Dict):
        """Plot distribution of group sizes."""
        all_group_sizes = []
        for time_data in data['group_sizes'].values():
            all_group_sizes.extend(time_data)
        
        if all_group_sizes:
            ax.hist(all_group_sizes, bins=20, alpha=0.7, color='#A23B72', 
                   edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(all_group_sizes), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(all_group_sizes):.1f}')
            
        ax.set_xlabel('Group Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('(c) Group Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_iteration_comparison(self, ax, data: Dict):
        """Plot comparison of iterations."""
        iterations = []
        mean_groups = []
        std_groups = []
        
        for iteration_data in data['groups_per_iteration']:
            iterations.append(iteration_data['iteration'])
            mean_groups.append(iteration_data['mean_groups'])
            std_groups.append(iteration_data['std_groups'])
        
        ax.errorbar(iterations, mean_groups, yerr=std_groups, fmt='o-', 
                   markersize=8, capsize=5, capthick=2, elinewidth=2,
                   color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('(d) Iteration Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_stability_analysis(self, ax, data: Dict):
        """Plot stability analysis across iterations."""
        if not data['stability_metrics']:
            return
            
        iterations = list(data['stability_metrics'].keys())
        stabilities = list(data['stability_metrics'].values())
        
        ax.bar(iterations, stabilities, alpha=0.7, color='#C73E1D')
        ax.axhline(np.mean(stabilities), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(stabilities):.3f}')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stability Index', fontsize=12, fontweight='bold')
        ax.set_title('(e) Stability Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_convergence_analysis(self, ax, data: Dict):
        """Plot convergence analysis."""
        if not data['convergence_data']:
            return
            
        conv_data = data['convergence_data']
        ax.plot(conv_data['iterations'], conv_data['mean_groups'], 'o-', 
               markersize=8, linewidth=2, color='#2E86AB')
        
        # Add convergence rate annotation
        ax.text(0.02, 0.98, f'Convergence Rate: {conv_data["convergence_rate"]:.3f}', 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               verticalalignment='top')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Groups', fontsize=12, fontweight='bold')
        ax.set_title('(f) Convergence Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_group_evolution_heatmap(self, ax, data: Dict):
        """Plot heatmap of group evolution over time and iterations."""
        if not data['groups_evolution']:
            return
            
        # Prepare data for heatmap
        times = sorted(data['groups_evolution'].keys())
        iterations = list(range(len(data['groups_per_iteration'])))
        
        # Create matrix for heatmap
        heatmap_data = []
        for iteration in iterations:
            row = []
            for time in times:
                if time in data['groups_evolution'] and len(data['groups_evolution'][time]) > iteration:
                    row.append(data['groups_evolution'][time][iteration])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        if heatmap_data:
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', 
                          interpolation='nearest')
            
            # Set labels
            ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
            ax.set_ylabel('Iterations', fontsize=12, fontweight='bold')
            ax.set_title('(g) Group Evolution Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of Groups', fontsize=12, fontweight='bold')

    def _create_time_evolution_plots(self, data: Dict):
        """Create detailed time evolution plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Detailed time evolution with confidence intervals
        self._plot_detailed_time_evolution(ax1, data)
        
        # Plot 2: Group formation rate
        self._plot_group_formation_rate(ax2, data)
        
        # Plot 3: Lane stability over time
        self._plot_lane_stability_over_time(ax3, data)
        
        # Plot 4: Comparative analysis
        self._plot_comparative_analysis(ax4, data)
        
        plt.suptitle('Detailed Time Evolution Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'time_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Time evolution plots saved!")

    def _plot_detailed_time_evolution(self, ax, data: Dict):
        """Plot detailed time evolution with confidence intervals."""
        if not data['groups_evolution']:
            return
            
        times = sorted(data['groups_evolution'].keys())
        means = [np.mean(data['groups_evolution'][t]) for t in times]
        stds = [np.std(data['groups_evolution'][t]) for t in times]
        
        # Calculate confidence intervals (95%)
        confidence_intervals = [1.96 * std for std in stds]
        
        ax.fill_between(times, 
                       [m - ci for m, ci in zip(means, confidence_intervals)],
                       [m + ci for m, ci in zip(means, confidence_intervals)],
                       alpha=0.3, color='#2E86AB', label='95% Confidence Interval')
        ax.plot(times, means, 'o-', linewidth=2, markersize=6, 
               color='#2E86AB', label='Mean Groups')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('Detailed Time Evolution with Confidence Intervals', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_group_formation_rate(self, ax, data: Dict):
        """Plot the rate of group formation over time."""
        if not data['groups_evolution']:
            return
            
        times = sorted(data['groups_evolution'].keys())
        means = [np.mean(data['groups_evolution'][t]) for t in times]
        
        # Calculate formation rate (derivative)
        if len(means) > 1:
            formation_rate = np.gradient(means, times)
            ax.plot(times[1:], formation_rate[1:], 'o-', linewidth=2, markersize=6, 
                   color='#F18F01', label='Formation Rate')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Group Formation Rate', fontsize=12, fontweight='bold')
        ax.set_title('Group Formation Rate Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_lane_stability_over_time(self, ax, data: Dict):
        """Plot lane stability over time."""
        if not data['groups_evolution']:
            return
            
        times = sorted(data['groups_evolution'].keys())
        stabilities = []
        
        for time in times:
            groups = data['groups_evolution'][time]
            if len(groups) > 1:
                stability = 1 - (np.std(groups) / np.mean(groups))
                stabilities.append(stability)
            else:
                stabilities.append(0)
        
        ax.plot(times, stabilities, 'o-', linewidth=2, markersize=6, 
               color='#A23B72', label='Lane Stability')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stability Index', fontsize=12, fontweight='bold')
        ax.set_title('Lane Stability Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_comparative_analysis(self, ax, data: Dict):
        """Plot comparative analysis between iterations."""
        if not data['groups_per_iteration']:
            return
            
        # Compare first and last iterations
        first_iter = data['groups_per_iteration'][0]
        last_iter = data['groups_per_iteration'][-1]
        
        ax.plot(first_iter['times'], first_iter['groups'], 'o-', 
               linewidth=2, markersize=6, color='#2E86AB', 
               label=f'First Iteration (Mean: {first_iter["mean_groups"]:.2f})')
        ax.plot(last_iter['times'], last_iter['groups'], 's-', 
               linewidth=2, markersize=6, color='#F18F01', 
               label=f'Last Iteration (Mean: {last_iter["mean_groups"]:.2f})')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('First vs Last Iteration Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _create_statistical_analysis_plots(self, data: Dict):
        """Create statistical analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Statistical summary
        self._plot_statistical_summary(ax1, data)
        
        # Plot 2: Distribution analysis
        self._plot_distribution_analysis(ax2, data)
        
        # Plot 3: Correlation analysis
        self._plot_correlation_analysis(ax3, data)
        
        # Plot 4: Performance metrics
        self._plot_performance_metrics(ax4, data)
        
        plt.suptitle('Statistical Analysis of Lane Formation', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Statistical analysis plots saved!")

    def _plot_statistical_summary(self, ax, data: Dict):
        """Plot statistical summary of the data."""
        if not data['groups_per_iteration']:
            return
            
        # Calculate overall statistics
        all_means = [iter_data['mean_groups'] for iter_data in data['groups_per_iteration']]
        all_stds = [iter_data['std_groups'] for iter_data in data['groups_per_iteration']]
        
        # Create box plot
        all_groups_data = []
        labels = []
        for i, iter_data in enumerate(data['groups_per_iteration']):
            all_groups_data.append(iter_data['groups'])
            labels.append(f'Iter {i+1}')
        
        bp = ax.boxplot(all_groups_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_groups_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Iterations', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Summary by Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_distribution_analysis(self, ax, data: Dict):
        """Plot distribution analysis."""
        if not data['groups_per_iteration']:
            return
            
        # Plot distributions for each iteration
        for i, iter_data in enumerate(data['groups_per_iteration']):
            ax.hist(iter_data['groups'], bins=15, alpha=0.6, 
                   label=f'Iteration {i+1}', density=True)
        
        ax.set_xlabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Groups by Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_correlation_analysis(self, ax, data: Dict):
        """Plot correlation analysis."""
        if not data['groups_per_iteration']:
            return
            
        # Calculate correlations between iterations
        iterations = list(range(len(data['groups_per_iteration'])))
        correlations = []
        
        for i in range(len(iterations) - 1):
            iter1_groups = data['groups_per_iteration'][i]['groups']
            iter2_groups = data['groups_per_iteration'][i + 1]['groups']
            
            # Align the data
            min_len = min(len(iter1_groups), len(iter2_groups))
            if min_len > 1:
                corr = np.corrcoef(iter1_groups[:min_len], iter2_groups[:min_len])[0, 1]
                correlations.append(corr)
            else:
                correlations.append(0)
        
        if correlations:
            ax.bar(range(len(correlations)), correlations, alpha=0.7, color='#C73E1D')
            ax.axhline(np.mean(correlations), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
        
        ax.set_xlabel('Iteration Pair', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title('Correlation Between Consecutive Iterations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_performance_metrics(self, ax, data: Dict):
        """Plot performance metrics."""
        if not data['groups_per_iteration']:
            return
            
        iterations = list(range(len(data['groups_per_iteration'])))
        means = [iter_data['mean_groups'] for iter_data in data['groups_per_iteration']]
        maxs = [iter_data['max_groups'] for iter_data in data['groups_per_iteration']]
        mins = [iter_data['min_groups'] for iter_data in data['groups_per_iteration']]
        
        ax.fill_between(iterations, mins, maxs, alpha=0.3, color='#2E86AB', 
                       label='Min-Max Range')
        ax.plot(iterations, means, 'o-', linewidth=2, markersize=8, 
               color='#2E86AB', label='Mean Groups')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics by Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _create_convergence_analysis(self, data: Dict):
        """Create convergence analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Convergence rate analysis
        self._plot_convergence_rate_analysis(ax1, data)
        
        # Plot 2: Stability convergence
        self._plot_stability_convergence(ax2, data)
        
        # Plot 3: Final state analysis
        self._plot_final_state_analysis(ax3, data)
        
        # Plot 4: Convergence summary
        self._plot_convergence_summary(ax4, data)
        
        plt.suptitle('Convergence Analysis of Lane Formation', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Convergence analysis plots saved!")

    def _plot_convergence_rate_analysis(self, ax, data: Dict):
        """Plot convergence rate analysis."""
        if not data['convergence_data']:
            return
            
        conv_data = data['convergence_data']
        
        # Calculate convergence rate over windows
        window_size = 3
        convergence_rates = []
        window_centers = []
        
        for i in range(window_size, len(conv_data['mean_groups'])):
            window_data = conv_data['mean_groups'][i-window_size:i]
            rate = np.mean(np.abs(np.diff(window_data)))
            convergence_rates.append(rate)
            window_centers.append(i - window_size/2)
        
        if convergence_rates:
            ax.plot(window_centers, convergence_rates, 'o-', linewidth=2, markersize=6, 
                   color='#2E86AB', label='Convergence Rate')
            ax.axhline(np.mean(convergence_rates), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(convergence_rates):.3f}')
        
        ax.set_xlabel('Iteration Window Center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Convergence Rate', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Rate Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_stability_convergence(self, ax, data: Dict):
        """Plot stability convergence."""
        if not data['stability_metrics']:
            return
            
        iterations = list(data['stability_metrics'].keys())
        stabilities = list(data['stability_metrics'].values())
        
        # Calculate moving average
        window_size = 3
        moving_avg = []
        for i in range(window_size, len(stabilities)):
            avg = np.mean(stabilities[i-window_size:i])
            moving_avg.append(avg)
        
        if moving_avg:
            ax.plot(iterations[window_size:], moving_avg, 'o-', linewidth=2, markersize=6, 
                   color='#F18F01', label='Moving Average')
            ax.plot(iterations, stabilities, 's-', alpha=0.6, linewidth=1, markersize=4, 
                   color='#A23B72', label='Raw Stability')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stability Index', fontsize=12, fontweight='bold')
        ax.set_title('Stability Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_final_state_analysis(self, ax, data: Dict):
        """Plot final state analysis."""
        if not data['groups_per_iteration']:
            return
            
        # Analyze final states
        final_groups = [iter_data['groups'][-1] if iter_data['groups'] else 0 
                       for iter_data in data['groups_per_iteration']]
        
        ax.hist(final_groups, bins=10, alpha=0.7, color='#C73E1D', 
               edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(final_groups), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(final_groups):.2f}')
        
        ax.set_xlabel('Final Number of Groups', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Final State Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_convergence_summary(self, ax, data: Dict):
        """Plot convergence summary."""
        if not data['convergence_data']:
            return
            
        conv_data = data['convergence_data']
        
        # Create summary statistics
        summary_stats = {
            'Total Iterations': len(conv_data['iterations']),
            'Mean Groups': f"{np.mean(conv_data['mean_groups']):.2f}",
            'Std Groups': f"{np.std(conv_data['mean_groups']):.2f}",
            'Convergence Rate': f"{conv_data['convergence_rate']:.3f}",
            'Final Groups': f"{conv_data['mean_groups'][-1]:.2f}" if conv_data['mean_groups'] else "N/A"
        }
        
        # Create text summary
        summary_text = '\n'.join([f'{k}: {v}' for k, v in summary_stats.items()])
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Convergence Summary', fontsize=14, fontweight='bold')
        ax.axis('off')


def lanes_by_iterations():
    """Main function to run the enhanced lanes by iterations experiment."""
    experiment = LanesByIterationsExperiment()
    experiment.run()


def plot_results():
    """
    Legacy function for backward compatibility.
    Creates the original simple plots plus enhanced versions.
    """
    print("Creating legacy plots for backward compatibility...")
    
    # Create the original simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Number of groups per time(all iterations)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of groups')

    ax2.set_title('Number of groups per time(averaged)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of groups')

    # Get the results files
    results_files = [f for f in os.listdir(os.path.join('experiments/lanes_by_iterations/results', 'latest')) if f.endswith('.csv')]

    # Read the results files
    groups_per_time = {}
    groups_per_time_averaged = {}
    for result_file in results_files:
        df = pd.read_csv(os.path.join('experiments/lanes_by_iterations/results/latest', result_file))
        particles = (len(df.columns) - 1) / 5
        for index, row in df.iterrows():
            if index % 5 != 0:
                continue

            groups = Clustering(row, int(particles)).calculate_groups()
            groups_per_time[row['time']] = len(groups)

            if row['time'] not in groups_per_time_averaged:
                groups_per_time_averaged[row['time']] = [len(groups)]
            else:
                groups_per_time_averaged[row['time']].append(len(groups))

        ax1.plot(groups_per_time.keys(), groups_per_time.values())
    
    mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time_averaged.items()}
    std_groups_per_time = {k: np.std(v) for k, v in groups_per_time_averaged.items()}
    ax2.plot(mean_groups_per_time.keys(), mean_groups_per_time.values(), label='all iterations')
    ax2.fill_between(
        list(mean_groups_per_time.keys()), 
        (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
        (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
        alpha=0.2
    )

    fig.savefig(f'experiments/lanes_by_iterations/groups_by_iterations.png')
    plt.close()
    
    print("Legacy plots saved!")
    
    # Also create enhanced plots
    experiment = LanesByIterationsExperiment()
    experiment._create_enhanced_plots()


if __name__ == '__main__':
    lanes_by_iterations()
