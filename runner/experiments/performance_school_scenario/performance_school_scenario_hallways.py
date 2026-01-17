import json
import os
import subprocess
import numpy as np
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.config_manager import config_manager
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from src.experiments import apply_publication_style

# Load the school configurations
def load_school_configs(config_type):
    """Load the different school configurations from JSON file."""
    if config_type == 'hallways':
        config_path = os.path.join(os.path.dirname(__file__), 'pedestrian_school_hallways.json')
    elif config_type == 'square':
        config_path = os.path.join(os.path.dirname(__file__), 'pedestrian_school_square.json')
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

# School scenario parameters
CLASSROOM_EDGE = 7.0  # meters per classroom edge, 49m²
PEDESTRIANS_PER_CLASSROOM = 25

# Implementation configurations to test
IMPLEMENTATION_CONFIGS = {
    'mmoc_border4': {
        'PEDESTRIAN_IMPLEMENTATION': 0,
        'BORDER_IMPLEMENTATION': 4,
        'name': 'RETQSS'
    },
    'retqss_border3': {
        'PEDESTRIAN_IMPLEMENTATION': 1,
        'BORDER_IMPLEMENTATION': 3,
        'name': 'RETQSS Optimizado'
    }
}

# School configuration types
SCHOOL_CONFIG_TYPES = {
    'hallways': 'Pasillos',
    'square': 'Patio central'
}

def get_simulated_time():
    """Get simulated time from config.json."""
    config = load_config('experiments/performance_school_scenario/config.json')
    return config['parameters']['FORCE_TERMINATION_AT']['value']

def calculate_scenario_parameters(grid_divisions, school_config):
    """
    Calculate scenario parameters based on grid divisions and school configuration.
    
    Args:
        grid_divisions: The GRID_DIVISIONS value (3, 7, 11, or 15)
        school_config: The configuration for this grid size
    
    Returns:
        dict: Calculated parameters including grid_size, classroom_count, pedestrian_count
    """
    # Calculate grid size
    grid_size = grid_divisions * CLASSROOM_EDGE
    
    # Count classrooms from the CLASSROOMS matrix
    classroom_matrix = np.array(school_config['CLASSROOMS'])
    classroom_count = np.sum(classroom_matrix)
    
    # Calculate number of pedestrians
    pedestrian_count = int(classroom_count * PEDESTRIANS_PER_CLASSROOM)
    
    return {
        'grid_divisions': grid_divisions,
        'grid_size': grid_size,
        'classroom_count': classroom_count,
        'pedestrian_count': pedestrian_count,
        'school_config': school_config
    }

def run_school_scenario_experiment():
    """
    Run performance experiments for different school scenario sizes.
    Tests two implementation configurations for each grid size and both school types.
    """
    print("Ejecutando experimentos de rendimiento para escenarios escolares...")
    
    all_results = []
    
    # Update configuration from command line arguments
    config_manager.update_from_dict({
        'skip_metrics': True
    })
    
    # Load base config from local experiment directory
    config = load_config('./experiments/performance_school_scenario/config.json')
    
    # Test both school configuration types
    for config_type, config_name_spanish in SCHOOL_CONFIG_TYPES.items()[:-1]:
        print(f"\n{'='*60}")
        print(f"Probando configuración escolar: {config_name_spanish}")
        print(f"{'='*60}")
        
        # Load school configurations for this type
        school_configs = load_school_configs(config_type)
        grid_divisions_list = [int(k) for k in school_configs.keys()]
        grid_divisions_list.sort()
        
        print(f"Probando {len(grid_divisions_list)} tamaños de grilla diferentes: {grid_divisions_list}")
        print(f"Probando {len(IMPLEMENTATION_CONFIGS)} configuraciones de implementación")
        
        # Calculate total experiments for this config type
        total_experiments = len(grid_divisions_list) * len(IMPLEMENTATION_CONFIGS)
        print(f"Total de experimentos para {config_name_spanish}: {total_experiments}")
        
        experiment_count = 0
        
        # Test each grid size with each implementation configuration
        for grid_divisions in grid_divisions_list:
            school_config = school_configs[str(grid_divisions)]
            scenario_params = calculate_scenario_parameters(grid_divisions, school_config)
            scenario_params['config_type'] = config_type
            scenario_params['config_name_spanish'] = config_name_spanish
            
            print(f"\nProbando tamaño de grilla {grid_divisions}x{grid_divisions} (Tamaño de Grilla: {scenario_params['grid_size']}m)")
            print(f"  Aulas: {scenario_params['classroom_count']}")
            print(f"  Peatones: {scenario_params['pedestrian_count']}")
            
            for impl_name, impl_config in IMPLEMENTATION_CONFIGS.items():
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] Ejecutando {impl_config['name']}...")
                
                result = run_single_experiment(
                    config, 
                    scenario_params, 
                    impl_config, 
                    impl_name,
                    config_type
                )
                all_results.append(result)
                print(f"  Completado")
    
    # Generate performance comparison plots
    print("\nGenerando gráficos de comparación de rendimiento...")
    plot_performance_comparison_separate(all_results)
    
    # Save results to CSV
    save_results_to_csv(all_results)
    
    print("\n" + "="*60)
    print("¡Todos los experimentos de escenarios escolares completados exitosamente!")
    print("Resultados guardados en CSV y gráficos generados.")

def run_single_experiment(base_config, scenario_params, impl_config, config_name, config_type):
    """
    Run a single experiment with specified parameters and return timing results.
    """
    # Create descriptive experiment name
    exp_name = f'{config_type}_grid_{scenario_params["grid_divisions"]}x{scenario_params["grid_divisions"]}_{config_name}'
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/performance_school_scenario/results', 
        exp_name
    )
    
    # Update config parameters
    config = base_config.copy()
    config['iterations'] = 1  # Reduced iterations for faster testing
    config['parameters']['N']['value'] = scenario_params['pedestrian_count']
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = impl_config['PEDESTRIAN_IMPLEMENTATION']
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = impl_config['BORDER_IMPLEMENTATION']
    config['parameters']['GRID_SIZE']['value'] = scenario_params['grid_size']
    
    # Set up school scenario maps (CLASSROOMS, HALLWAYS, OBSTACLES)
    setup_school_scenario_maps(config, scenario_params['school_config'])
    
    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Update model file parameters
    model_path = '/home/martin/Documents/UBA/Tesis/retqss/model/helbing_school_hallway.mo'
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(scenario_params['pedestrian_count']) + '/', model_path])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(scenario_params['grid_divisions']) + '/', model_path])
    
    # # Compile the C++ code and model
    compile_c_code()
    compile_model('helbing_school_hallway')
    
    # Run experiment
    run_experiment(
        config, 
        output_dir, 
        'helbing_school_hallway', 
        plot=False, 
        copy_results=True
    )
    
    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)
    
    # Read timing results from metrics.csv
    metrics_file = os.path.join( 
        'experiments/performance_school_scenario/results', 
        exp_name, 
        'latest',
        'metrics.csv'
    )
    
    # Read detailed metrics if available
    detailed_metrics = None
    if os.path.exists(metrics_file):
        try:
            metrics_df = pd.read_csv(metrics_file)
            if not metrics_df.empty and 'time' in metrics_df.columns:
                detailed_metrics = {
                    'total_iterations': len(metrics_df),
                    'avg_iteration_time': metrics_df['time'].mean(),
                    'min_iteration_time': metrics_df['time'].min(),
                    'max_iteration_time': metrics_df['time'].max(),
                    'std_iteration_time': metrics_df['time'].std(),
                    'avg_memory_usage': metrics_df['memory_usage'].mean() if 'memory_usage' in metrics_df.columns else None,
                    'std_memory_usage': metrics_df['memory_usage'].std() if 'memory_usage' in metrics_df.columns else None,
                }
        except (pd.errors.EmptyDataError, KeyError):
            detailed_metrics = None
    
    return {
        'grid_divisions': scenario_params['grid_divisions'],
        'grid_size': scenario_params['grid_size'],
        'classroom_count': scenario_params['classroom_count'],
        'pedestrian_count': scenario_params['pedestrian_count'],
        'implementation': impl_config['name'],
        'config_name': config_name,
        'config_type': config_type,
        'config_name_spanish': scenario_params['config_name_spanish'],
        'output_dir': output_dir,
        'detailed_metrics': detailed_metrics
    }

def setup_school_scenario_maps(config, school_config):
    """
    Set up CLASSROOMS, HALLWAYS, and OBSTACLES maps for the school scenario.
    """
    # Set the map parameters directly from the school configuration
    config['parameters']['CLASSROOMS']['map'] = school_config['CLASSROOMS']
    config['parameters']['HALLWAYS']['map'] = school_config['HALLWAYS']
    config['parameters']['OBSTACLES']['map'] = school_config['OBSTACLES']

def plot_performance_comparison_separate(results):
    """
    Generate separate performance comparison plots for each school scenario.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_school_scenario/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate separate plots for each configuration type
    for config_type, config_name_spanish in SCHOOL_CONFIG_TYPES.items():
        print(f"Generando gráficos para: {config_name_spanish}")
        
        # Filter data for this configuration type
        config_data = df[df['config_type'] == config_type]
        
        if config_data.empty:
            print(f"  No hay datos para {config_name_spanish}")
            continue
        
        # Generate plot for this configuration
        generate_single_scenario_plot(config_data, config_type, config_name_spanish, results_dir)
    
    print("Gráficos de comparación de rendimiento generados exitosamente!")

def generate_single_scenario_plot(config_data, config_type, config_name_spanish, results_dir):
    """
    Generate a single comprehensive plot for one school scenario.
    """
    apply_publication_style()

    simulated_time = get_simulated_time()
    implementations = list(config_data['implementation'].unique())
    classroom_counts = sorted(config_data['classroom_count'].unique())
    x_pos = np.arange(len(classroom_counts))
    bar_width = 0.7 / max(1, len(implementations))

    style_map = {
        'RETQSS': {'color': '#4ECDC4', 'edgecolor': '#00B894', 'marker': 's'},
        'RETQSS Optimizado': {'color': '#6C5CE7', 'edgecolor': '#5A4FCF', 'marker': '^'},
    }

    # Create performance comparison plot (2x1 grid - only 2 graphs)
    fig, axes = plt.subplots(2, 1, figsize=(24, 16))
    fig.suptitle(f'Escenario Escuela - {config_name_spanish}', fontsize=30, fontweight='bold', y=0.98)

    rtf_by_impl = {}
    memory_by_impl = {}

    for impl_name in implementations:
        impl_data = config_data[config_data['implementation'] == impl_name].sort_values('classroom_count')
        impl_by_classroom = {row['classroom_count']: row for _, row in impl_data.iterrows()}

        rtf_means = []
        rtf_stds = []
        time_means = []
        mem_means = []
        mem_stds = []

        for classroom_count in classroom_counts:
            row = impl_by_classroom.get(classroom_count)
            metrics = row['detailed_metrics'] if row is not None else None
            avg_time = metrics['avg_iteration_time'] if metrics else None
            std_time = metrics['std_iteration_time'] if metrics else None

            if avg_time is not None and not pd.isna(avg_time) and avg_time > 0:
                rtf_mean = simulated_time / avg_time
                rtf_std = simulated_time * std_time / (avg_time ** 2) if std_time is not None and not pd.isna(std_time) else 0
            else:
                rtf_mean = 0
                rtf_std = 0
                avg_time = 0

            rtf_means.append(rtf_mean)
            rtf_stds.append(rtf_std)
            time_means.append(avg_time)

            mem_mean = metrics['avg_memory_usage'] if metrics else None
            mem_std = metrics['std_memory_usage'] if metrics else None
            mem_means.append(mem_mean if mem_mean is not None and not pd.isna(mem_mean) else None)
            mem_stds.append(mem_std if mem_std is not None and not pd.isna(mem_std) else None)

        rtf_by_impl[impl_name] = (rtf_means, rtf_stds, time_means)
        memory_by_impl[impl_name] = (mem_means, mem_stds)

    all_rtf = [value for rtf_means, _, _ in rtf_by_impl.values() for value in rtf_means]
    all_rtf_std = [value for _, rtf_stds, _ in rtf_by_impl.values() for value in rtf_stds]
    max_rtf = max(all_rtf) if all_rtf else 1
    max_rtf_std = max(all_rtf_std) if all_rtf_std else 0
    min_rtf = min(all_rtf) if all_rtf else 0

    # Plot 1: RTF vs Number of Classrooms
    ax1 = axes[0]

    for i, impl_name in enumerate(implementations):
        rtf_means, rtf_stds, time_means = rtf_by_impl[impl_name]
        style = style_map.get(impl_name, {})
        bar_positions = x_pos + (i - (len(implementations) - 1) / 2) * bar_width
        bars = ax1.bar(
            bar_positions,
            rtf_means,
            yerr=rtf_stds,
            color=style.get('color', 'skyblue'),
            edgecolor=style.get('edgecolor', '#333333'),
            width=bar_width,
            capsize=4,
            alpha=0.85,
            label=impl_name,
        )

        label_offset = max_rtf * 0.06 if max_rtf > 0 else 0.06
        for bar, mean, std, time_mean in zip(bars, rtf_means, rtf_stds, time_means):
            if mean + std > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + std + label_offset,
                    f'{mean:.2f}x±{std:.2f}\n({time_mean:.1f}s)',
                    ha='center',
                    va='bottom',
                    fontsize=14,
                )

    ax1.set_xlabel('Cantidad de aulas')
    ax1.set_ylabel('RTF - tiempo simulado(s) / tiempo real(s)')
    ax1.set_title('Rendimiento vs cantidad de aulas')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classroom_counts, rotation=45, ha='right')

    y_min = max(0, min_rtf - max_rtf_std * 1.5)
    y_max = max_rtf * 1.6 if max_rtf > 0 else 1
    ax1.set_ylim(y_min, y_max)

    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color='red', linestyle='--', label='Tiempo Real (1.0x)'),
        Line2D([], [], linestyle='None', label=f'Tiempo simulado: {simulated_time:.0f}s'),
    ])
    labels.extend(['Tiempo Real (1.0x)', f'Tiempo simulado: {simulated_time:.0f}s'])
    ax1.legend(handles=handles, labels=labels, loc='upper left')
    ax1.grid(True)

    # Plot 2: Memory usage comparison (if available)
    ax2 = axes[1]

    for i, impl_name in enumerate(implementations):
        memory_means, memory_stds = memory_by_impl[impl_name]
        if all(value is None for value in memory_means):
            continue

        y_values = [value if value is not None else np.nan for value in memory_means]
        y_errors = [value if value is not None else 0 for value in memory_stds]
        style = style_map.get(impl_name, {})

        ax2.errorbar(
            classroom_counts,
            y_values,
            yerr=y_errors if any(y_errors) else None,
            fmt=f"{style.get('marker', 'o')}-",
            label=impl_name,
            linewidth=4,
            markersize=10,
            color=style.get('color', '#4ECDC4'),
            capsize=6,
            capthick=3,
            elinewidth=3,
            alpha=0.8,
            markeredgecolor=style.get('edgecolor', '#333333'),
            markeredgewidth=2,
        )

    ax2.set_xlabel('Cantidad de aulas')
    ax2.set_ylabel('Uso Promedio de Memoria (MB)')
    ax2.set_title('Uso de Memoria vs cantidad de aulas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add explanatory text about students per classroom
    fig.text(0.5, 0.02, f'Nota: La cantidad de estudiantes es igual a {PEDESTRIANS_PER_CLASSROOM} por aula', 
             ha='center', va='bottom', fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the explanatory text
    
    # Save the plot
    filename = f'comparacion_rendimiento_{config_type}.png'
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico guardado: {filename}")

def plot_performance_comparison(results):
    """
    Generate performance comparison plots for the school scenario experiments.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_school_scenario/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create comprehensive performance comparison plot
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Execution time vs Grid Size (separated by config type)
    plt.subplot(2, 3, 1)
    
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, config_type in enumerate(df['config_type'].unique()):
        config_data = df[df['config_type'] == config_type]
        for j, impl_name in enumerate(config_data['implementation'].unique()):
            impl_data = config_data[config_data['implementation'] == impl_name].sort_values('grid_size')
            if not impl_data.empty:
                times = impl_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
                stds = impl_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
                label = f"{impl_name} ({impl_data['config_name_spanish'].iloc[0]})"
                plt.errorbar(impl_data['grid_size'], times, yerr=stds,
                            fmt=f'{markers[j]}-', label=label, linewidth=2, markersize=6, 
                            capsize=3, color=colors[i], alpha=0.8)
    
    plt.xlabel('Tamaño del escenario (metros)')
    plt.ylabel('Tiempo Promedio de Ejecución (s)')
    plt.title('Rendimiento vs Tamaño del escenario: Escenarios Escolares')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Execution time vs Number of Pedestrians
    plt.subplot(2, 3, 2)
    
    for i, config_type in enumerate(df['config_type'].unique()):
        config_data = df[df['config_type'] == config_type]
        for j, impl_name in enumerate(config_data['implementation'].unique()):
            impl_data = config_data[config_data['implementation'] == impl_name].sort_values('pedestrian_count')
            if not impl_data.empty:
                times = impl_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
                stds = impl_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
                label = f"{impl_name} ({impl_data['config_name_spanish'].iloc[0]})"
                plt.errorbar(impl_data['pedestrian_count'], times, yerr=stds,
                            fmt=f'{markers[j]}-', label=label, linewidth=2, markersize=6, 
                            capsize=3, color=colors[i], alpha=0.8)
    
    plt.xlabel('Número de Peatones')
    plt.ylabel('Tiempo Promedio de Ejecución (s)')
    plt.title('Rendimiento vs Número de Peatones: Escenarios Escolares')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Speedup comparison for each config type
    plt.subplot(2, 3, 3)
    
    for i, config_type in enumerate(df['config_type'].unique()):
        config_data = df[df['config_type'] == config_type]
        mmoc_data = config_data[config_data['implementation'] == 'RETQSS'].sort_values('grid_size')
        retqss_data = config_data[config_data['implementation'] == 'RETQSS Optimizado'].sort_values('grid_size')
        
        if not mmoc_data.empty and not retqss_data.empty:
            # Merge data to calculate speedup
            merged_data = pd.merge(mmoc_data, retqss_data, on='grid_size', suffixes=('_mmoc', '_retqss'))
            mmoc_times = merged_data['detailed_metrics_mmoc'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_times = merged_data['detailed_metrics_retqss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_data['speedup'] = mmoc_times / retqss_times
            
            config_name = merged_data['config_name_spanish_mmoc'].iloc[0]
            plt.plot(merged_data['grid_size'], merged_data['speedup'], 
                    'o-', color=colors[i], linewidth=2, markersize=6, 
                    label=f'Mejora de Rendimiento ({config_name})', alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Rendimiento Igual')
    plt.xlabel('Tamaño del escenario (metros)')
    plt.ylabel('Mejora de Rendimiento (Tiempo RETQSS / Tiempo RETQSS Optimizado)')
    plt.title('Mejora de Rendimiento: RETQSS vs RETQSS Optimizado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Memory usage comparison (if available)
    plt.subplot(2, 3, 4)
    
    for i, config_type in enumerate(df['config_type'].unique()):
        config_data = df[df['config_type'] == config_type]
        for j, impl_name in enumerate(config_data['implementation'].unique()):
            impl_data = config_data[config_data['implementation'] == impl_name].sort_values('grid_size')
            if not impl_data.empty:
                memory_usage = impl_data['detailed_metrics'].apply(lambda x: x['avg_memory_usage'] if x and x['avg_memory_usage'] else None)
                if not memory_usage.isna().all():
                    label = f"{impl_name} ({impl_data['config_name_spanish'].iloc[0]})"
                    plt.plot(impl_data['grid_size'], memory_usage,
                            f'{markers[j]}-', label=label, linewidth=2, markersize=6, 
                            color=colors[i], alpha=0.8)
    
    plt.xlabel('Tamaño del escenario (metros)')
    plt.ylabel('Uso Promedio de Memoria (MB)')
    plt.title('Uso de Memoria vs Tamaño de Grilla: Escenarios Escolares')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Comparison by config type (average performance)
    plt.subplot(2, 3, 5)
    
    # Calculate average performance for each config type and implementation
    config_summary = []
    for config_type in df['config_type'].unique():
        for impl_name in df['implementation'].unique():
            subset = df[(df['config_type'] == config_type) & (df['implementation'] == impl_name)]
            if not subset.empty:
                avg_time = subset['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0).mean()
                config_summary.append({
                    'config_type': config_type,
                    'implementation': impl_name,
                    'avg_time': avg_time
                })
    
    config_summary_df = pd.DataFrame(config_summary)
    
    if not config_summary_df.empty:
        x_pos = np.arange(len(config_summary_df['config_type'].unique()))
        width = 0.35
        
        for i, impl_name in enumerate(config_summary_df['implementation'].unique()):
            impl_data = config_summary_df[config_summary_df['implementation'] == impl_name]
            times = impl_data['avg_time'].values
            plt.bar(x_pos + i*width, times, width, label=impl_name, alpha=0.8)
        
        plt.xlabel('Tipo de Configuración Escolar')
        plt.ylabel('Tiempo Promedio de Ejecución (s)')
        plt.title('Comparación de Rendimiento por Tipo de Configuración')
        plt.xticks(x_pos + width/2, [SCHOOL_CONFIG_TYPES[ct] for ct in config_summary_df['config_type'].unique()])
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Grid size vs Performance (alternative view)
    plt.subplot(2, 3, 6)
    
    for i, config_type in enumerate(df['config_type'].unique()):
        config_data = df[df['config_type'] == config_type]
        for j, impl_name in enumerate(config_data['implementation'].unique()):
            impl_data = config_data[config_data['implementation'] == impl_name].sort_values('grid_size')
            if not impl_data.empty:
                times = impl_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
                stds = impl_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
                label = f"{impl_name} ({impl_data['config_name_spanish'].iloc[0]})"
                plt.errorbar(impl_data['grid_size'], times, yerr=stds,
                            fmt=f'{markers[j]}-', label=label, linewidth=2, markersize=6, 
                            capsize=3, color=colors[i], alpha=0.8)
    
    plt.xlabel('Tamaño del escenario (metros)')
    plt.ylabel('Tiempo Promedio de Ejecución (s)')
    plt.title('Rendimiento vs Tamaño del escenario: Escenarios Escolares')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparacion_rendimiento_escenarios_escolares.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(results):
    """
    Save experiment results to CSV file for further analysis.
    """
    df = pd.DataFrame(results)
    
    # Flatten detailed_metrics into separate columns
    metrics_data = []
    for _, row in df.iterrows():
        base_data = {k: v for k, v in row.items() if k != 'detailed_metrics'}
        if row['detailed_metrics']:
            base_data.update(row['detailed_metrics'])
        else:
            base_data.update({
                'total_iterations': None,
                'avg_iteration_time': None,
                'min_iteration_time': None,
                'max_iteration_time': None,
                'std_iteration_time': None,
                'avg_memory_usage': None,
                'std_memory_usage': None
            })
        metrics_data.append(base_data)
    
    results_df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    results_dir = 'experiments/performance_school_scenario/results'
    csv_path = os.path.join(results_dir, 'resultados_rendimiento_escenarios_escolares.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Resultados guardados en: {csv_path}")

def generate_flowgraph_visualizations():
    """
    Generate static FlowGraph visualizations for both school scenarios.
    """
    print("\nGenerando visualizaciones FlowGraph para escenarios escolares...")
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_school_scenario/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate visualization for each school configuration type
    for config_type, config_name_spanish in SCHOOL_CONFIG_TYPES.items():
        print(f"Generando visualización para: {config_name_spanish}")
        
        # Load school configurations for this type
        school_configs = load_school_configs(config_type)
        
        # Use the largest grid size for visualization (most detailed)
        grid_divisions = max([int(k) for k in school_configs.keys()])
        school_config = school_configs[str(grid_divisions)]
        
        # Calculate scenario parameters
        scenario_params = calculate_scenario_parameters(grid_divisions, school_config)
        
        # Generate the visualization
        generate_single_flowgraph(
            config_type, 
            config_name_spanish, 
            scenario_params, 
            school_config,
            results_dir
        )
    
    print("Visualizaciones FlowGraph generadas exitosamente!")

def generate_single_flowgraph(config_type, config_name_spanish, scenario_params, school_config, results_dir):
    """
    Generate a single FlowGraph visualization for a school scenario.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Get grid parameters
    grid_divisions = scenario_params['grid_divisions']
    grid_size = scenario_params['grid_size']
    cell_size = grid_size / grid_divisions
    
    # Set up the plot area
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')

    # Disable grid
    ax.grid(False)
    

    # Define colors for different cell types
    cell_colors = {
        'obstacle': 'black',
        'hallway': 'lightgray', 
        'classroom': 'lightblue'
    }
    
    # Convert matrices to numpy arrays for easier processing
    obstacles = np.array(school_config['OBSTACLES'])
    hallways = np.array(school_config['HALLWAYS'])
    classrooms = np.array(school_config['CLASSROOMS'])

    # Color each cell based on its type
    for i in range(grid_divisions):
        for j in range(grid_divisions):
            x = j * cell_size
            y = (grid_divisions - 1 - i) * cell_size  # Flip Y axis to match matrix indexing
            
            # Determine cell type (priority: obstacle > classroom > hallway)
            if obstacles[i, j] == 1:
                cell_type = 'obstacle'
                alpha = 0.8
            elif classrooms[i, j] == 1:
                cell_type = 'classroom'
                alpha = 0.6
            elif hallways[i, j] == 1:
                cell_type = 'hallway'
                alpha = 0.4
            else:
                cell_type = 'hallway' # Default to hallway because i can't fix the issue with the empty cells
                alpha = 0.4
            
            if cell_type != 'empty':
                color = cell_colors[cell_type]
                rect = Rectangle(
                    (x, y), 
                    cell_size, 
                    cell_size, 
                    facecolor=color, 
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(rect)
    
    # Add main title to the figure
    fig.suptitle(f'Escenario Escuela - {config_name_spanish}', fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtitle with details
    ax.set_title(f'Tamaño de Grilla: {grid_divisions}x{grid_divisions} ({grid_size}m x {grid_size}m) | '
                f'Aulas: {scenario_params["classroom_count"]} | '
                f'Peatones: {scenario_params["pedestrian_count"]}', 
                fontsize=12, pad=15)
    
    ax.set_xlabel('Posición X (metros)', fontsize=12)
    ax.set_ylabel('Posición Y (metros)', fontsize=12)
    
    # Create legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=cell_colors['obstacle'], alpha=0.8, label='Obstáculos'),
        Rectangle((0, 0), 1, 1, facecolor=cell_colors['classroom'], alpha=0.6, label='Aulas'),
        Rectangle((0, 0), 1, 1, facecolor=cell_colors['hallway'], alpha=0.4, label='Pasillos')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
              title='Leyenda', fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add grid information text
    info_text = f'División de Grilla: {grid_divisions}x{grid_divisions}\n'
    info_text += f'Tamaño de Celda: {cell_size:.1f}m x {cell_size:.1f}m\n'
    info_text += f'Área Total: {grid_size:.1f}m x {grid_size:.1f}m'
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Save the visualization
    filename = f'flowgraph_escenario_{config_type}.png'
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Visualización guardada: {filename}")

def performance_school_scenario_hallways():
    # Generate FlowGraph visualizations first
    generate_flowgraph_visualizations()
    
    # Then run the performance experiments
    run_school_scenario_experiment()
