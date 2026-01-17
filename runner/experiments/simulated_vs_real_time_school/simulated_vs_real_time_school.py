import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import load_config, create_output_dir
from src.config_manager import config_manager
from src.experiments import (
    ConfigBuilder, ModelUpdater, ExperimentRunner,
    apply_publication_style, DEFAULT_COLORS
)

# Import experiment configuration
from experiments.simulated_vs_real_time_school.experiment_config import (
    CLASSROOM_EDGE, PEDESTRIANS_PER_CLASSROOM, GRID_DIVISIONS, GRID_SIZE,
    CLASSROOM_COUNT, PEDESTRIAN_COUNT, SIMULATION_DURATIONS, IMPLEMENTATIONS,
    RUN_EXPERIMENT, MODEL_NAME, BORDER_IMPLEMENTATION_UNOPTIMIZED,
    BORDER_IMPLEMENTATION_OPTIMIZED, VOLUME_NEIGHBORHOOD_TYPE, NUM_ITERATIONS,
    MAIN_FIGURE_SIZE, SECONDARY_FIGURE_SIZE, SHOW_VALUE_LABELS,
    validate_configuration
)

warnings.filterwarnings('ignore')
apply_publication_style()


def simulated_vs_real_time_school():
    """
    Run experiments comparing simulated time vs real time for the school scenario.
    
    Compares:
    - Model without optimizations (PEDESTRIAN_IMPLEMENTATION=1)
    - Model with optimizations (PEDESTRIAN_IMPLEMENTATION=2)
    
    For various simulation durations from seconds to tens of minutes.
    """
    # Validate configuration first
    if not validate_configuration():
        print("\n✗ Configuración inválida. Por favor corrige los errores y vuelve a intentar.")
        return
    
    print("="*80)
    print("EXPERIMENTO: Tiempo Simulado vs Tiempo Real en Escenario Escuela")
    print("="*80)
    print(f"\nConfiguración del escenario:")
    print(f"  - Escenario: Escuela (pasillos)")
    print(f"  - Tamaño de grilla: {GRID_DIVISIONS}x{GRID_DIVISIONS} ({GRID_SIZE}m x {GRID_SIZE}m)")
    print(f"  - Aulas: {CLASSROOM_COUNT}")
    print(f"  - Peatones: {PEDESTRIAN_COUNT}")
    print(f"  - Duraciones a simular: {SIMULATION_DURATIONS} segundos")
    print(f"\nModelos a comparar:")
    for impl_id, impl_info in IMPLEMENTATIONS.items():
        print(f"  - Implementación {impl_id}: {impl_info['name']} ({impl_info['description']})")
    
    if RUN_EXPERIMENT:
        total_experiments = len(SIMULATION_DURATIONS) * len(IMPLEMENTATIONS)
        print(f"\nTotal de experimentos: {total_experiments}")
        print("="*80)
        
        results = []
        current = 0

        config_manager.update_from_dict({
            'skip_metrics': True
        })
        
        for duration in SIMULATION_DURATIONS:
            for impl_id, impl_info in IMPLEMENTATIONS.items():
                current += 1
                print(f"\n{'='*80}")
                print(f"[{current}/{total_experiments}] Ejecutando experimento:")
                print(f"  Duración simulada: {duration}s ({duration/60:.1f} minutos)")
                print(f"  Implementación: {impl_info['name']}")
                print(f"{'='*80}")
                
                result = run_single_experiment(duration, impl_id, impl_info)
                results.append(result)
                
                print(f"\nResultado:")
                print(f"  Iteraciones: {result['num_iterations']}")
                print(f"  Tiempo real promedio: {result['real_time_mean']:.2f}s (±{result['real_time_std']:.2f}s)")
                print(f"  Factor de tiempo real promedio: {result['real_time_factor_mean']:.2f}x (±{result['real_time_factor_std']:.2f}x)")
                if result['real_time_factor_mean'] > 1:
                    print(f"  → Simulación más rápida que tiempo real")
                elif result['real_time_factor_mean'] < 1:
                    print(f"  → Simulación más lenta que tiempo real")
                else:
                    print(f"  → Simulación en tiempo real")
        
        # Save results to CSV
        save_results(results)
        
        # Generate comparison plot
        print("\n" + "="*80)
        print("Generando visualización de resultados...")
        print("="*80)
        plot_results(results)
    
    print("\n" + "="*80)
    print("¡Experimento completado exitosamente!")
    print("="*80)


def run_single_experiment(duration, implementation_id, implementation_info):
    """
    Execute a single experiment with specified duration and implementation.
    
    Args:
        duration: Simulation duration in seconds
        implementation_id: Implementation ID (1 or 2)
        implementation_info: Dictionary with implementation details
        
    Returns:
        Dictionary with experiment results including real-time factor
    """
    config = load_config('./experiments/simulated_vs_real_time_school/config.json')
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/simulated_vs_real_time_school/results',
        f'duration_{duration}s_impl_{implementation_id}'
    )
    print(f"Directorio de salida creado: {output_dir}")
    
    # Select appropriate border implementation based on model implementation
    border_implementation = (BORDER_IMPLEMENTATION_UNOPTIMIZED if implementation_id == 1 
                            else BORDER_IMPLEMENTATION_OPTIMIZED)
    
    # Configure experiment parameters
    ConfigBuilder(config) \
        .set_iterations(NUM_ITERATIONS) \
        .set_pedestrian_count(PEDESTRIAN_COUNT) \
        .set_pedestrian_implementation(implementation_id) \
        .set_border_implementation(border_implementation) \
        .set_volume_neighborhood_type(VOLUME_NEIGHBORHOOD_TYPE) \
        .set_grid_size(GRID_SIZE) \
        .set_parameter('FORCE_TERMINATION_AT', duration)
    
    # Save configuration
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Update model parameters
    model = ModelUpdater('../retqss/model/helbing_school_hallway.mo')
    model.update_parameters({
        'GRID_DIVISIONS': GRID_DIVISIONS,
        'N': PEDESTRIAN_COUNT
    })
    
    # Run the experiment
    print("Iniciando simulación...")
    ExperimentRunner.run_standard_experiment(
        config, 
        output_dir, 
        MODEL_NAME,
        copy_results=False
    )
    
    # Read metrics from the generated CSV file
    metrics_file = os.path.join(output_dir, 'metrics.csv')
    
    if not os.path.exists(metrics_file):
        print(f"⚠️ Advertencia: No se encontró el archivo de métricas en {metrics_file}")
        return {
            'simulated_duration': duration,
            'implementation_id': implementation_id,
            'implementation_name': implementation_info['name'],
            'real_time_mean': 0,
            'real_time_std': 0,
            'real_time_factor_mean': 0,
            'real_time_factor_std': 0,
            'num_iterations': 0,
            'output_dir': output_dir
        }
    
    # Load metrics
    df_metrics = pd.read_csv(metrics_file)
    
    # Extract real execution times (in seconds)
    # The 'time' column contains the real execution time for each iteration
    real_times = df_metrics['time'].values
    
    # Calculate statistics
    real_time_mean = np.mean(real_times)
    real_time_std = np.std(real_times)
    
    # Calculate real-time factor for each iteration
    # Factor > 1: simulation is faster than real time
    # Factor < 1: simulation is slower than real time
    # Factor = 1: simulation runs at real time
    real_time_factors = duration / real_times
    real_time_factor_mean = np.mean(real_time_factors)
    real_time_factor_std = np.std(real_time_factors)
    
    print(f"Simulación completada:")
    print(f"  Tiempo real promedio: {real_time_mean:.2f}s (±{real_time_std:.2f}s)")
    print(f"  Factor de tiempo real promedio: {real_time_factor_mean:.2f}x (±{real_time_factor_std:.2f}x)")
    
    return {
        'simulated_duration': duration,
        'implementation_id': implementation_id,
        'implementation_name': implementation_info['name'],
        'real_time_mean': real_time_mean,
        'real_time_std': real_time_std,
        'real_time_factor_mean': real_time_factor_mean,
        'real_time_factor_std': real_time_factor_std,
        'num_iterations': len(real_times),
        'output_dir': output_dir
    }


def save_results(results):
    """Save experiment results to CSV file."""
    df = pd.DataFrame(results)
    
    output_path = 'experiments/simulated_vs_real_time_school/results_summary.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"{'Duración (s)':<15} {'Implementación':<25} {'Tiempo Real (s)':<25} {'Factor':<20}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['simulated_duration']:<15} {row['implementation_name']:<25} "
              f"{row['real_time_mean']:.2f}±{row['real_time_std']:.2f}s{'':<11} "
              f"{row['real_time_factor_mean']:.2f}±{row['real_time_factor_std']:.2f}x")
    
    print("="*90)


def plot_results(results):
    """
    Generate comparative line plot of simulated time vs real time factor.
    
    Creates a plot showing how the real-time factor evolves with increasing
    simulation duration for both model implementations.
    """
    df = pd.DataFrame(results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=MAIN_FIGURE_SIZE)
    
    # Plot each implementation
    for impl_id, impl_info in IMPLEMENTATIONS.items():
        impl_data = df[df['implementation_id'] == impl_id].sort_values('simulated_duration')
        
        if not impl_data.empty:
            # Plot with error bars
            ax.errorbar(
                impl_data['simulated_duration'],
                impl_data['real_time_factor_mean'],
                yerr=impl_data['real_time_factor_std'],
                marker=impl_info['marker'],
                color=impl_info['color'],
                label=impl_info['name'],
            )
            
            # Add value labels on points
            if SHOW_VALUE_LABELS:
                for _, row in impl_data.iterrows():
                    ax.annotate(
                        f"{row['real_time_factor_mean']:.2f}x",
                        xy=(row['simulated_duration'], row['real_time_factor_mean']),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        color=impl_info['color']
                    )
    
    # Add horizontal line at y=1 (real-time threshold)
    ax.axhline(
        y=1.0,
        color='black',
        linestyle='--',
        label='Umbral de Tiempo Real (1.0x)',
        zorder=1
    )
    
    # Add shaded regions
    ax.axhspan(
        0, 1.0,
        alpha=0.1,
        color='red',
        label='Más lento que tiempo real'
    )
    ax.axhspan(
        1.0, ax.get_ylim()[1],
        alpha=0.1,
        color='green',
        label='Más rápido que tiempo real'
    )
    
    # Labels and title
    ax.set_xlabel('Tiempo Simulado (segundos)')
    ax.set_ylabel('Factor de Tiempo Real\n(Tiempo Simulado / Tiempo Real)', 
                  )
    ax.set_title('Tiempo Simulado vs Tiempo Real - Escenario Escuela\n' +
                f'({CLASSROOM_COUNT} aulas, {PEDESTRIAN_COUNT} peatones)')
    
    # Add secondary x-axis with minutes
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Tiempo Simulado (minutos)')
    
    # Set ticks for secondary axis
    duration_ticks = [d for d in SIMULATION_DURATIONS]
    minute_labels = [f"{d/60:.1f}" for d in duration_ticks]
    ax2.set_xticks(duration_ticks)
    ax2.set_xticklabels(minute_labels)
    
    # Grid
    ax.grid(True)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='best')
    
    # Get number of iterations from first result
    num_iterations = df['num_iterations'].iloc[0] if len(df) > 0 else 0
    
    # Add information box
    info_text = (
        f'Escenario: Escuela (pasillos)\n'
        f'Grilla: {GRID_DIVISIONS}×{GRID_DIVISIONS} ({GRID_SIZE}m × {GRID_SIZE}m)\n'
        f'Aulas: {CLASSROOM_COUNT} | Peatones: {PEDESTRIAN_COUNT}\n'
        f'Iteraciones por punto: {num_iterations}'
    )
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment='top'
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'experiments/simulated_vs_real_time_school/simulated_vs_real_time_comparison.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Also create a secondary plot: Real execution time comparison
    plot_execution_time_comparison(results)


def plot_execution_time_comparison(results):
    """
    Generate a secondary plot comparing absolute real execution times.
    """
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=SECONDARY_FIGURE_SIZE)
    
    # Plot each implementation
    for impl_id, impl_info in IMPLEMENTATIONS.items():
        impl_data = df[df['implementation_id'] == impl_id].sort_values('simulated_duration')
        
        if not impl_data.empty:
            # Plot with error bars
            ax.errorbar(
                impl_data['simulated_duration'],
                impl_data['real_time_mean'],
                yerr=impl_data['real_time_std'],
                marker=impl_info['marker'],
                color=impl_info['color'],
                label=impl_info['name'],
            )
    
    # Add diagonal line representing real-time execution (simulated time = real time)
    max_duration = df['simulated_duration'].max()
    ax.plot(
        [0, max_duration],
        [0, max_duration],
        color='black',
        linestyle='--',
        label='Línea de Tiempo Real (1:1)'
    )
    
    ax.set_xlabel('Tiempo Simulado (segundos)')
    ax.set_ylabel('Tiempo Real de Ejecución (segundos)')
    ax.set_title('Comparación de Tiempo de Ejecución Real - Escenario Escuela\n' +
                f'({CLASSROOM_COUNT} aulas, {PEDESTRIAN_COUNT} peatones)')
    
    ax.grid(True)
    ax.legend(loc='best')
    
    # Get number of iterations from first result
    num_iterations = df['num_iterations'].iloc[0] if len(df) > 0 else 0
    
    # Add information box
    info_text = (
        f'Escenario: Escuela (pasillos)\n'
        f'Grilla: {GRID_DIVISIONS}×{GRID_DIVISIONS} ({GRID_SIZE}m × {GRID_SIZE}m)\n'
        f'Iteraciones: {num_iterations}\n'
        f'Nota: Puntos por debajo de la línea diagonal\n'
        f'indican simulación más rápida que tiempo real'
    )
    
    ax.text(
        0.98, 0.02,
        info_text,
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right'
    )
    
    plt.tight_layout()
    
    output_path = 'experiments/simulated_vs_real_time_school/execution_time_comparison.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"Gráfico de tiempo de ejecución guardado en: {output_path}")


if __name__ == '__main__':
    simulated_vs_real_time_school()
