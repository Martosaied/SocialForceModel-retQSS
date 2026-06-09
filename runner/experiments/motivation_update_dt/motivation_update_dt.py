import argparse
import json
import os
import subprocess
import sys

# Add current directory for importing personal_space_invation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personal_space_invation import (
    load_multiple_experiments,
    load_data_from_csv,
    save_results_to_csv,
    plot_comparative_analysis,
)

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src import utils
from src.constants import Constants
from src.experiments import apply_publication_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.math.Clustering import Clustering
from src.plots.DensityRowGraph import DensityRowGraph


MOTIVATION_UPDATE_DT = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Valores de motivation update dt a probar

WIDTH = 50
PEDESTRIAN_COUNT = int(50 * (50 * 0.4) * 0.3)
    
run_simulation = False


def get_simulated_time():
    """
    Obtiene el tiempo simulado desde el archivo config.json.
    """
    config = load_config('experiments/motivation_update_dt/config.json')
    return config['parameters']['FORCE_TERMINATION_AT']['value']


def motivation_update_dt(csv_path=None):
    print(f"Ejecutando iteraciones para {PEDESTRIAN_COUNT} peatones variando Motivation Update DT y graficando carriles...\n")
    if run_simulation:
        for motivation_dt in MOTIVATION_UPDATE_DT:
            print(f"Ejecutando experimento para motivation_update_dt: {motivation_dt}")
            run(motivation_dt)

    # Graficar los resultados
    plot_results(invasion_csv_path=csv_path)

def run(motivation_dt):
    """
    Ejecuta el experimento para un valor dado de motivation_update_dt.
    """
    config = load_config('experiments/motivation_update_dt/config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(f'experiments/motivation_update_dt/results/motivation_dt_{motivation_dt}')
    print(f"Directorio de salida creado: {output_dir}")

    config['iterations'] = 30
    config['parameters']['N']['value'] = PEDESTRIAN_COUNT
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE
    config['parameters']['MOTIVATION_UPDATE_DT']['value'] = motivation_dt
    config['parameters']['GROUPS_START_INDEX']['value'] = 500

    # Agregar desde dónde hasta dónde se generan los peatones
    config['parameters']['FROM_Y']['value'] = 15
    config['parameters']['TO_Y']['value'] = 35

    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Motivation Update DT={motivation_dt}")

    # Reemplazar las divisiones de la grilla en el modelo
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/helbing_only_qss.mo'])

    # Compilar el código C++ si se solicita
    compile_c_code()

    # Compilar el modelo si se solicita
    compile_model('helbing_only_qss')

    # Ejecutar experimento
    run_experiment(
        config, 
        output_dir, 
        'helbing_only_qss', 
        plot=False, 
        copy_results=True
    )

    # Copiar resultados del directorio de salida al directorio latest
    copy_results_to_latest(output_dir)

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")

def plot_results(invasion_csv_path=None):
    """
    Grafica los resultados del experimento de Motivation Update DT.
    """
    print("Generando gráficos de resultados...")
    
    # Aplicar estilo de publicación (incluye fuente sans-serif)
    apply_publication_style()
    
    # Obtener el tiempo simulado desde el config
    simulated_time = get_simulated_time()
    print(f"Tiempo simulado: {simulated_time} segundos")
    
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/motivation_update_dt/results') 
                   if os.path.isdir(os.path.join('experiments/motivation_update_dt/results', d))]

    # Inicializar almacenamiento de datos
    data = {
        'motivation_dts': [],
        'groups_mean': [],
        'groups_std': [],
        'speed_ratio_mean': [],
        'speed_ratio_std': [],
        'time_mean': [],  # Execution time in seconds
        'time_std': [],   # Execution time std
        'memory_mean': [],
        'memory_std': [],
        'motivation_updates_per_sec_mean': [],
        'motivation_updates_per_sec_std': [],
        'groups_data': [],
        'speed_ratio_data': [],
        'time_data': [],  # Execution time data in seconds
        'memory_data': [],
        'motivation_updates_per_sec_data': []
    }

    # Recopilar datos de archivos metrics.csv
    for result_dir in results_dirs:
        # Si el resultado no es de un valor de motivation_update_dt, se salta
        if not any(str(item) in result_dir for item in MOTIVATION_UPDATE_DT):
            continue
        
        try:
            motivation_dt = float(result_dir.split('motivation_dt_')[1])
            metrics_path = os.path.join('experiments/motivation_update_dt/results', result_dir, 'latest', 'metrics.csv')
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Extraer datos de grupos
                groups_data = df['clustering_based_groups'].dropna().tolist()
                time_data = df['time'].dropna().tolist()
                memory_data = df['memory_usage'].dropna().tolist()
                
                if groups_data:
                    # Calcular estadísticas
                    groups_mean = np.mean(groups_data)
                    groups_std = np.std(groups_data, ddof=1)
                    
                    # time_data is already in seconds
                    time_data_seconds = time_data
                    
                    # Calcular simulation speed ratio (simulated_time / execution_time)
                    # Higher is better (simulation runs faster than real-time)
                    speed_ratio_data = [simulated_time / t for t in time_data_seconds]
                    
                    # Calcular motivation updates per second
                    # Total motivation updates = simulated_time / motivation_update_dt
                    total_motivation_updates = simulated_time / motivation_dt
                    motivation_updates_per_sec = [total_motivation_updates / t for t in time_data_seconds]
                    
                    # Almacenar resultados
                    data['motivation_dts'].append(motivation_dt)
                    data['groups_mean'].append(groups_mean)
                    data['groups_std'].append(groups_std)
                    data['speed_ratio_mean'].append(np.mean(speed_ratio_data) if speed_ratio_data else 0)
                    data['speed_ratio_std'].append(np.std(speed_ratio_data, ddof=1) if speed_ratio_data else 0)
                    data['time_mean'].append(np.mean(time_data_seconds) if time_data_seconds else 0)
                    data['time_std'].append(np.std(time_data_seconds, ddof=1) if time_data_seconds else 0)
                    data['memory_mean'].append(np.mean(memory_data) if memory_data else 0)
                    data['memory_std'].append(np.std(memory_data, ddof=1) if memory_data else 0)
                    data['motivation_updates_per_sec_mean'].append(np.mean(motivation_updates_per_sec))
                    data['motivation_updates_per_sec_std'].append(np.std(motivation_updates_per_sec, ddof=1))
                    data['groups_data'].append(groups_data)
                    data['speed_ratio_data'].append(speed_ratio_data)
                    data['time_data'].append(time_data_seconds)
                    data['memory_data'].append(memory_data)
                    data['motivation_updates_per_sec_data'].append(motivation_updates_per_sec)
                    
                    print(f"Procesado dt={motivation_dt}: {len(groups_data)} puntos de datos")
                    
        except Exception as e:
            print(f"Advertencia: No se pudo procesar {result_dir}: {e}")
            continue

    if not data['motivation_dts']:
        print("¡No se encontraron datos para graficar!")
        return

    # Run personal space invasion analysis (use CSV if passed or available for speed)
    print("\n🔍 Ejecutando análisis de invasiones del espacio personal...")
    exp_dir = 'experiments/motivation_update_dt'
    csv_path = invasion_csv_path or os.path.join(exp_dir, 'personal_space_invasion_results.csv')
    if csv_path and os.path.exists(csv_path):
        experiments_data = load_data_from_csv(csv_path)
    else:
        results_dir = os.path.join(exp_dir, 'results')
        config_path = os.path.join(exp_dir, 'config.json')
        experiments_data = load_multiple_experiments(results_dir, config_path)
        if experiments_data:
            save_results_to_csv(experiments_data, exp_dir)
    if experiments_data:
        plot_comparative_analysis(experiments_data, exp_dir)
        invasion_by_dt = {
            dt: experiments_data[dt]['aggregated_stats']
            for dt in experiments_data
        }
    else:
        invasion_by_dt = {}
        print("⚠️ No se encontraron datos de invasiones para incluir en la tabla")

    # Ordenar datos por motivation_dt
    sorted_indices = np.argsort(data['motivation_dts'])
    for key in data:
        if key not in ['groups_data', 'speed_ratio_data', 'time_data', 'memory_data', 'motivation_updates_per_sec_data']:
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    motivation_dts = data['motivation_dts']
    groups_means = data['groups_mean']
    groups_stds = data['groups_std']
    speed_ratio_means = data['speed_ratio_mean']
    speed_ratio_stds = data['speed_ratio_std']
    time_means = data['time_mean']
    time_stds = data['time_std']
    
    # Gráfico 1: Grupos de carriles vs Motivation Update DT
    fig1, ax1 = plt.subplots(figsize=(18, 14))
    
    bars1 = ax1.bar(range(len(motivation_dts)), groups_means, yerr=groups_stds, 
                   color='lightgreen', width=0.6)
    ax1.set_xlabel('Motivation Update DT (segundos)')
    ax1.set_ylabel('Número de Carriles')
    ax1.set_xticks(range(len(motivation_dts)))
    ax1.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=45, ha='right')
    max_groups = max(groups_means) if len(groups_means) > 0 else 0
    if max_groups > 0:
        ax1.set_ylim(0, max_groups * 1.4)
    ax1.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, top=0.94, right=0.98)
    plt.savefig('experiments/motivation_update_dt/lane_formation_by_motivation_dt.png')
    plt.close()
    
    # Gráfico 2: Tiempo de Ejecución vs Motivation Update DT
    fig2, ax2 = plt.subplots(figsize=(18, 14))

    bars2 = ax2.bar(range(len(motivation_dts)), time_means, yerr=time_stds,
                   color='skyblue', width=0.6)
    ax2.set_xlabel('Motivation Update DT (segundos)')
    ax2.set_ylabel('Tiempo promedio de ejecución (segundos)')
    ax2.set_xticks(range(len(motivation_dts)))
    ax2.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=45, ha='right')

    # Set y-axis limits
    max_val = max(time_means) if len(time_means) > 0 else 1
    max_std = max(time_stds) if len(time_stds) > 0 else 0
    y_max = (max_val + max_std) * 1.4
    ax2.set_ylim(0, y_max)

    # Add horizontal line at simulated time for reference
    ax2.axhline(y=simulated_time, color='green', linestyle='--', linewidth=2, alpha=0.5)
    legend_handles = [
        Line2D([0], [0], color='green', linestyle='--', label=f'Tiempo simulado: {simulated_time:.0f}s'),
    ]
    ax2.legend(handles=legend_handles, loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, top=0.94, right=0.98)
    plt.savefig('experiments/motivation_update_dt/performance_by_motivation_dt.png')
    plt.close()
    
    # Imprimir resumen
    print("\n" + "="*100)
    print("RESUMEN DE RESULTADOS MOTIVATION UPDATE DT")
    print("="*100)
    print(f"{'Motivation DT':<15} {'Carriles':<15} {'Colisiones':<18} {'RTF (ratio)':<15} {'Tiempo Ejec (s)':<18}")
    print("-" * 100)
    
    for i, dt in enumerate(motivation_dts):
        groups_mean = groups_means[i]
        groups_std = groups_stds[i]
        speed_ratio_mean = speed_ratio_means[i]
        speed_ratio_std = speed_ratio_stds[i]
        time_mean = time_means[i]
        time_std = time_stds[i]
        coll_str = (f"{invasion_by_dt[dt]['total_invasions_mean']:.1f}±{invasion_by_dt[dt]['total_invasions_std']:.1f}"
                    if dt in invasion_by_dt else "N/A")
        print(f"{dt:<15.3f} {groups_mean:.1f}±{groups_std:.1f}         {coll_str:<18} "
              f"{speed_ratio_mean:.2f}x±{speed_ratio_std:.2f}     {time_mean:.2f}±{time_std:.2f}")
    
    print("="*100)
    print(f"\nNota: RTF (Real-Time Factor) = tiempo simulado ({simulated_time:.0f}s) / tiempo de ejecución")
    print("      RTF > 1.0 significa que la simulación corre más rápido que tiempo real")
    print("      Ejemplo: RTF = 5.0x significa 5 veces más rápido que tiempo real")

    # Generate LaTeX table with exact values (including collisions from personal space invasion)
    latex_dir = 'experiments/motivation_update_dt'
    os.makedirs(latex_dir, exist_ok=True)
    latex_path = os.path.join(latex_dir, 'motivation_update_dt_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Resultados del experimento Motivation Update DT: Carriles, Colisiones (invasiones espacio personal), RTF y tiempo de ejecución.}' + '\n')
        f.write(r'\label{tab:motivation_update_dt}' + '\n')
        f.write(r'\begin{tabular}{cccccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'Motivation DT (s) & Carriles (mean $\pm$ std) & Colisiones (mean $\pm$ std) & RTF (mean $\pm$ std) & Tiempo Ejec (s) (mean $\pm$ std) \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, dt in enumerate(motivation_dts):
            groups_mean = groups_means[i]
            groups_std = groups_stds[i]
            speed_ratio_mean = speed_ratio_means[i]
            speed_ratio_std = speed_ratio_stds[i]
            time_mean = time_means[i]
            time_std = time_stds[i]
            if dt in invasion_by_dt:
                inv_mean = invasion_by_dt[dt]['total_invasions_mean']
                inv_std = invasion_by_dt[dt]['total_invasions_std']
                coll_cell = f'{inv_mean:.2f} $\\pm$ {inv_std:.2f}'
            else:
                coll_cell = '--'
            f.write(f'{dt:.3f} & {groups_mean:.2f} $\\pm$ {groups_std:.2f} & {coll_cell} & {speed_ratio_mean:.4f} $\\pm$ {speed_ratio_std:.4f} & {time_mean:.2f} $\\pm$ {time_std:.2f} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"\nLaTeX table saved to: {latex_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motivation Update DT experiment')
    parser.add_argument('--csv', '-c', help='Path to personal_space_invasion_results.csv to use (skips reprocessing result files)')
    args = parser.parse_args()
    motivation_update_dt(csv_path=args.csv)
