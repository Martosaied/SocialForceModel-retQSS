import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src import utils
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.math.Clustering import Clustering
from src.plots.DensityRowGraph import DensityRowGraph


MOTIVATION_UPDATE_DT = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Valores de motivation update dt a probar

WIDTH = 50
PEDESTRIAN_COUNT = int(50 * (50 * 0.4) * 0.3)

run_simulation = True


def get_simulation_duration():
    """
    Obtiene la duración de la simulación desde el archivo config.json.
    """
    config = load_config('experiments/motivation_update_dt/config.json')
    return config['parameters']['FORCE_TERMINATION_AT']['value']


def motivation_update_dt():
    print(f"Ejecutando iteraciones para {PEDESTRIAN_COUNT} peatones variando Motivation Update DT y graficando carriles...\n")
    if run_simulation:
        for motivation_dt in MOTIVATION_UPDATE_DT:
            print(f"Ejecutando experimento para motivation_update_dt: {motivation_dt}")
            run(motivation_dt)

    # Graficar los resultados
    plot_results()

def run(motivation_dt):
    """
    Ejecuta el experimento para un valor dado de motivation_update_dt.
    """
    config = load_config('experiments/motivation_update_dt/config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(f'experiments/motivation_update_dt/results/motivation_dt_{motivation_dt}')
    print(f"Directorio de salida creado: {output_dir}")

    config['iterations'] = 10
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

def plot_results():
    """
    Grafica los resultados del experimento de Motivation Update DT.
    """
    print("Generando gráficos de resultados...")
    
    # Obtener la duración de la simulación desde el config
    simulation_duration = get_simulation_duration()
    print(f"Duración de la simulación: {simulation_duration} segundos")
    
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/motivation_update_dt/results') 
                   if os.path.isdir(os.path.join('experiments/motivation_update_dt/results', d))]

    # Inicializar almacenamiento de datos
    data = {
        'motivation_dts': [],
        'groups_mean': [],
        'groups_std': [],
        'time_mean': [],
        'time_std': [],
        'memory_mean': [],
        'memory_std': [],
        'motivation_updates_per_sec_mean': [],
        'motivation_updates_per_sec_std': [],
        'groups_data': [],
        'time_data': [],
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
                    
                    # Calcular motivation updates per second
                    # Total motivation updates = simulation_duration / motivation_update_dt
                    total_motivation_updates = simulation_duration / motivation_dt
                    
                    # Convertir tiempo de ms a segundos y calcular updates per second
                    time_data_seconds = [t / 1000.0 for t in time_data]  # Convertir ms a segundos
                    motivation_updates_per_sec = [total_motivation_updates / t for t in time_data_seconds]
                    
                    # Almacenar resultados
                    data['motivation_dts'].append(motivation_dt)
                    data['groups_mean'].append(groups_mean)
                    data['groups_std'].append(groups_std)
                    data['time_mean'].append(np.mean(time_data) if time_data else 0)
                    data['time_std'].append(np.std(time_data, ddof=1) if time_data else 0)
                    data['memory_mean'].append(np.mean(memory_data) if memory_data else 0)
                    data['memory_std'].append(np.std(memory_data, ddof=1) if memory_data else 0)
                    data['motivation_updates_per_sec_mean'].append(np.mean(motivation_updates_per_sec))
                    data['motivation_updates_per_sec_std'].append(np.std(motivation_updates_per_sec, ddof=1))
                    data['groups_data'].append(groups_data)
                    data['time_data'].append(time_data)
                    data['memory_data'].append(memory_data)
                    data['motivation_updates_per_sec_data'].append(motivation_updates_per_sec)
                    
                    print(f"Procesado dt={motivation_dt}: {len(groups_data)} puntos de datos")
                    
        except Exception as e:
            print(f"Advertencia: No se pudo procesar {result_dir}: {e}")
            continue

    if not data['motivation_dts']:
        print("¡No se encontraron datos para graficar!")
        return

    # Ordenar datos por motivation_dt
    sorted_indices = np.argsort(data['motivation_dts'])
    for key in data:
        if key not in ['groups_data', 'time_data', 'memory_data', 'motivation_updates_per_sec_data']:
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    # Crear gráfico principal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Análisis de Motivation Update DT: Formación de Carriles y Rendimiento', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Grupos de carriles vs Motivation Update DT
    motivation_dts = data['motivation_dts']
    groups_means = data['groups_mean']
    groups_stds = data['groups_std']
    
    bars1 = ax1.bar(range(len(motivation_dts)), groups_means, yerr=groups_stds, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Motivation Update DT')
    ax1.set_ylabel('Número de Grupos de Carriles')
    ax1.set_title('Formación de Carriles vs Motivation Update DT')
    ax1.set_xticks(range(len(motivation_dts)))
    ax1.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=45)
    ax1.set_ylim(0, max(groups_means) + max(groups_means) * 0.4)
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras (solo si hay espacio suficiente)
    for i, (bar, mean, std) in enumerate(zip(bars1, groups_means, groups_stds)):
        height = bar.get_height()
        if height + std > 0:  # Solo mostrar si la barra es visible
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(groups_means) * 0.05,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=11, rotation=90)
    
    # Gráfico 2: Rendimiento vs Motivation Update DT
    time_means = data['time_mean']
    time_stds = data['time_std']
    
    bars2 = ax2.bar(range(len(motivation_dts)), time_means, yerr=time_stds, 
                   capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Motivation Update DT')
    ax2.set_ylabel('Tiempo de Ejecución (ms)')
    ax2.set_title('Rendimiento vs Motivation Update DT')
    ax2.set_xticks(range(len(motivation_dts)))
    ax2.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=45)
    ax2.set_ylim(0, max(time_means) + max(time_means) * 0.4)
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras (solo si hay espacio suficiente)
    for i, (bar, mean, std) in enumerate(zip(bars2, time_means, time_stds)):
        height = bar.get_height()
        if height + std > 0:  # Solo mostrar si la barra es visible
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(time_means) * 0.05,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=11, rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Más espacio para las etiquetas rotadas
    plt.savefig('experiments/motivation_update_dt/performance_by_motivation_dt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir resumen
    print("\n" + "="*90)
    print("RESUMEN DE RESULTADOS MOTIVATION UPDATE DT")
    print("="*90)
    print(f"{'Motivation DT':<12} {'Grupos':<10} {'Tiempo (ms)':<12}")
    print("-" * 90)
    
    for i, dt in enumerate(motivation_dts):
        groups_mean = groups_means[i]
        groups_std = groups_stds[i]
        time_mean = time_means[i]
        time_std = time_stds[i]
        print(f"{dt:<12.3f} {groups_mean:.1f}±{groups_std:.1f}     {time_mean:.1f}±{time_std:.1f}")
    
    print("="*90)

if __name__ == '__main__':
    motivation_update_dt()
