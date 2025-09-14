import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.constants import Constants

VOLUMES = [5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
PEDESTRIANS_IMPLEMENTATION = {
    Constants.PEDESTRIAN_NEIGHBORHOOD: "retqss",
}
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3

"""
Generar una matriz con 0s y 1s, donde 0 es un espacio libre y 1 es un obstáculo.
La matriz es una matriz cuadrada con el tamaño volúmenes * volúmenes.
Los obstáculos se colocan en el centro de la matriz como una pared que divide la matriz en dos.
El índice comienza desde la esquina inferior izquierda como 1. Y asciende hacia arriba a través de la columna.
"""
def generate_obstacles(volumes):
    obstacles = np.zeros((volumes, volumes))
    obstacles[volumes//2, :] = 1
    return obstacles.T.tolist()
    

def breaking_obstacles():
    print("Ejecutando experimentos para 300 peatones en diferentes volúmenes para ver si los carriles se rompen...\n")
    for volume in VOLUMES:
        for implementation in PEDESTRIANS_IMPLEMENTATION:
            print(f"Ejecutando experimento para {volume} volúmenes con implementación {implementation}...")
            run(volume, implementation)
            print(f"Experimento para {volume} volúmenes con implementación {implementation} completado.\n")

    # Graficar los resultados
    print("Graficando resultados...")
    # plot_results()

def run(volume, implementation):
    """
    Ejecuta el experimento para un número dado de peatones.
    """
    config = load_config('./experiments/breaking_obstacles/config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(
        'experiments/breaking_obstacles/results', 
        f'volume_{volume}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Directorio de salida creado: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
    config['iterations'] = 1
    config['parameters']['N']['value'] = pedestrians
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_NEIGHBORHOOD
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_SURROUNDING_VOLUMES
    config['parameters']['OBSTACLES'] = {
        'name': 'OBSTACLES',
        'map': generate_obstacles(volume),
        'type': 'map'
    }

    # Agregar desde dónde hasta dónde se generan los peatones
    config['parameters']['FROM_Y'] = {
      "name": "FROM_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) - int(WIDTH / 2)
    }
    config['parameters']['TO_Y'] = {
      "name": "TO_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) + int(WIDTH / 2)
    }


    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(volume) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(pedestrians) + '/', '../retqss/model/social_force_model.mo'])

    # Compilar el código C++ si se solicita
    compile_c_code()

    # Compilar el modelo si se solicita
    compile_model('social_force_model')

    # Ejecutar experimento
    run_experiment(
        config, 
        output_dir, 
        'social_force_model', 
        plot=False, 
        copy_results=True
    )

    # Copiar resultados del directorio de salida al directorio latest
    copy_results_to_latest(output_dir)

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")


def plot_results():
    """
    Plot the results of the experiments.
    """
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/breaking_obstacles/results') if os.path.isdir(os.path.join('experiments/breaking_obstacles/results', d))]

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": [],
        "no_helbing": []
    }
    for results_dir in results_dirs:
        implementation = results_dir.split('_implementation_')[1]
        results_dirs_by_implementation[implementation].append(results_dir)

    # Make 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.set_title('Mean velocity Y per amount of volumes')
    ax1.set_xlabel('Amount of volumes per side')
    ax1.set_ylabel('Mean velocity Y(m/s)')

    ax2.set_title('Clustering based groups per amount of volumes')
    ax2.set_xlabel('Amount of volumes per side')
    ax2.set_ylabel('Number of lanes formed')

    ax3.set_title('Time per amount of volumes')
    ax3.set_xlabel('Amount of volumes per side')
    ax3.set_ylabel('Time(ms)')

    ax4.set_title('Memory usage per amount of volumes')
    ax4.set_xlabel('Amount of volumes per side')
    ax4.set_ylabel('Memory usage(KB)')

    density_based_groups_per_volume = {}
    clustering_based_groups_per_volume = {}
    memory_per_volume = {}
    time_per_volume = {}
    mean_velocity_y_per_volume = {}
    for result_dir in results_dirs_by_implementation['no_helbing']:
        # Get the results files
        results_files = [f for f in os.listdir(os.path.join('experiments/breaking_obstacles/results', result_dir, 'latest')) if f == 'metrics.csv']

        # Read the results files
        df = pd.read_csv(os.path.join('experiments/breaking_obstacles/results', result_dir, 'latest', 'metrics.csv'))
        density_based_groups_per_volume[0] = np.mean(df['density_based_groups'].tolist())
        clustering_based_groups_per_volume[0] = np.mean( df['clustering_based_groups'].tolist())

    for result_dir in results_dirs_by_implementation['retqss']:
        amount_of_volumes = int(result_dir.split('volume_')[1].split('_')[0])
        # Get the results files
        results_files = [f for f in os.listdir(os.path.join('experiments/breaking_obstacles/results', result_dir, 'latest')) if f.endswith('.csv')]
        for result_file in results_files:
            df = pd.read_csv(os.path.join('experiments/breaking_obstacles/results', result_dir, 'latest', result_file))
            # Calculate average velocity on the Y axis
            prev_mean_velocity_y = None
            for index, row in df.iterrows():
                vy = row.get(f'VY[{index + 1}]')
                if vy is not None:
                    velocity_y = np.mean(row.get(f'VY[{index + 1}]'))
                    if prev_mean_velocity_y is not None:
                        if amount_of_volumes not in mean_velocity_y_per_volume:
                            mean_velocity_y_per_volume[amount_of_volumes] = []
                        mean_velocity_y_per_volume[amount_of_volumes].append(abs(velocity_y - prev_mean_velocity_y))
                    else:
                        prev_mean_velocity_y = velocity_y


        df = pd.read_csv(os.path.join('experiments/breaking_obstacles/results', result_dir, 'latest', 'metrics.csv'))
        density_based_groups_per_volume[amount_of_volumes] = np.mean(df['density_based_groups'].tolist())
        clustering_based_groups_per_volume[amount_of_volumes] = np.mean(df['clustering_based_groups'].tolist())
        memory_per_volume[amount_of_volumes] = np.mean(df['memory_usage'].tolist())
        time_per_volume[amount_of_volumes] = np.mean(df['time'].tolist())
        mean_velocity_y_per_volume[amount_of_volumes] = np.mean(mean_velocity_y_per_volume[amount_of_volumes])



    density_based_groups_per_volume = dict(sorted(density_based_groups_per_volume.items()))
    clustering_based_groups_per_volume = dict(sorted(clustering_based_groups_per_volume.items()))
    memory_per_volume = dict(sorted(memory_per_volume.items()))
    time_per_volume = dict(sorted(time_per_volume.items()))
    mean_velocity_y_per_volume = dict(sorted(mean_velocity_y_per_volume.items()))

    # Plot a boxplot per amount of volumes
    ax1.bar(list(map(str, mean_velocity_y_per_volume.keys())), mean_velocity_y_per_volume.values(), width=0.5, align='center')
    ax2.bar(list(map(str, clustering_based_groups_per_volume.keys())), clustering_based_groups_per_volume.values(), width=0.5, align='center')
    ax3.bar(list(map(str, time_per_volume.keys())), time_per_volume.values(), width=0.5, align='center')
    ax4.bar(list(map(str, memory_per_volume.keys())), memory_per_volume.values(), width=0.5, align='center')
    plt.legend()
    plt.savefig(f'experiments/breaking_obstacles/breaking_obstacles.png')
    plt.close()


if __name__ == '__main__':
    breaking_obstacles()
