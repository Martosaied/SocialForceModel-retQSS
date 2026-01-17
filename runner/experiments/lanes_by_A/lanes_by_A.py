import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import load_config, create_output_dir
from src.constants import Constants
from src.math.Clustering import Clustering
from src.experiments import (
    ConfigBuilder, ModelUpdater, ExperimentRunner,
    apply_publication_style, DEFAULT_COLORS
)

apply_publication_style()


A = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4, 5, 6, 7, 8, 9, 10]
WIDTH = 10
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_by_A():
    print("Ejecutando iteraciones para peatones con ancho fijo y graficando carriles por A...\n")
    for a in A:
        print(f"Ejecutando experimento para A: {a}")
        run(a)

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(a):
    """Execute experiment for given A parameter."""
    config = load_config('experiments/lanes_by_A/config.json')
    output_dir = create_output_dir(f'experiments/lanes_by_A/results/A_{a}')
    print(f"Directorio de salida creado: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * VOLUMES)

    # Use ConfigBuilder for cleaner parameter setup
    ConfigBuilder(config) \
        .set_pedestrian_count(pedestrians) \
        .set_pedestrian_implementation(Constants.PEDESTRIAN_MMOC) \
        .set_corridor(VOLUMES, WIDTH) \
        .set_parameter('PEDESTRIAN_A_2', a)
    
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update model parameters
    model = ModelUpdater('../retqss/model/social_force_model.mo')
    model.update_parameters({
        'GRID_DIVISIONS': VOLUMES,
        'N': pedestrians
    })

    # Use ExperimentRunner for compilation and execution
    ExperimentRunner.run_standard_experiment(config, output_dir, 'social_force_model')

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")


def plot_results():
    """
    Grafica los resultados de los experimentos.
    """
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/lanes_by_A/results') if os.path.isdir(os.path.join('experiments/lanes_by_A/results', d))]

    plt.figure(figsize=(10, 10))
    plt.title('Número promedio de grupos por A')
    plt.xlabel('A')
    plt.ylabel('Número de grupos')

    # Leer los directorios de resultados
    average_groups_per_A = {
        a: []
        for a in A
    }
    std_groups_per_A = {
        a: []
        for a in A
    }
    for result_dir in results_dirs:
        a = float(result_dir.split('_')[1])
        for result_file in os.listdir(os.path.join('experiments/lanes_by_A/results', result_dir, 'latest')):
            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/lanes_by_A/results', result_dir, 'latest', result_file))
                particles = (len(df.columns) - 1) / 5
                groups_per_A = []
                for index, row in df.iterrows():
                    if index < 100 and index % 5 != 0:
                        continue
                    groups = Clustering(
                        row, 
                        int(particles), 
                    ).calculate_groups(
                        from_y=(VOLUMES/ 2) - int(WIDTH / 2), 
                        to_y=(VOLUMES/ 2) + int(WIDTH / 2)
                    )
                    groups_per_A.append(len(groups))

                average_groups_per_A[a].append(np.mean(groups_per_A))

    # Promediar los grupos por ancho
    for a in A:
        std_groups_per_A[a] = np.std(average_groups_per_A[a])
        average_groups_per_A[a] = np.mean(average_groups_per_A[a])

    # Ordenar los grupos por ancho
    average_groups_per_A = dict(sorted(average_groups_per_A.items(), key=lambda item: item[0]))
    std_groups_per_A = dict(sorted(std_groups_per_A.items(), key=lambda item: item[0]))

    n_groups = np.array(list(average_groups_per_A.values()))
    std_n_groups = np.array(list(std_groups_per_A.values()))
    As = np.array(list(average_groups_per_A.keys()))


    plt.errorbar(As, n_groups, yerr=std_n_groups, fmt='o', label='Puntos de Datos')
    # # Ajustar línea usando numpy polyfit (grado 1 = lineal)
    # slope, intercept = np.polyfit(Bs, n_groups, 1)
    # line = slope * Bs + intercept
    # plt.plot(Bs, line, label='Línea Ajustada', color='red')

    plt.legend()
    plt.xlabel('A')
    plt.ylabel('Número de grupos')
    plt.title('Gráfico de Dispersión con Línea de Mejor Ajuste')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_A/groups_by_A.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_A()
