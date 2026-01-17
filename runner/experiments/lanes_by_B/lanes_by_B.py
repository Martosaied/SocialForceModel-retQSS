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

B = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.0]
WIDTH = 15
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_by_B():
    print("Ejecutando iteraciones para peatones con ancho fijo y graficando carriles por B...\n")
    for b in B:
        print(f"Ejecutando experimento para B: {b}")
        # run(b)

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(b):
    """Execute experiment for given B parameter."""
    config = load_config('experiments/lanes_by_B/config.json')
    output_dir = create_output_dir(f'experiments/lanes_by_B/results/B_{b}')
    print(f"Directorio de salida creado: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * VOLUMES)

    # Use ConfigBuilder for cleaner parameter setup
    ConfigBuilder(config) \
        .set_pedestrian_count(pedestrians) \
        .set_pedestrian_implementation(Constants.PEDESTRIAN_MMOC) \
        .set_corridor(VOLUMES, WIDTH) \
        .set_parameter('PEDESTRIAN_B_2', b) \
        .set_parameter('PEDESTRIAN_B_1', b)

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
    results_dirs = [d for d in os.listdir('experiments/lanes_by_B/results') if os.path.isdir(os.path.join('experiments/lanes_by_B/results', d))]

    plt.figure(figsize=(10, 10))
    plt.title('Número promedio de grupos por B')
    plt.xlabel('B')
    plt.ylabel('Número de grupos')

    # Leer los directorios de resultados
    average_groups_per_B = {
        b: []
        for b in B
    }
    std_groups_per_B = {
        b: []
        for b in B
    }
    for result_dir in results_dirs:
        b = float(result_dir.split('_')[1])
        for result_file in os.listdir(os.path.join('experiments/lanes_by_B/results', result_dir, 'latest')):
            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/lanes_by_B/results', result_dir, 'latest', result_file))
                particles = (len(df.columns) - 1) / 5
                groups_per_B = []
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
                    groups_per_B.append(len(groups))

                average_groups_per_B[b].append(np.mean(groups_per_B))

    # Promediar los grupos por ancho
    for b in B:
        std_groups_per_B[b] = np.std(average_groups_per_B[b])
        average_groups_per_B[b] = np.mean(average_groups_per_B[b])

    # Ordenar los grupos por ancho
    average_groups_per_B = dict(sorted(average_groups_per_B.items(), key=lambda item: item[0]))
    std_groups_per_B = dict(sorted(std_groups_per_B.items(), key=lambda item: item[0]))

    n_groups = np.array(list(average_groups_per_B.values()))
    std_n_groups = np.array(list(std_groups_per_B.values()))
    Bs = np.array(list(average_groups_per_B.keys()))


    plt.errorbar(Bs, n_groups, yerr=std_n_groups, fmt='o', label='Puntos de Datos')
    # # Ajustar línea usando numpy polyfit (grado 1 = lineal)
    # slope, intercept = np.polyfit(Bs, n_groups, 1)
    # line = slope * Bs + intercept
    # plt.plot(Bs, line, label='Línea Ajustada', color='red')

    plt.legend()
    plt.xlabel('B')
    plt.ylabel('Número de grupos')
    plt.title('Gráfico de Dispersión con Línea de Mejor Ajuste')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_B/groups_by_B.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_B()
