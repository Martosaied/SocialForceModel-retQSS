import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density


Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
WIDTH = 15
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_by_R():
    print("Ejecutando iteraciones para peatones con ancho fijo y graficando carriles por R...\n")
    for r in Rs:
        print(f"Ejecutando experimento para R: {r}")
        run(r)

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(r):
    """
    Ejecuta el experimento para un número dado de peatones.
    """
    config = load_config('experiments/lanes_by_R/config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(f'experiments/lanes_by_R/results/R_{r}')
    print(f"Directorio de salida creado: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * VOLUMES)

    config['parameters'][0]['value'] = pedestrians
    config['parameters'][1]['value'] = Constants.PEDESTRIAN_MMOC


    # Reemplazar el mapa en la configuración
    generated_map = generate_map(VOLUMES,WIDTH)
    config['parameters'].append({
      "name": "OBSTACLES",
      "type": "map",
      "map": generated_map
    })

    # Agregar desde dónde hasta dónde se generan los peatones
    config['parameters'].append({
      "name": "FROM_Y",
      "type": "value",
      "value": (VOLUMES/ 2) - int(WIDTH / 2)
    })
    config['parameters'].append({
      "name": "TO_Y",
      "type": "value",
      "value": (VOLUMES/ 2) + int(WIDTH / 2)
    })
    config['parameters'].append({
      "name": "PEDESTRIAN_R",
      "type": "value",
      "value": r
    })
    config['parameters'].append({
      "name": "BORDER_R",
      "type": "value",
      "value": r
    })

    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Reemplazar las divisiones de la grilla en el modelo
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/social_force_model.mo'])
    # Reemplazar los peatones en el modelo
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
    Grafica los resultados de los experimentos.
    """
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/lanes_by_R/results') if os.path.isdir(os.path.join('experiments/lanes_by_R/results', d))]

    plt.figure(figsize=(10, 10))
    plt.title('Número promedio de grupos por R')
    plt.xlabel('R')
    plt.ylabel('Número de grupos')

    # Leer los directorios de resultados
    average_groups_per_R = {
        r: []
        for r in Rs
    }
    std_groups_per_R = {
        r: []
        for r in Rs
    }
    for result_dir in results_dirs:
        r = float(result_dir.split('_')[1])
        if not os.path.exists(os.path.join('experiments/lanes_by_R/results', result_dir, 'latest')):
            continue
        for result_file in os.listdir(os.path.join('experiments/lanes_by_R/results', result_dir, 'latest')):
            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/lanes_by_R/results', result_dir, 'latest', result_file))
                particles = (len(df.columns) - 1) / 5
                groups = Density(
                    grid_size=100, 
                    map_size=100, 
                ).calculate_lanes_by_density(
                    df,
                    particles
                )
                average_groups_per_R[r].append(groups)

    # Promediar los grupos por ancho
    for r in Rs:
        std_groups_per_R[r] = np.std(average_groups_per_R[r])
        average_groups_per_R[r] = np.mean(average_groups_per_R[r])

    # Ordenar los grupos por ancho
    average_groups_per_R = dict(sorted(average_groups_per_R.items(), key=lambda item: item[0]))
    std_groups_per_R = dict(sorted(std_groups_per_R.items(), key=lambda item: item[0]))

    n_groups = np.array(list(average_groups_per_R.values()))
    std_n_groups = np.array(list(std_groups_per_R.values()))
    Rss = np.array(list(average_groups_per_R.keys()))


    plt.errorbar(Rss, n_groups, yerr=std_n_groups, fmt='o', label='Puntos de Datos')
    # # Ajustar línea usando numpy polyfit (grado 1 = lineal)
    # slope, intercept = np.polyfit(Bs, n_groups, 1)
    # line = slope * Bs + intercept
    # plt.plot(Bs, line, label='Línea Ajustada', color='red')

    plt.legend()
    plt.xlabel('R')
    plt.ylabel('Número de grupos')
    plt.title('Gráfico de Dispersión con Línea de Mejor Ajuste')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_R/groups_by_R.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_R()
