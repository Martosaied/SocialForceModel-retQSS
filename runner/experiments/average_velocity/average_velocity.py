import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
    Este experimento se utiliza para graficar la velocidad promedio de los peatones en diferentes escenarios.
    Queremos ver si la velocidad promedio se ve afectada por la formación de carriles.

    Calcularemos la velocidad promedio en 6 momentos diferentes de la simulación para diferentes números de volúmenes.
    También queremos generar el frame de la simulación para cada momento para ver la formación de carriles.
"""

GRID_DIVISIONS = [1, 2, 3, 5, 10, 20, 50, 100]

def average_velocity():
    print("Ejecutando experimentos para diferentes números de volúmenes...\n")
    for n in GRID_DIVISIONS:
        print(f"Ejecutando experimento para {n} volúmenes...")
        run(n)
        print(f"Experimento para {n} volúmenes completado.\n")

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(n):
    """
    Ejecuta el experimento para un número dado de volúmenes.
    """
    config = load_config('config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(
        'experiments/average_velocity/results', 
        f'n_{n}'
    )
    print(f"Directorio de salida creado: {output_dir}")

    config['parameters']['N']['value'] = 1000 # N
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = 1 # PEDESTRIAN_IMPLEMENTATION

    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(n) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = 1000/', '../retqss/model/social_force_model.mo'])

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
    )

    # Copiar resultados del directorio de salida al directorio latest
    copy_results_to_latest(output_dir)

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")


def plot_results():
    """
    Grafica los resultados de los experimentos.
    """
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/average_velocity/results') if os.path.isdir(os.path.join('experiments/average_velocity/results', d))]
    
    # Ordenar los directorios de resultados por N
    results_dirs = sorted(results_dirs, key=lambda x: int(x.split('n_')[1]))

    times = [5, 10, 15, 20, 25, 29]
    average_velocities = {
        1: [],
        2: [],
        3: [],
        5: [],
        10: [],
        20: [],
        50: [],
        100: []
    }

    # Obtener la velocidad promedio de los resultados
    for results_dir in results_dirs:
        data = pd.read_csv(os.path.join('experiments/average_velocity/results', results_dir, 'latest', 'result_0.csv'))
        volumes = int(results_dir.split('n_')[1])
        for time in times:
            velocities = data[data['time'] == float(time)]

            vx_columns = [col for col in velocities.columns if col.startswith("VX")]
            vy_columns = [col for col in velocities.columns if col.startswith("VY")]
            velocities_x = velocities[vx_columns]
            velocities_y = velocities[vy_columns]
                        
            # Calcular la velocidad promedio
            avg_velocities_x = velocities_x.mean(axis=1).values[0]
            avg_velocities_y = velocities_y.mean(axis=1).values[0]

            # Calcular la longitud del vector de velocidad
            avg_velocities = np.sqrt(avg_velocities_x**2 + avg_velocities_y**2)
            average_velocities[volumes].append(avg_velocities)

    # Generar un gráfico para cada número de volúmenes
    # Poner cada histograma en comparación con el de 1 volumen
    # El eje y es la velocidad promedio y el eje x es el tiempo
    plt.figure(figsize=(10, 5))
    plt.plot(times, average_velocities[1], label=f'1 volumen')
    for volumes in GRID_DIVISIONS[1:]: 
        plt.plot(times, average_velocities[volumes], label=f'{volumes} volúmenes')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad promedio')
    plt.title('Velocidad promedio de peatones en diferentes escenarios')
    
    plt.legend()
    plt.savefig(f'experiments/average_velocity/results/average_velocity.png')
    plt.close()

if __name__ == '__main__':
    average_velocity()
