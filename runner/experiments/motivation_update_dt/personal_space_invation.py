#!/usr/bin/env python3
"""
Script para medir invasiones del espacio personal en simulaciones de peatones.
Cuenta cuántas veces las partículas se acercan demasiado (distancia < 2 * PEDESTRIAN_R)
basado en el parámetro PEDESTRIAN_R del config.json.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import sys
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
import warnings
import time
warnings.filterwarnings('ignore')

if __name__ == '__main__' or __package__ is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.experiments import apply_publication_style

def load_config(config_path):
    """
    Carga el archivo de configuración y extrae PEDESTRIAN_R.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        pedestrian_r = config['parameters']['PEDESTRIAN_R']['value']
        groups_start_index = config['parameters']['GROUPS_START_INDEX']['value']
        print(f"✅ Config cargado: PEDESTRIAN_R = {pedestrian_r}")
        return pedestrian_r, groups_start_index
    except Exception as e:
        print(f"❌ Error cargando config {config_path}: {e}")
        return None

def load_result_file(file_path, verbose=False):
    """
    Carga un archivo result_*.csv y retorna un DataFrame limpio.
    """
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"✅ Archivo cargado: {file_path}")
            print(f"   Dimensiones: {df.shape}")
        
        # Convertir formato de columnas por peatón a formato largo
        df_long = convert_to_long_format(df)
        if verbose:
            print(f"   Formato largo: {df_long.shape}")
        return df_long
    except Exception as e:
        print(f"❌ Error cargando {file_path}: {e}")
        return None

def convert_to_long_format(df):
    """
    Convierte el formato de columnas por peatón a formato largo (long format).
    """
    # Obtener número de peatones del número de columnas
    n_pedestrians = (len(df.columns) - 1) // 5  # -1 por la columna 'time', /5 por PX,PY,VX,VY,PS
    
    # Crear lista para almacenar datos reorganizados
    data_rows = []
    
    for _, row in df.iterrows():
        time = row['time']
        
        for i in range(1, n_pedestrians + 1):
            # Verificar si las columnas existen
            px_col = f'PX[{i}]'
            py_col = f'PY[{i}]'
            vx_col = f'VX[{i}]'
            vy_col = f'VY[{i}]'
            ps_col = f'PS[{i}]'
            
            if all(col in row for col in [px_col, py_col, vx_col, vy_col, ps_col]):
                # Solo incluir si el peatón está activo (PS != 0 o posición válida)
                if (not pd.isna(row[px_col]) and not pd.isna(row[py_col]) and 
                    row[px_col] != 0 and row[py_col] != 0):
                    
                    data_rows.append({
                        'time': time,
                        'pedestrian_id': i,
                        'x': row[px_col],
                        'y': row[py_col],
                        'velocity_x': row[vx_col],
                        'velocity_y': row[vy_col],
                        'state': row[ps_col]
                    })
    
    return pd.DataFrame(data_rows)

def calculate_personal_space_invasions(df, pedestrian_r, groups_start_index):
    """
    Calcula las invasiones del espacio personal para cada timestep.
    
    Args:
        df: DataFrame con datos de peatones en formato largo
        pedestrian_r: Radio del espacio personal (PEDESTRIAN_R)
        groups_start_index: Índice de inicio de grupos

    Returns:
        dict: Análisis de invasiones por timestep
    """
    print(f"🔍 Analizando invasiones del espacio personal (radio = {pedestrian_r})")
    
    # Personal space radius = PEDESTRIAN_R (diámetro del espacio personal)
    personal_space_radius = pedestrian_r
    
    invasion_data = []
    total_invasions = 0
    
    # Agrupar por timestep
    for time, group in df.groupby('time'):
        if time < groups_start_index / 10:
            continue

        if len(group) < 2:  # Necesitamos al menos 2 peatones
            continue
            
        # Obtener posiciones y velocidades de todos los peatones activos en este timestep
        positions = group[['x', 'y']].values
        velocities = group[['velocity_x', 'velocity_y']].values
        pedestrian_ids = group['pedestrian_id'].values
        
        # Calcular matriz de distancias
        distances = pdist(positions)
        distance_matrix = squareform(distances)
        
        # Umbral más estricto: 1.5 * radio en lugar de 2 * radio
        invasion_threshold = personal_space_radius * 1.5
        
        # Encontrar pares que están dentro del espacio personal
        invasion_pairs = []
        invasion_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = distance_matrix[i, j]
                
                # Verificar si están dentro del umbral más estricto
                if distance < invasion_threshold:
                    # Calcular direcciones de movimiento
                    vel_i = velocities[i]
                    vel_j = velocities[j]
                    
                    # Calcular magnitudes de velocidad
                    speed_i = np.linalg.norm(vel_i)
                    speed_j = np.linalg.norm(vel_j)
                    
                    # Si ambos se están moviendo (velocidad > umbral mínimo)
                    min_speed_threshold = 0.01  # Evitar divisiones por cero
                    if speed_i > min_speed_threshold and speed_j > min_speed_threshold:
                        # Normalizar vectores de velocidad para obtener direcciones
                        dir_i = vel_i / speed_i
                        dir_j = vel_j / speed_j
                        
                        # Calcular ángulo entre direcciones usando producto punto
                        # cos(angle) = dot(dir_i, dir_j)
                        cos_angle = np.dot(dir_i, dir_j)
                        # Limitar a [-1, 1] por posibles errores numéricos
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        
                        # Filtrar: si se mueven en la misma dirección (ángulo < 45 grados = π/4)
                        # No contar como invasión problemática
                        if angle < np.pi / 4:  # 45 grados
                            continue  # Saltar esta invasión
                    
                    # Si llegamos aquí, es una invasión problemática
                    invasion_pairs.append((pedestrian_ids[i], pedestrian_ids[j], distance))
                    invasion_count += 1
        
        invasion_data.append({
            'time': time,
            'n_pedestrians': len(group),
            'invasion_count': invasion_count,
            'invasion_pairs': invasion_pairs,
            'invasion_rate': invasion_count / (len(group) * (len(group) - 1) / 2) if len(group) > 1 else 0
        })
        
        total_invasions += invasion_count
    
    print(f"✅ Análisis completado: {total_invasions} invasiones totales en {len(invasion_data)} timesteps")
    
    return invasion_data

def analyze_invasion_statistics(invasion_data):
    """
    Analiza estadísticas de las invasiones del espacio personal.
    """
    if not invasion_data:
        return {}
    
    invasion_counts = [data['invasion_count'] for data in invasion_data]
    invasion_rates = [data['invasion_rate'] for data in invasion_data]
    n_pedestrians = [data['n_pedestrians'] for data in invasion_data]
    
    # Estadísticas básicas
    stats = {
        'total_invasions': sum(invasion_counts),
        'total_timesteps': len(invasion_data),
        'avg_invasions_per_timestep': np.mean(invasion_counts),
        'max_invasions_per_timestep': np.max(invasion_counts),
        'min_invasions_per_timestep': np.min(invasion_counts),
        'std_invasions_per_timestep': np.std(invasion_counts),
        'avg_invasion_rate': np.mean(invasion_rates),
        'max_invasion_rate': np.max(invasion_rates),
        'avg_pedestrians_per_timestep': np.mean(n_pedestrians),
        'invasion_frequency': sum(invasion_counts) / len(invasion_data) if invasion_data else 0
    }
    
    return stats

def calculate_aggregated_statistics(all_stats):
    """
    Calcula estadísticas agregadas (mean y std) a partir de múltiples runs.
    """
    if not all_stats:
        return {}
    
    # Extraer todas las métricas de todos los runs
    metrics = {}
    for stats in all_stats:
        for key, value in stats.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    
    # Calcular mean y std para cada métrica
    aggregated = {}
    for key, values in metrics.items():
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
        aggregated[f'{key}_min'] = np.min(values)
        aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated


def load_multiple_experiments(results_dir, config_path):
    """
    Carga datos de múltiples experimentos y analiza invasiones.
    Ahora carga todos los archivos result_*.csv y calcula mean y std.
    """
    # Cargar configuración
    pedestrian_r, groups_start_index = load_config(config_path)
    if pedestrian_r is None:
        return {}
    
    experiments_data = {}
    
    # Buscar directorios de resultados
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith('motivation_dt_'):
            try:
                motivation_dt = float(item.split('motivation_dt_')[1])
                latest_dir = os.path.join(item_path, 'latest')
                
                if os.path.exists(latest_dir):
                    # Buscar todos los archivos result_*.csv
                    result_files = []
                    for file in os.listdir(latest_dir):
                        if file.startswith('result_') and file.endswith('.csv'):
                            result_files.append(os.path.join(latest_dir, file))
                    
                    if result_files:
                        print(f"📁 Encontrados {len(result_files)} archivos result para motivation_dt={motivation_dt}")
                        
                        # Cargar y analizar todos los archivos (una sola pasada)
                        all_data = []
                        all_invasion_data = []
                        all_stats = []
                        
                        for i, result_file in enumerate(sorted(result_files), 1):
                            if len(result_files) > 5:  # Solo mostrar progreso si hay muchos archivos
                                print(f"   Procesando archivo {i}/{len(result_files)}: {os.path.basename(result_file)}")
                            
                            df = load_result_file(result_file, verbose=False)
                            if df is not None:
                                # Almacenar el DataFrame para evitar recarga
                                all_data.append(df)
                                
                                # Analizar invasiones
                                invasion_data = calculate_personal_space_invasions(df, pedestrian_r, groups_start_index)
                                stats = analyze_invasion_statistics(invasion_data)
                                all_invasion_data.append(invasion_data)
                                all_stats.append(stats)
                        
                        if all_stats:
                            # Calcular estadísticas agregadas (mean y std)
                            aggregated_stats = calculate_aggregated_statistics(all_stats)
                            
                            experiments_data[motivation_dt] = {
                                'all_data': all_data,  # Ya cargados, sin duplicar
                                'all_invasion_data': all_invasion_data,
                                'all_stats': all_stats,
                                'aggregated_stats': aggregated_stats,
                                'n_runs': len(all_stats)
                            }
                            
                            # Limpiar memoria si no necesitamos los datos raw después del análisis
                            # (comentado por si se necesitan después)
                            # all_data.clear()
                            print(f"✅ Analizado motivation_dt={motivation_dt}: {len(all_stats)} runs, {aggregated_stats['total_invasions_mean']:.1f}±{aggregated_stats['total_invasions_std']:.1f} invasiones promedio")
            except Exception as e:
                print(f"❌ Error procesando {item}: {e}")
                continue
    
    return experiments_data

def save_results_to_csv(experiments_data, output_dir):
    """
    Guarda los resultados agregados en un archivo CSV.
    """
    if not experiments_data:
        return
    
    # Preparar datos para CSV
    motivation_dts = sorted(experiments_data.keys())
    
    rows = []
    for dt in motivation_dts:
        agg_stats = experiments_data[dt]['aggregated_stats']
        n_runs = experiments_data[dt]['n_runs']
        
        rows.append({
            'motivation_update_dt': dt,
            'n_runs': n_runs,
            'total_invasions_mean': agg_stats.get('total_invasions_mean', 0),
            'total_invasions_std': agg_stats.get('total_invasions_std', 0),
            'total_invasions_min': agg_stats.get('total_invasions_min', 0),
            'total_invasions_max': agg_stats.get('total_invasions_max', 0),
            'total_timesteps_mean': agg_stats.get('total_timesteps_mean', 0),
            'avg_invasions_per_timestep_mean': agg_stats.get('avg_invasions_per_timestep_mean', 0),
            'avg_invasions_per_timestep_std': agg_stats.get('avg_invasions_per_timestep_std', 0),
            'max_invasions_per_timestep_mean': agg_stats.get('max_invasions_per_timestep_mean', 0),
            'min_invasions_per_timestep_mean': agg_stats.get('min_invasions_per_timestep_mean', 0),
            'std_invasions_per_timestep_mean': agg_stats.get('std_invasions_per_timestep_mean', 0),
            'avg_invasion_rate_mean': agg_stats.get('avg_invasion_rate_mean', 0),
            'avg_invasion_rate_std': agg_stats.get('avg_invasion_rate_std', 0),
            'max_invasion_rate_mean': agg_stats.get('max_invasion_rate_mean', 0),
            'avg_pedestrians_per_timestep_mean': agg_stats.get('avg_pedestrians_per_timestep_mean', 0),
            'invasion_frequency_mean': agg_stats.get('invasion_frequency_mean', 0),
            'invasion_frequency_std': agg_stats.get('invasion_frequency_std', 0)
        })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'personal_space_invasion_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en CSV: {csv_path}")

def load_data_from_csv(csv_path):
    """
    Carga datos desde un CSV previamente generado y los convierte al formato
    esperado por plot_comparative_analysis.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Archivo CSV no encontrado: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV cargado: {csv_path}")
        print(f"   {len(df)} experimentos encontrados")
        
        experiments_data = {}
        for _, row in df.iterrows():
            motivation_dt = row['motivation_update_dt']
            n_runs = int(row['n_runs'])
            
            # Construir aggregated_stats desde el CSV
            aggregated_stats = {
                'total_invasions_mean': row['total_invasions_mean'],
                'total_invasions_std': row['total_invasions_std'],
                'total_invasions_min': row.get('total_invasions_min', 0),
                'total_invasions_max': row.get('total_invasions_max', 0),
                'total_timesteps_mean': row.get('total_timesteps_mean', 0),
                'avg_invasions_per_timestep_mean': row['avg_invasions_per_timestep_mean'],
                'avg_invasions_per_timestep_std': row['avg_invasions_per_timestep_std'],
                'max_invasions_per_timestep_mean': row.get('max_invasions_per_timestep_mean', 0),
                'min_invasions_per_timestep_mean': row.get('min_invasions_per_timestep_mean', 0),
                'std_invasions_per_timestep_mean': row.get('std_invasions_per_timestep_mean', 0),
                'avg_invasion_rate_mean': row['avg_invasion_rate_mean'],
                'avg_invasion_rate_std': row['avg_invasion_rate_std'],
                'max_invasion_rate_mean': row.get('max_invasion_rate_mean', 0),
                'avg_pedestrians_per_timestep_mean': row.get('avg_pedestrians_per_timestep_mean', 0),
                'invasion_frequency_mean': row['invasion_frequency_mean'],
                'invasion_frequency_std': row['invasion_frequency_std']
            }
            
            experiments_data[motivation_dt] = {
                'aggregated_stats': aggregated_stats,
                'n_runs': n_runs
            }
        
        return experiments_data
    except Exception as e:
        print(f"❌ Error cargando CSV: {e}")
        return {}

def plot_comparative_analysis(experiments_data, output_dir):
    """
    Genera gráfico de invasiones totales por Motivation DT con barras de error.
    """
    if not experiments_data:
        return

    apply_publication_style()
    
    # Preparar datos para comparación
    motivation_dts = sorted(experiments_data.keys())
    total_invasions_mean = [experiments_data[dt]['aggregated_stats']['total_invasions_mean'] for dt in motivation_dts]
    total_invasions_std = [experiments_data[dt]['aggregated_stats']['total_invasions_std'] for dt in motivation_dts]
    n_runs = [experiments_data[dt]['n_runs'] for dt in motivation_dts]
    
    # Crear figura con el gráfico de invasiones totales con barras de error
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Gráfico: Colisiones totales por Motivation DT con barras de error
    x_pos = range(len(motivation_dts))
    bars = ax.bar(x_pos, total_invasions_mean, color='moccasin', 
                   yerr=total_invasions_std)
    
    ax.set_xlabel('Motivation Update DT (s)')
    ax.set_ylabel('Colisiones Totales')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=90)
    ax.grid(True)
    
    # Calcular el máximo valor incluyendo barras de error para establecer el límite Y
    max_value = max([mean + std for mean, std in zip(total_invasions_mean, total_invasions_std)])
    y_max = max_value * 1.40  # 35% más de espacio para los labels
    ax.set_ylim(0, y_max)
    
    # Agregar anotaciones con mean + std en vertical, asegurándose de que quepan
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, total_invasions_mean, total_invasions_std)):
        height = bar.get_height()
        # Calcular posición del label dentro del área visible
        label_y = min(height + std_val + max_value * 0.03, y_max * 0.97)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom',
                rotation=90)
    
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.25)
    plt.savefig(os.path.join(output_dir, 'personal_space_invasion_comparative_analysis.png'))
    plt.close()
    
    print(f"✅ Gráfico comparativo guardado en: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analizar invasiones del espacio personal en simulaciones de peatones')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt/results',
                       help='Directorio con los resultados de experimentos')
    parser.add_argument('--config', '-c',
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt/config.json',
                       help='Archivo de configuración con PEDESTRIAN_R')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt', 
                       help='Directorio de salida para los resultados')
    parser.add_argument('--single-file', '-f',
                       help='Analizar un solo archivo result_0.csv')
    parser.add_argument('--csv', '--use-csv',
                       help='Usar datos de un CSV previamente generado en lugar de procesar archivos result_*.csv')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    print("🔍 ANÁLISIS DE INVASIONES DEL ESPACIO PERSONAL")
    print("="*60)
    
    # Cargar datos desde CSV o procesar archivos
    if args.csv:
        print(f"\n📂 Cargando datos desde CSV: {args.csv}")
        start_time = time.time()
        experiments_data = load_data_from_csv(args.csv)
        load_time = time.time() - start_time
        print(f"⏱️  Tiempo de carga: {load_time:.2f} segundos")
    else:
        # Análisis de múltiples experimentos
        print(f"\n📂 Cargando datos desde: {args.results_dir}")
        start_time = time.time()
        experiments_data = load_multiple_experiments(args.results_dir, args.config)
        load_time = time.time() - start_time
        print(f"⏱️  Tiempo de carga: {load_time:.2f} segundos")
    
    if not experiments_data:
        print("❌ No se encontraron datos válidos. Terminando.")
        return
    
    print(f"\n✅ Cargados {len(experiments_data)} experimentos exitosamente")
    
    # Guardar resultados en CSV solo si no se usó CSV como entrada
    if not args.csv:
        save_results_to_csv(experiments_data, args.output)
    
    # Generar gráfico comparativo
    plot_comparative_analysis(experiments_data, args.output)

if __name__ == '__main__':
    main()
