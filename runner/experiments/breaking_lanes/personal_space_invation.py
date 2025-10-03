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
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
import warnings
import time
warnings.filterwarnings('ignore')

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
            
        # Obtener posiciones de todos los peatones activos en este timestep
        positions = group[['x', 'y']].values
        pedestrian_ids = group['pedestrian_id'].values
        
        # Calcular matriz de distancias
        distances = pdist(positions)
        distance_matrix = squareform(distances)
        
        # Encontrar pares que están dentro del espacio personal
        invasion_pairs = []
        invasion_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distance_matrix[i, j] < personal_space_radius:
                    invasion_pairs.append((pedestrian_ids[i], pedestrian_ids[j], distance_matrix[i, j]))
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



def print_invasion_report(invasion_data, stats, experiment_name="experiment"):
    """
    Imprime un reporte detallado de las invasiones del espacio personal.
    """
    print("\n" + "="*80)
    print(f"REPORTE DE INVASIONES DEL ESPACIO PERSONAL - {experiment_name.upper()}")
    print("="*80)
    
    print(f"\n📊 ESTADÍSTICAS GENERALES")
    print(f"Total de invasiones: {stats['total_invasions']}")
    print(f"Total de timesteps: {stats['total_timesteps']}")
    print(f"Frecuencia de invasión: {stats['invasion_frequency']:.4f} invasiones/timestep")
    
    print(f"\n📈 ESTADÍSTICAS POR TIMESTEP")
    print(f"Invasiones promedio por timestep: {stats['avg_invasions_per_timestep']:.2f} ± {stats['std_invasions_per_timestep']:.2f}")
    print(f"Máximo de invasiones en un timestep: {stats['max_invasions_per_timestep']}")
    print(f"Mínimo de invasiones en un timestep: {stats['min_invasions_per_timestep']}")
    
    print(f"\n📊 TASAS DE INVASIÓN")
    print(f"Tasa de invasión promedio: {stats['avg_invasion_rate']:.4f}")
    print(f"Tasa de invasión máxima: {stats['max_invasion_rate']:.4f}")
    
    print(f"\n👥 POBLACIÓN")
    print(f"Promedio de peatones por timestep: {stats['avg_pedestrians_per_timestep']:.1f}")
    
    # Análisis temporal
    if len(invasion_data) > 10:
        first_half = invasion_data[:len(invasion_data)//2]
        second_half = invasion_data[len(invasion_data)//2:]
        
        first_half_avg = np.mean([d['invasion_count'] for d in first_half])
        second_half_avg = np.mean([d['invasion_count'] for d in second_half])
        
        print(f"\n⏰ ANÁLISIS TEMPORAL")
        print(f"Primera mitad (promedio): {first_half_avg:.2f} invasiones/timestep")
        print(f"Segunda mitad (promedio): {second_half_avg:.2f} invasiones/timestep")
        print(f"Cambio temporal: {second_half_avg - first_half_avg:+.2f} invasiones/timestep")
    
    print("="*80)

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
        if os.path.isdir(item_path) and item.startswith('cell_size_'):
            try:
                cell_size = float(item.split('cell_size_')[1].split('_implementation_')[0])
                latest_dir = os.path.join(item_path, 'latest')
                
                if os.path.exists(latest_dir):
                    # Buscar todos los archivos result_*.csv
                    result_files = []
                    for file in os.listdir(latest_dir):
                        if file.startswith('result_') and file.endswith('.csv'):
                            result_files.append(os.path.join(latest_dir, file))
                    
                    if result_files:
                        print(f"📁 Encontrados {len(result_files)} archivos result para cell_size={cell_size}")
                        
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
                            
                            experiments_data[cell_size] = {
                                'all_data': all_data,  # Ya cargados, sin duplicar
                                'all_invasion_data': all_invasion_data,
                                'all_stats': all_stats,
                                'aggregated_stats': aggregated_stats,
                                'n_runs': len(all_stats)
                            }
                            
                            # Limpiar memoria si no necesitamos los datos raw después del análisis
                            # (comentado por si se necesitan después)
                            # all_data.clear()
                            print(f"✅ Analizado cell_size={cell_size}: {len(all_stats)} runs, {aggregated_stats['total_invasions_mean']:.1f}±{aggregated_stats['total_invasions_std']:.1f} invasiones promedio")
            except Exception as e:
                print(f"❌ Error procesando {item}: {e}")
                continue
    
    return experiments_data

def plot_comparative_analysis(experiments_data, output_dir):
    """
    Genera gráfico de invasiones totales por Cell Size con barras de error.
    """
    if not experiments_data:
        return
    
    # Preparar datos para comparación
    cell_sizes = sorted(experiments_data.keys())
    total_invasions_mean = [experiments_data[dt]['aggregated_stats']['total_invasions_mean'] for dt in cell_sizes]
    total_invasions_std = [experiments_data[dt]['aggregated_stats']['total_invasions_std'] for dt in cell_sizes]
    n_runs = [experiments_data[dt]['n_runs'] for dt in cell_sizes]
    
    # Crear figura con el gráfico de invasiones totales con barras de error
    plt.figure(figsize=(12, 6))
    
    # Gráfico: Invasiones totales por Cell Size con barras de error
    x_pos = range(len(cell_sizes))
    bars = plt.bar(x_pos, total_invasions_mean, alpha=0.7, color='moccasin', 
                   edgecolor='orange', yerr=total_invasions_std, capsize=5, error_kw={'elinewidth': 2})
    
    plt.xlabel('Cell Size')
    plt.ylabel('Invasiones Totales (Mean ± Std)')
    plt.title('Invasiones Totales por Cell Size')
    plt.xticks(x_pos, [f'{dt:.3f}' for dt in cell_sizes], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Agregar anotaciones con mean + std
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, total_invasions_mean, total_invasions_std)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 5,
                f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'personal_space_invasion_comparative_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico comparativo guardado en: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analizar invasiones del espacio personal en simulaciones de peatones')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/breaking_lanes/results',
                       help='Directorio con los resultados de experimentos')
    parser.add_argument('--config', '-c',
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/breaking_lanes/config.json',
                       help='Archivo de configuración con PEDESTRIAN_R')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/breaking_lanes', 
                       help='Directorio de salida para los resultados')
    parser.add_argument('--single-file', '-f',
                       help='Analizar un solo archivo result_0.csv')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    print("🔍 ANÁLISIS DE INVASIONES DEL ESPACIO PERSONAL")
    print("="*60)
    
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
    
    
    # Generar gráfico comparativo
    plot_comparative_analysis(experiments_data, args.output)
    
    # Imprimir reportes agregados
    print("\n📊 REPORTES AGREGADOS (Mean ± Std)")
    for cell_size in sorted(experiments_data.keys()):
        data = experiments_data[cell_size]
        agg_stats = data['aggregated_stats']
        n_runs = data['n_runs']
        
        print(f"\n--- Cell Size: {cell_size:.3f} (n={n_runs} runs) ---")
        print(f"Total invasiones: {agg_stats['total_invasions_mean']:.1f} ± {agg_stats['total_invasions_std']:.1f}")
        print(f"Promedio/timestep: {agg_stats['avg_invasions_per_timestep_mean']:.2f} ± {agg_stats['avg_invasions_per_timestep_std']:.2f}")
        print(f"Tasa promedio: {agg_stats['avg_invasion_rate_mean']:.4f} ± {agg_stats['avg_invasion_rate_std']:.4f}")
        print(f"Frecuencia: {agg_stats['invasion_frequency_mean']:.4f} ± {agg_stats['invasion_frequency_std']:.4f}")
    
    # Reporte comparativo
    print("\n📊 REPORTE COMPARATIVO (Mean ± Std)")
    print("="*80)
    print(f"{'Cell Size':<15} {'Invasiones Totales':<25} {'Promedio/Timestep':<25} {'Tasa Promedio':<20} {'Runs':<8}")
    print("-" * 95)
    for cell_size in sorted(experiments_data.keys()):
        agg_stats = experiments_data[cell_size]['aggregated_stats']
        n_runs = experiments_data[cell_size]['n_runs']
        print(f"{cell_size:<15.3f} "
              f"{agg_stats['total_invasions_mean']:.1f}±{agg_stats['total_invasions_std']:.1f}{'':<15} "
              f"{agg_stats['avg_invasions_per_timestep_mean']:.2f}±{agg_stats['avg_invasions_per_timestep_std']:.2f}{'':<15} "
              f"{agg_stats['avg_invasion_rate_mean']:.4f}±{agg_stats['avg_invasion_rate_std']:.4f}{'':<10} "
              f"{n_runs:<8}")
    
    print("\n✅ Análisis completado. Resultados guardados en:", args.output)
    print("="*60)

if __name__ == '__main__':
    main()

