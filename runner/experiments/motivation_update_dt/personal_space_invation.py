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
import re
from scipy.spatial import cKDTree
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
        # Para este análisis solo necesitamos (time, id, x, y); evitar columnas extra acelera mucho.
        df_long = convert_to_long_format(df, minimal=True)
        if verbose:
            print(f"   Formato largo: {df_long.shape}")
        return df_long
    except Exception as e:
        print(f"❌ Error cargando {file_path}: {e}")
        return None

_PX_RE = re.compile(r"^PX\[(\d+)\]$")
_PY_RE = re.compile(r"^PY\[(\d+)\]$")
_VX_RE = re.compile(r"^VX\[(\d+)\]$")
_VY_RE = re.compile(r"^VY\[(\d+)\]$")
_PS_RE = re.compile(r"^PS\[(\d+)\]$")

def _extract_ids(df_columns, rx):
    ids = []
    for c in df_columns:
        m = rx.match(c)
        if m:
            ids.append(int(m.group(1)))
    return set(ids)

def convert_to_long_format(df, minimal=False):
    """
    Convierte el formato de columnas por peatón a formato largo (long format).
    """
    if 'time' not in df.columns:
        return pd.DataFrame(columns=['time', 'pedestrian_id', 'x', 'y'])

    cols = list(df.columns)
    px_ids = _extract_ids(cols, _PX_RE)
    py_ids = _extract_ids(cols, _PY_RE)

    if minimal:
        ped_ids = sorted(px_ids & py_ids)
    else:
        vx_ids = _extract_ids(cols, _VX_RE)
        vy_ids = _extract_ids(cols, _VY_RE)
        ps_ids = _extract_ids(cols, _PS_RE)
        ped_ids = sorted(px_ids & py_ids & vx_ids & vy_ids & ps_ids)

    if not ped_ids:
        if minimal:
            return pd.DataFrame(columns=['time', 'pedestrian_id', 'x', 'y'])
        return pd.DataFrame(columns=['time', 'pedestrian_id', 'x', 'y', 'velocity_x', 'velocity_y', 'state'])

    px_cols = [f'PX[{i}]' for i in ped_ids]
    py_cols = [f'PY[{i}]' for i in ped_ids]

    time = df['time'].to_numpy(copy=False)
    px = df[px_cols].to_numpy(copy=False)
    py = df[py_cols].to_numpy(copy=False)

    n_t = time.shape[0]
    n_p = len(ped_ids)
    out = {
        'time': np.repeat(time, n_p),
        'pedestrian_id': np.tile(np.asarray(ped_ids, dtype=np.int32), n_t),
        'x': px.reshape(-1),
        'y': py.reshape(-1),
    }

    if not minimal:
        vx_cols = [f'VX[{i}]' for i in ped_ids]
        vy_cols = [f'VY[{i}]' for i in ped_ids]
        ps_cols = [f'PS[{i}]' for i in ped_ids]
        out['velocity_x'] = df[vx_cols].to_numpy(copy=False).reshape(-1)
        out['velocity_y'] = df[vy_cols].to_numpy(copy=False).reshape(-1)
        out['state'] = df[ps_cols].to_numpy(copy=False).reshape(-1)

    df_long = pd.DataFrame(out)

    # Solo incluir si el peatón está activo (posición válida != 0 y no NaN)
    # Nota: se mantiene la misma definición que antes (x,y no nulos y distintos de 0).
    mask = (
        df_long['x'].notna()
        & df_long['y'].notna()
        & (df_long['x'] != 0)
        & (df_long['y'] != 0)
    )
    return df_long.loc[mask].reset_index(drop=True)

def calculate_personal_space_invasions(df, pedestrian_r, groups_start_index, collect_pairs=False):
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

    # time_threshold = groups_start_index / 10
    # if 'time' in df.columns:
        # df = df[df['time'] >= time_threshold]
    
    # Agrupar por timestep
    for time, group in df.groupby('time', sort=False):
        if len(group) < 2:  # Necesitamos al menos 2 peatones
            continue
            
        # Obtener posiciones de todos los peatones activos en este timestep
        positions = group[['x', 'y']].values
        pedestrian_ids = group['pedestrian_id'].values

        # Umbral más estricto: 1.5 * radio en lugar de 2 * radio
        invasion_threshold = personal_space_radius * 1.99
        
        # Encontrar pares que están dentro del espacio personal (rápido con KDTree)
        tree = cKDTree(positions)
        pairs_idx = tree.query_pairs(r=invasion_threshold, output_type='ndarray')
        invasion_count = int(pairs_idx.shape[0])

        invasion_pairs = []
        if collect_pairs and invasion_count > 0:
            # Calcular distancias solo para los pares detectados (evita O(n^2))
            diffs = positions[pairs_idx[:, 0]] - positions[pairs_idx[:, 1]]
            dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
            a = pedestrian_ids[pairs_idx[:, 0]]
            b = pedestrian_ids[pairs_idx[:, 1]]
            invasion_pairs = list(zip(a.tolist(), b.tolist(), dists.tolist()))
        
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
                        
                        # Cargar y analizar todos los archivos
                        all_stats = []
                        
                        # Limitar runs si hay demasiados; slice después de ordenar.
                        for i, result_file in enumerate(sorted(result_files[:5]), 1):
                            if len(result_files) > 5:  # Solo mostrar progreso si hay muchos archivos
                                print(f"   Procesando archivo {i}/{len(result_files)}: {os.path.basename(result_file)}")
                            
                            df = load_result_file(result_file, verbose=False)
                            if df is not None:
                                # Analizar invasiones
                                invasion_data = calculate_personal_space_invasions(df, pedestrian_r, groups_start_index, collect_pairs=False)
                                stats = analyze_invasion_statistics(invasion_data)  # invasion_pairs no se usan; evitamos generarlas
                                all_stats.append(stats)
                        
                        if all_stats:
                            # Calcular estadísticas agregadas (mean y std)
                            aggregated_stats = calculate_aggregated_statistics(all_stats)
                            
                            experiments_data[motivation_dt] = {
                                'all_stats': all_stats,
                                'aggregated_stats': aggregated_stats,
                                'n_runs': len(all_stats)
                            }
                            
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
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Gráfico: Colisiones totales por Motivation DT con barras de error
    x_pos = range(len(motivation_dts))
    bars = ax.bar(x_pos, total_invasions_mean, color='lightsalmon', 
                   yerr=total_invasions_std)
    
    ax.set_xlabel('Motivation Update DT (s)')
    ax.set_ylabel('Colisiones Totales')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{dt:.3f}' for dt in motivation_dts], rotation=45)
    ax.grid(True)
    
    # Calcular el máximo valor incluyendo barras de error para establecer el límite Y
    max_value = max([mean + std for mean, std in zip(total_invasions_mean, total_invasions_std)])
    y_max = max_value * 1.40
    ax.set_ylim(0, y_max)
    
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.25)
    plt.savefig(os.path.join(output_dir, 'personal_space_invasion_comparative_analysis.png'))
    plt.close()
    
    print(f"✅ Gráfico comparativo guardado en: {output_dir}")

    # Generate LaTeX table with exact values
    latex_path = os.path.join(output_dir, 'personal_space_invasion_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Invasiones del espacio personal por Motivation Update DT.}' + '\n')
        f.write(r'\label{tab:personal_space_invasion_motivation_dt}' + '\n')
        f.write(r'\begin{tabular}{ccccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'Motivation DT (s) & Invasiones Totales (mean $\pm$ std) & Promedio/Timestep (mean $\pm$ std) & Tasa Promedio (mean $\pm$ std) & Runs \\' + '\n')
        f.write(r'\hline' + '\n')
        for dt in motivation_dts:
            agg = experiments_data[dt]['aggregated_stats']
            f.write(f'{dt:.3f} & {agg["total_invasions_mean"]:.2f} $\\pm$ {agg["total_invasions_std"]:.2f} & ')
            f.write(f'{agg["avg_invasions_per_timestep_mean"]:.2f} $\\pm$ {agg["avg_invasions_per_timestep_std"]:.2f} & ')
            f.write(f'{agg["avg_invasion_rate_mean"]:.4f} $\\pm$ {agg["avg_invasion_rate_std"]:.4f} & ')
            f.write(f'{experiments_data[dt]["n_runs"]} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")

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
    
    # Caso: analizar un solo archivo
    if args.single_file:
        pedestrian_r, groups_start_index = load_config(args.config)
        if pedestrian_r is None:
            return
        start_time = time.time()
        df = load_result_file(args.single_file, verbose=True)
        if df is None:
            return
        invasion_data = calculate_personal_space_invasions(df, pedestrian_r, groups_start_index, collect_pairs=False)
        stats = analyze_invasion_statistics(invasion_data)
        elapsed = time.time() - start_time
        print(f"\n⏱️  Tiempo total (single file): {elapsed:.2f} segundos")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

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
