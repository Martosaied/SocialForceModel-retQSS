#!/usr/bin/env python3
"""
Script para analizar la suavidad de la función Y(t) de cada partícula para diferentes valores de MOTIVATION_UPDATE_DT.
Un modelo roto produce funciones Y(t) en zigzag, mientras que un modelo
que funciona bien produce funciones más suaves y continuas.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Add the runner directory to the Python path to allow imports from src
script_dir = os.path.dirname(os.path.abspath(__file__))
runner_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if runner_dir not in sys.path:
    sys.path.insert(0, runner_dir)

from src.experiments import apply_publication_style

def load_result_file(file_path):
    """
    Carga un archivo result_0.csv y retorna un DataFrame limpio.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Archivo cargado: {file_path}")
        print(f"   Dimensiones: {df.shape}")
        
        # Convertir formato de columnas por peatón a formato largo
        df_long = convert_to_long_format(df)
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

def analyze_function_smoothness(df):
    """
    Analiza la suavidad de la función Y(t) para cada partícula.
    
    Args:
        df: DataFrame con datos de peatones
    
    Returns:
        dict: Análisis de suavidad por peatón
    """
    df = df.copy()
    df = df.sort_values(['pedestrian_id', 'time'])
    
    # Calcular derivadas de orden superior
    df['dt'] = df.groupby('pedestrian_id')['time'].diff()
    df['y_velocity'] = df.groupby('pedestrian_id')['y'].diff() / df['dt']
    df['y_acceleration'] = df.groupby('pedestrian_id')['y_velocity'].diff() / df['dt']
    df['x_velocity'] = df.groupby('pedestrian_id')['x'].diff() / df['dt']
    df['x_acceleration'] = df.groupby('pedestrian_id')['x_velocity'].diff() / df['dt']
    
    # Análisis de suavidad por peatón
    smoothness_analysis = {}
    
    for ped_id in df['pedestrian_id'].unique():
        ped_data = df[df['pedestrian_id'] == ped_id].copy()
        ped_data = ped_data.reset_index(drop=True)
        
        if len(ped_data) < 5:  # Necesitamos al menos 5 puntos para análisis
            continue
        
        # Calcular métricas de suavidad
        y_values = ped_data['y'].values

        # 1. Número de cambios de signo en la aceleración Y
        y_accel_sign_changes = np.sum(np.diff(np.sign(ped_data['y_acceleration'].dropna())) != 0)

        # 2. Número de cambios de signo en la aceleración X
        x_accel_sign_changes = np.sum(np.diff(np.sign(ped_data['x_acceleration'].dropna())) != 0)

        # 3. Suavidad basada en la regularidad de las derivadas
        y_velocity_std = ped_data['y_velocity'].std()
        y_acceleration_std = ped_data['y_acceleration'].std()
        x_velocity_std = ped_data['x_velocity'].std()
        x_acceleration_std = ped_data['x_acceleration'].std()

        # 4. Puntuación de suavidad compuesta
        # Valores más altos = más suave, valores más bajos = más zigzag
        smoothness_score = calculate_smoothness_score(
            0, 0, y_accel_sign_changes,  # total_variation=0, velocity_sign_changes=0
            y_velocity_std, y_acceleration_std,
            len(ped_data)
        )
        
        smoothness_analysis[ped_id] = {
            'y_accel_sign_changes': y_accel_sign_changes,
            'x_accel_sign_changes': x_accel_sign_changes,
            'y_velocity_std': y_velocity_std,
            'y_acceleration_std': y_acceleration_std,
            'x_velocity_std': x_velocity_std,
            'x_acceleration_std': x_acceleration_std,
            'smoothness_score': smoothness_score,
            'total_steps': len(ped_data),
            'y_range': np.max(y_values) - np.min(y_values)
        }
    
    return smoothness_analysis

def find_motivation_dt_directories(results_dir):
    """
    Encuentra todos los directorios de motivation_dt disponibles.
    """
    motivation_dt_dirs = []
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            if item.startswith('motivation_dt_') and os.path.isdir(os.path.join(results_dir, item)):
                dt_value = item.replace('motivation_dt_', '')
                try:
                    dt_float = float(dt_value)
                    motivation_dt_dirs.append((dt_float, item))
                except ValueError:
                    continue

    # Ordenar por valor de motivation_dt
    motivation_dt_dirs.sort(key=lambda x: x[0])

    # Reamove last one motivation_dt_dir (mismo comportamiento que deltaq)
    if motivation_dt_dirs:
        motivation_dt_dirs.pop()

    return motivation_dt_dirs

def load_all_result_files(motivation_dt_dir_path):
    """
    Carga todos los archivos result_N.csv de un directorio motivation_dt.
    """
    result_files = []
    latest_dir = os.path.join(motivation_dt_dir_path, 'latest')

    if os.path.exists(latest_dir):
        for file in os.listdir(latest_dir):
            if file.startswith('result_') and file.endswith('.csv'):
                result_files.append(os.path.join(latest_dir, file))

    return sorted(result_files)

def calculate_smoothness_score(total_variation, velocity_sign_changes, accel_sign_changes,
                             velocity_std, acceleration_std,
                             total_steps):
    """
    Calcula una puntuación de suavidad compuesta.
    Valores más altos = más suave, valores más bajos = más zigzag.
    """
    # Normalizar métricas
    norm_variation = min(1, total_variation / 10)  # Normalizar variación total
    norm_velocity_changes = min(1, velocity_sign_changes / total_steps)  # Cambios por paso
    norm_accel_changes = min(1, accel_sign_changes / total_steps)  # Cambios por paso
    norm_velocity_std = min(1, velocity_std / 2)  # Normalizar desviación estándar
    norm_accel_std = min(1, acceleration_std / 5)
    
    # Puntuación de suavidad (0 = muy zigzag, 1 = muy suave)
    smoothness = 1.0
    
    # Penalizar variación excesiva
    smoothness -= 0.2 * norm_variation
    
    # Penalizar cambios de signo frecuentes (zigzag)
    smoothness -= 0.3 * norm_velocity_changes
    smoothness -= 0.2 * norm_accel_changes
    
    # Penalizar desviaciones estándar altas
    smoothness -= 0.1 * norm_velocity_std
    smoothness -= 0.1 * norm_accel_std
    
    return max(0, min(1, smoothness))

def load_multiple_motivation_dt_data(results_dir):
    """
    Carga datos de múltiples experimentos de motivation_update_dt, incluyendo todos los result_N.csv.
    """
    motivation_dt_data = {}

    motivation_dt_dirs = find_motivation_dt_directories(results_dir)

    print(f"🔍 Encontrados {len(motivation_dt_dirs)} experimentos de Motivation Update DT:")

    for dt_value, dt_dir in motivation_dt_dirs:
        dt_path = os.path.join(results_dir, dt_dir)
        result_files = load_all_result_files(dt_path)

        if result_files:
            print(f"  - Cargando DT={dt_value} ({len(result_files)} archivos)...")

            all_dfs = []
            all_smoothness = {}
            file_counter = 0

            for result_file in result_files[:2]:
                df = load_result_file(result_file)
                if df is not None:
                    smoothness = analyze_function_smoothness(df)
                    all_dfs.append(df)

                    # Combinar análisis de suavidad con identificador único
                    for ped_id, data in smoothness.items():
                        unique_ped_id = f"{file_counter}_{ped_id}"
                        all_smoothness[unique_ped_id] = data

                    file_counter += 1

            if all_dfs:
                # Combinar todos los DataFrames
                combined_df = pd.concat(all_dfs, ignore_index=True)
                motivation_dt_data[dt_value] = {
                    'df': combined_df,
                    'smoothness': all_smoothness,
                    'file_paths': result_files,
                    'num_files': len(result_files)
                }
                print(f"    ✅ Cargado exitosamente ({len(all_smoothness)} peatones analizados de {len(result_files)} archivos)")
            else:
                print(f"    ❌ Error al cargar datos")
        else:
            print(f"  - ⚠️  No se encontraron archivos result_*.csv para DT={dt_value}")

    return motivation_dt_data

def write_analysis_to_csv(motivation_dt_data, output_file):
    """
    Escribe todos los resultados de análisis de suavidad en un CSV consolidado.
    """
    print(f"📝 Escribiendo resultados consolidados en: {output_file}")

    all_results = []

    for dt_value, data in motivation_dt_data.items():
        smoothness = data['smoothness']
        num_files = data.get('num_files', 1)

        for ped_id, analysis in smoothness.items():
            # Extraer información del archivo si está disponible
            file_info = ped_id.split('_', 1) if '_' in ped_id else ('0', ped_id)
            file_num = file_info[0]
            original_ped_id = file_info[1]

            result_row = {
                'motivation_update_dt': dt_value,
                'file_number': int(file_num),
                'pedestrian_id': original_ped_id,
                'y_accel_sign_changes': analysis['y_accel_sign_changes'],
                'x_accel_sign_changes': analysis['x_accel_sign_changes'],
                'y_velocity_std': analysis['y_velocity_std'],
                'y_acceleration_std': analysis['y_acceleration_std'],
                'x_velocity_std': analysis['x_velocity_std'],
                'x_acceleration_std': analysis['x_acceleration_std'],
                'smoothness_score': analysis['smoothness_score'],
                'total_steps': analysis['total_steps'],
                'y_range': analysis['y_range'],
                'num_files_in_motivation_dt': num_files
            }
            all_results.append(result_row)

    # Crear DataFrame y guardar
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)

    print(f"✅ Guardados {len(all_results)} análisis de suavidad en {output_file}")
    print(f"   - {len(motivation_dt_data)} valores de Motivation Update DT")
    print(f"   - {len(set([r['motivation_update_dt'] for r in all_results]))} experimentos únicos")
    print(f"   - {len(set([r['pedestrian_id'] for r in all_results]))} peatones únicos")

    return results_df

def load_analysis_from_csv(csv_file):
    """
    Carga los resultados de análisis desde un CSV consolidado.
    """
    print(f"📂 Cargando análisis desde: {csv_file}")

    if not os.path.exists(csv_file):
        print(f"❌ No se encontró el archivo: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    print(f"✅ Cargados {len(df)} análisis de suavidad")

    # Reorganizar datos en formato motivation_dt_data
    motivation_dt_data = {}

    for dt in df['motivation_update_dt'].unique():
        dt_df = df[df['motivation_update_dt'] == dt]

        # Reconstruir análisis de suavidad
        smoothness = {}
        for _, row in dt_df.iterrows():
            ped_id = f"{row['file_number']}_{row['pedestrian_id']}"
            smoothness[ped_id] = {
                'y_accel_sign_changes': row['y_accel_sign_changes'],
                'x_accel_sign_changes': row['x_accel_sign_changes'],
                'y_velocity_std': row['y_velocity_std'],
                'y_acceleration_std': row['y_acceleration_std'],
                'x_velocity_std': row['x_velocity_std'],
                'x_acceleration_std': row['x_acceleration_std'],
                'smoothness_score': row['smoothness_score'],
                'total_steps': row['total_steps'],
                'y_range': row['y_range']
            }

        motivation_dt_data[dt] = {
            'smoothness': smoothness,
            'num_files': dt_df['num_files_in_motivation_dt'].iloc[0] if len(dt_df) > 0 else 1
        }

    print(f"✅ Reorganizados datos para {len(motivation_dt_data)} experimentos de Motivation Update DT")
    return motivation_dt_data

def plot_smoothness_components_multi(motivation_dt_data, output_dir):
    """
    Genera gráfico de componentes de suavidad para múltiples motivation_dt con promedio y desviación estándar.
    """
    apply_publication_style()

    fig = plt.figure(figsize=(16, 12))

    # Calcular estadísticas de componentes para cada motivation_dt
    dt_values = []
    components_stats = {
        'Cambios en el signo de la aceleración Y': {'mean': [], 'std': []},
        'Cambios en el signo de la aceleración X': {'mean': [], 'std': []},
    }

    smoothness_stats = {'mean': [], 'std': []}
    for dt, data in motivation_dt_data.items():
        smoothness = data['smoothness']
        if smoothness:
            dt_values.append(dt)

            # Calcular estadísticas para cada componente
            y_accel_changes = [d['y_accel_sign_changes'] for d in smoothness.values()]
            x_accel_changes = [d['x_accel_sign_changes'] for d in smoothness.values()]

            components_stats['Cambios en el signo de la aceleración Y']['mean'].append(np.mean(y_accel_changes))
            components_stats['Cambios en el signo de la aceleración Y']['std'].append(np.std(y_accel_changes))

            components_stats['Cambios en el signo de la aceleración X']['mean'].append(np.mean(x_accel_changes))
            components_stats['Cambios en el signo de la aceleración X']['std'].append(np.std(x_accel_changes))

            scores = [d['smoothness_score'] for d in smoothness.values()]
            smoothness_stats['mean'].append(np.mean(scores))
            smoothness_stats['std'].append(np.std(scores))

    if dt_values:
        x = np.arange(len(dt_values))
        markers = ['o', 's']  # Different markers for clarity

        # Gráfico: Líneas con marcadores y barras de error
        for i, (component, stats) in enumerate(components_stats.items()):
            plt.errorbar(x, stats['mean'], yerr=stats['std'],
                        marker=markers[i],
                        label=component)

        plt.xlabel('Motivation Update DT (s)', fontsize=32)
        plt.ylabel('Número de Cambios de Signo', fontsize=32)
        plt.xticks(x, [f'{dt:.3f}' for dt in dt_values], rotation=45, ha='right', fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(loc='upper left')
        plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.92)
    plt.savefig(os.path.join(output_dir, 'smoothness_components_multi_motivation_dt.png'))
    plt.close()

    # Generate LaTeX table with exact values
    if dt_values:
        latex_path = os.path.join(output_dir, 'compare_motivation_dt_stability_results.tex')
        with open(latex_path, 'w') as f:
            f.write(r'\begin{table}[htbp]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Análisis de suavidad por Motivation Update DT: Cambios de signo en aceleración y puntuación de suavidad.}' + '\n')
            f.write(r'\label{tab:compare_motivation_dt_stability}' + '\n')
            f.write(r'\begin{tabular}{cccc}' + '\n')
            f.write(r'\hline' + '\n')
            f.write(r'Motivation DT (s) & Cambios signo acel. Y (mean $\pm$ std) & Cambios signo acel. X (mean $\pm$ std) & Suavidad (mean $\pm$ std) \\' + '\n')
            f.write(r'\hline' + '\n')
            for i, dt in enumerate(dt_values):
                y_m = components_stats['Cambios en el signo de la aceleración Y']['mean'][i]
                y_s = components_stats['Cambios en el signo de la aceleración Y']['std'][i]
                x_m = components_stats['Cambios en el signo de la aceleración X']['mean'][i]
                x_s = components_stats['Cambios en el signo de la aceleración X']['std'][i]
                sm_m = smoothness_stats['mean'][i]
                sm_s = smoothness_stats['std'][i]
                f.write(f'{dt:.3f} & {y_m:.4f} $\\pm$ {y_s:.4f} & {x_m:.4f} $\\pm$ {x_s:.4f} & {sm_m:.4f} $\\pm$ {sm_s:.4f} \\\\\n')
            f.write(r'\hline' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        print(f"LaTeX table saved to: {latex_path}")

def process_data_and_save_csv(results_dir, output_dir):
    """
    Procesa todos los datos y guarda los resultados en un CSV consolidado.
    """
    print("🔍 PROCESAMIENTO DE DATOS - ANÁLISIS DE SUAVIDAD")
    print("="*60)

    # Cargar datos de múltiples motivation_dt
    print(f"\n📂 Cargando datos desde: {results_dir}")
    motivation_dt_data = load_multiple_motivation_dt_data(results_dir)

    if not motivation_dt_data:
        print("❌ No se encontraron datos válidos de Motivation Update DT. Terminando.")
        return None

    print(f"\n✅ Cargados {len(motivation_dt_data)} experimentos de Motivation Update DT exitosamente")

    # Escribir resultados consolidados en CSV
    csv_file = os.path.join(output_dir, 'smoothness_analysis_consolidated.csv')
    write_analysis_to_csv(motivation_dt_data, csv_file)

    return csv_file

def generate_plots_from_csv(csv_file, output_dir):
    """
    Genera todos los gráficos a partir del CSV consolidado.
    """
    print("\n📊 GENERACIÓN DE GRÁFICOS DESDE CSV CONSOLIDADO")
    print("="*60)

    # Cargar datos desde CSV
    motivation_dt_data = load_analysis_from_csv(csv_file)

    if not motivation_dt_data:
        print("❌ No se pudieron cargar los datos desde el CSV. Terminando.")
        return

    # Generar gráficos multi-motivation_dt
    print("\n📊 Generando gráficos de análisis de suavidad multi-Motivation DT...")

    print("  - Generando gráfico de componentes de suavidad multi-Motivation DT...")
    plot_smoothness_components_multi(motivation_dt_data, output_dir)

    # Imprimir reporte resumido
    print("\n📊 REPORTE RESUMIDO DE SUAVIDAD POR MOTIVATION UPDATE DT")
    print("="*60)

    for dt in sorted(motivation_dt_data.keys()):
        data = motivation_dt_data[dt]
        smoothness = data['smoothness']
        if smoothness:
            scores = [d['smoothness_score'] for d in smoothness.values()]
            avg_smooth = np.mean(scores)
            std_smooth = np.std(scores)
            print(f"DT={dt:.3f}: Suavidad={avg_smooth:.3f}±{std_smooth:.3f} (n={len(scores)})")

    # Encontrar el mejor y peor Motivation DT
    best_dt = max(motivation_dt_data.keys(),
                  key=lambda x: np.mean([d['smoothness_score'] for d in motivation_dt_data[x]['smoothness'].values()])
                  if motivation_dt_data[x]['smoothness'] else 0)
    worst_dt = min(motivation_dt_data.keys(),
                   key=lambda x: np.mean([d['smoothness_score'] for d in motivation_dt_data[x]['smoothness'].values()])
                   if motivation_dt_data[x]['smoothness'] else 1)

    print(f"\n🏆 MEJOR Motivation DT: {best_dt:.3f} (más suave)")
    print(f"⚠️  PEOR Motivation DT: {worst_dt:.3f} (más zigzag)")

    print(f"\n✅ Gráficos generados y guardados en: {output_dir}")
    print("📁 Archivos generados:")
    print("  - smoothness_analysis_consolidated.csv")
    print("  - smoothness_components_multi_motivation_dt.png")
    print("  - compare_motivation_dt_stability_results.tex")

def main():
    parser = argparse.ArgumentParser(description='Analizar suavidad de la función Y(t) para múltiples Motivation Update DT')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt/results',
                       help='Directorio con los resultados de experimentos Motivation Update DT')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt', 
                       help='Directorio de salida para los resultados')
    parser.add_argument('--csv-file', '-c',
                       help='Archivo CSV consolidado existente (si se proporciona, solo genera gráficos)')
    parser.add_argument('--process-only', action='store_true',
                       help='Solo procesar datos y guardar CSV, no generar gráficos')
    parser.add_argument('--plots-only', action='store_true',
                       help='Solo generar gráficos desde CSV existente')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    if args.plots_only and args.csv_file:
        # Solo generar gráficos desde CSV existente
        generate_plots_from_csv(args.csv_file, args.output)
    elif args.process_only:
        # Solo procesar datos y guardar CSV
        csv_file = process_data_and_save_csv(args.results_dir, args.output)
        if csv_file:
            print(f"\n✅ Procesamiento completado. CSV guardado en: {csv_file}")
    else:
        # Proceso completo: procesar datos y generar gráficos
        if args.csv_file and os.path.exists(args.csv_file):
            # Usar CSV existente
            print("📂 Usando CSV consolidado existente...")
            generate_plots_from_csv(args.csv_file, args.output)
        else:
            # Procesar datos y generar gráficos
            csv_file = process_data_and_save_csv(args.results_dir, args.output)
            if csv_file:
                generate_plots_from_csv(csv_file, args.output)

if __name__ == '__main__':
    main()
