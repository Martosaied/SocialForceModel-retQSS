#!/usr/bin/env python3
"""
Script para analizar la suavidad de la función Y(t) de cada partícula.
Un modelo roto produce funciones Y(t) en zigzag, mientras que un modelo
que funciona bien produce funciones más suaves y continuas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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
        time_values = ped_data['time'].values
        
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

def find_deltaq_directories(results_dir):
    """
    Encuentra todos los directorios de deltaq disponibles.
    """
    deltaq_dirs = []
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            if item.startswith('deltaq_') and os.path.isdir(os.path.join(results_dir, item)):
                deltaq_value = item.replace('deltaq_', '')
                try:
                    deltaq_float = float(deltaq_value)
                    deltaq_dirs.append((deltaq_float, item))
                except ValueError:
                    continue
    
    # Ordenar por valor de deltaq
    deltaq_dirs.sort(key=lambda x: x[0])

    # Reamove last one deltaq_dir
    deltaq_dirs.pop()
    return deltaq_dirs

def load_all_result_files(deltaq_dir_path):
    """
    Carga todos los archivos result_N.csv de un directorio deltaq.
    """
    result_files = []
    latest_dir = os.path.join(deltaq_dir_path, 'latest')
    
    if os.path.exists(latest_dir):
        for file in os.listdir(latest_dir):
            if file.startswith('result_') and file.endswith('.csv'):
                result_files.append(os.path.join(latest_dir, file))
    
    return sorted(result_files)

def load_multiple_deltaq_data(results_dir):
    """
    Carga datos de múltiples experimentos de deltaq, incluyendo todos los result_N.csv.
    """
    deltaq_data = {}
    deltaq_dirs = find_deltaq_directories(results_dir)
    
    print(f"🔍 Encontrados {len(deltaq_dirs)} experimentos de DeltaQ:")
    
    for deltaq_value, deltaq_dir in deltaq_dirs:
        deltaq_path = os.path.join(results_dir, deltaq_dir)
        result_files = load_all_result_files(deltaq_path)
        
        if result_files:
            print(f"  - Cargando ΔQ={deltaq_value} ({len(result_files)} archivos)...")
            
            all_dfs = []
            all_smoothness = {}
            file_counter = 0
            
            for result_file in result_files:
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
                deltaq_data[deltaq_value] = {
                    'df': combined_df,
                    'smoothness': all_smoothness,
                    'file_paths': result_files,
                    'num_files': len(result_files)
                }
                print(f"    ✅ Cargado exitosamente ({len(all_smoothness)} peatones analizados de {len(result_files)} archivos)")
            else:
                print(f"    ❌ Error al cargar datos")
        else:
            print(f"  - ⚠️  No se encontraron archivos result_*.csv para ΔQ={deltaq_value}")
    
    return deltaq_data

def write_analysis_to_csv(deltaq_data, output_file):
    """
    Escribe todos los resultados de análisis de suavidad en un CSV consolidado.
    """
    print(f"📝 Escribiendo resultados consolidados en: {output_file}")
    
    all_results = []
    
    for deltaq_value, data in deltaq_data.items():
        smoothness = data['smoothness']
        num_files = data.get('num_files', 1)
        
        for ped_id, analysis in smoothness.items():
            # Extraer información del archivo si está disponible
            file_info = ped_id.split('_', 1) if '_' in ped_id else ('0', ped_id)
            file_num = file_info[0]
            original_ped_id = file_info[1]
            
            result_row = {
                'deltaq': deltaq_value,
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
                'num_files_in_deltaq': num_files
            }
            all_results.append(result_row)
    
    # Crear DataFrame y guardar
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Guardados {len(all_results)} análisis de suavidad en {output_file}")
    print(f"   - {len(deltaq_data)} valores de DeltaQ")
    print(f"   - {len(set([r['deltaq'] for r in all_results]))} experimentos únicos")
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
    
    # Reorganizar datos en formato deltaq_data
    deltaq_data = {}
    
    for deltaq in df['deltaq'].unique():
        deltaq_df = df[df['deltaq'] == deltaq]
        
        # Reconstruir análisis de suavidad
        smoothness = {}
        for _, row in deltaq_df.iterrows():
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
        
        deltaq_data[deltaq] = {
            'smoothness': smoothness,
            'num_files': deltaq_df['num_files_in_deltaq'].iloc[0] if len(deltaq_df) > 0 else 1
        }
    
    print(f"✅ Reorganizados datos para {len(deltaq_data)} experimentos de DeltaQ")
    return deltaq_data

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

def plot_y_functions_multi(deltaq_data, output_dir):
    """
    Genera gráfico de las funciones Y(t) de muestra para múltiples deltaq.
    """
    plt.figure(figsize=(14, 10))
    plt.title('Funciones Y(t) de Muestra - Análisis de Suavidad por DeltaQ', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(deltaq_data)))
    
    for i, (deltaq, data) in enumerate(deltaq_data.items()):
        df = data['df']
        smoothness = data['smoothness']
        
        if smoothness:
            scores = [(pid, data['smoothness_score']) for pid, data in smoothness.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            sample_pedestrians = [pid for pid, _ in scores[:2]]  # Los más suaves
        else:
            sample_pedestrians = []
        
        # Plotear funciones Y(t)
        for j, ped_id in enumerate(sample_pedestrians):
            ped_data = df[df['pedestrian_id'] == ped_id].sort_values('time')
            if len(ped_data) > 0:
                alpha = 0.8 if j == 0 else 0.5
                linewidth = 2 if j == 0 else 1
                label = f'ΔQ=1e{float(deltaq)}' if j == 0 else ""
                plt.plot(ped_data['time'], ped_data['y'], color=colors[i], 
                        alpha=alpha, linewidth=linewidth, label=label)
    
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición Y (m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'y_functions_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_distribution_multi(deltaq_data, output_dir):
    """
    Genera gráfico de distribución de puntuaciones de suavidad para múltiples deltaq con estadísticas detalladas.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Distribución de Puntuaciones de Suavidad por DeltaQ - Análisis Estadístico', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(deltaq_data)))
    
    # Gráfico 1: Histogramas superpuestos
    ax1.set_title('Distribución de Puntuaciones de Suavidad', fontsize=12, fontweight='bold')
    for i, (deltaq, data) in enumerate(deltaq_data.items()):
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            ax1.hist(scores, bins=15, alpha=0.6, color=colors[i], 
                    label=f'ΔQ=1e{float(deltaq)} (n={len(scores)})', density=True)
    
    ax1.set_xlabel('Puntuación de Suavidad (0=Zigzag, 1=Suave)')
    ax1.set_ylabel('Densidad')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Box plots
    ax2.set_title('Distribución de Suavidad - Box Plots', fontsize=12, fontweight='bold')
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, (deltaq, data) in enumerate(deltaq_data.items()):
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            box_data.append(scores)
            box_labels.append(f'1e{float(deltaq)}')
            box_colors.append(colors[i])
    
    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_xlabel('DeltaQ')
    ax2.set_ylabel('Puntuación de Suavidad')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Gráfico 3: Violin plots
    ax3.set_title('Distribución de Suavidad - Violin Plots', fontsize=12, fontweight='bold')
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for i, (deltaq, data) in enumerate(deltaq_data.items()):
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            violin_data.append(scores)
            violin_labels.append(f'1e{float(deltaq)}')
            violin_colors.append(colors[i])
    
    if violin_data:
        parts = ax3.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i])
            pc.set_alpha(0.7)
    
    ax3.set_xlabel('DeltaQ')
    ax3.set_ylabel('Puntuación de Suavidad')
    ax3.set_xticks(range(len(violin_labels)))
    ax3.set_xticklabels(violin_labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Estadísticas resumidas
    ax4.axis('off')
    ax4.set_title('Estadísticas de Distribución', fontsize=12, fontweight='bold')
    
    # Calcular estadísticas para cada deltaq
    stats_data = []
    for deltaq, data in deltaq_data.items():
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            stats_row = [
                f'1e{float(deltaq)}',
                f'{len(scores)}',
                f'{np.mean(scores):.3f}',
                f'{np.std(scores):.3f}',
                f'{np.min(scores):.3f}',
                f'{np.max(scores):.3f}',
                f'{np.median(scores):.3f}',
                f'{np.percentile(scores, 25):.3f}',
                f'{np.percentile(scores, 75):.3f}'
            ]
            stats_data.append(stats_row)
    
    headers = ['DeltaQ', 'N', 'Media', 'Desv.Est.', 'Mín', 'Máx', 'Mediana', 'Q1', 'Q3']
    
    table = ax4.table(cellText=stats_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Colorear filas según suavidad promedio
    for i, row in enumerate(stats_data):
        if len(row) > 2:
            mean_score = float(row[2])  # Media
            color_intensity = mean_score  # 0-1 scale
            color = plt.cm.RdYlGn(color_intensity)
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_distribution_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_components_multi(deltaq_data, output_dir):
    """
    Genera gráfico de componentes de suavidad para múltiples deltaq con promedio y desviación estándar.
    """
    plt.figure(figsize=(12, 8))
    plt.title('Componentes de Suavidad por DeltaQ - Análisis Estadístico', fontsize=16, fontweight='bold')
    
    # Calcular estadísticas de componentes para cada deltaq
    deltaq_values = []
    components_stats = {
        'Cambios en el signo de la aceleración Y': {'mean': [], 'std': []},
        'Cambios en el signo de la aceleración X': {'mean': [], 'std': []},
    }
    
    for deltaq, data in deltaq_data.items():
        smoothness = data['smoothness']
        if smoothness:
            deltaq_values.append(deltaq)
            
            # Calcular estadísticas para cada componente
            y_accel_changes = [d['y_accel_sign_changes'] for d in smoothness.values()]
            x_accel_changes = [d['x_accel_sign_changes'] for d in smoothness.values()]
            
            components_stats['Cambios en el signo de la aceleración Y']['mean'].append(np.mean(y_accel_changes))
            components_stats['Cambios en el signo de la aceleración Y']['std'].append(np.std(y_accel_changes))
            
            components_stats['Cambios en el signo de la aceleración X']['mean'].append(np.mean(x_accel_changes))
            components_stats['Cambios en el signo de la aceleración X']['std'].append(np.std(x_accel_changes))
    
    if deltaq_values:
        x = np.arange(len(deltaq_values))
        width = 0.35
        colors = ['#1f77b4', '#ff7f0e']
        
        # Gráfico: Promedios con barras de error (desviación estándar)
        for i, (component, stats) in enumerate(components_stats.items()):
            plt.bar(x + i * width, stats['mean'], width, 
                   yerr=stats['std'], capsize=5,
                   label=component, alpha=0.8, color=colors[i])
        
        plt.xlabel('DeltaQ')
        plt.ylabel('Valor Promedio ± Desv. Est.')
        plt.xticks(x + width, [f'1e{float(deltaq)}' for deltaq in deltaq_values], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_components_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_summary_multi(deltaq_data, output_dir):
    """
    Genera gráfico de resumen estadístico para múltiples deltaq con información detallada.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Resumen de Análisis de Suavidad por DeltaQ - Análisis Estadístico Completo', fontsize=16, fontweight='bold')
    
    # Calcular estadísticas para cada deltaq
    deltaq_values = []
    avg_smoothness = []
    std_smoothness = []
    n_pedestrians = []
    
    for deltaq, data in deltaq_data.items():
        smoothness = data['smoothness']
        if smoothness:
            scores = [d['smoothness_score'] for d in smoothness.values()]
            deltaq_values.append(deltaq)
            avg_smoothness.append(np.mean(scores))
            std_smoothness.append(np.std(scores))
            n_pedestrians.append(len(scores))
    
    if deltaq_values:
        # Gráfico 1: Suavidad promedio con barras de error
        ax1.errorbar(deltaq_values, avg_smoothness, yerr=std_smoothness, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax1.set_xlabel('DeltaQ')
        ax1.set_ylabel('Suavidad Promedio ± Desv. Est.')
        ax1.set_title('Suavidad Promedio por DeltaQ')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Formatear etiquetas del eje X
        deltaq_labels = [f'1e{float(deltaq)}' for deltaq in deltaq_values]
        ax1.set_xticks(deltaq_values)
        ax1.set_xticklabels(deltaq_labels, rotation=45)
        
        # Colorear puntos según suavidad
        colors = plt.cm.RdYlGn([s for s in avg_smoothness])
        for i, (x, y) in enumerate(zip(deltaq_values, avg_smoothness)):
            ax1.scatter(x, y, c=[colors[i]], s=100, zorder=5)
        
        # Gráfico 2: Desviación estándar de suavidad
        ax2.bar(range(len(deltaq_values)), std_smoothness, 
               color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_xlabel('DeltaQ')
        ax2.set_ylabel('Desviación Estándar de Suavidad')
        ax2.set_title('Variabilidad de Suavidad por DeltaQ')
        ax2.set_xticks(range(len(deltaq_values)))
        ax2.set_xticklabels(deltaq_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Número de peatones analizados
        ax3.bar(range(len(deltaq_values)), n_pedestrians, 
               color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax3.set_xlabel('DeltaQ')
        ax3.set_ylabel('Número de Peatones')
        ax3.set_title('Número de Peatones Analizados por DeltaQ')
        ax3.set_xticks(range(len(deltaq_values)))
        ax3.set_xticklabels(deltaq_labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Tabla de estadísticas detalladas
        ax4.axis('off')
        ax4.set_title('Estadísticas Detalladas', fontsize=12, fontweight='bold')
        
        # Crear tabla de datos
        table_data = []
        for i, deltaq in enumerate(deltaq_values):
            scores = [d['smoothness_score'] for d in deltaq_data[deltaq]['smoothness'].values()]
            # Calcular estadísticas adicionales
            min_score = np.min(scores)
            max_score = np.max(scores)
            median_score = np.median(scores)
            
            table_data.append([
                f'1e{float(deltaq)}',
                f'{avg_smoothness[i]:.3f}',
                f'{std_smoothness[i]:.3f}',
                f'{min_score:.3f}',
                f'{max_score:.3f}',
                f'{median_score:.3f}',
                f'{n_pedestrians[i]}'
            ])
        
        headers = ['DeltaQ', 'Promedio', 'Desv.Est.', 'Mínimo', 'Máximo', 'Mediana', 'N']
        
        table = ax4.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Colorear filas según suavidad promedio
        for i in range(len(table_data)):
            color = colors[i] if i < len(colors) else 'white'
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_summary_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_smoothness_report(smoothness1, smoothness2, file1_name, file2_name):
    """
    Imprime un reporte detallado del análisis de suavidad.
    """
    print("\n" + "="*80)
    print("REPORTE DE SUAVIDAD DE LA FUNCIÓN Y(t)")
    print("="*80)
    
    print(f"\n📁 ARCHIVOS ANALIZADOS:")
    print(f"   File 1: {file1_name}")
    print(f"   File 2: {file2_name}")
    
    print(f"\n🔍 ANÁLISIS DE SUAVIDAD:")
    print(f"{'Métrica':<25} {'File 1':<15} {'File 2':<15} {'Diferencia':<15}")
    print("-" * 75)
    
    # Suavidad promedio
    if smoothness1:
        avg_smooth1 = np.mean([data['smoothness_score'] for data in smoothness1.values()])
        std_smooth1 = np.std([data['smoothness_score'] for data in smoothness1.values()])
    else:
        avg_smooth1 = std_smooth1 = 0
    
    if smoothness2:
        avg_smooth2 = np.mean([data['smoothness_score'] for data in smoothness2.values()])
        std_smooth2 = np.std([data['smoothness_score'] for data in smoothness2.values()])
    else:
        avg_smooth2 = std_smooth2 = 0
    
    print(f"{'Suavidad Promedio':<25} {avg_smooth1:<15.3f} {avg_smooth2:<15.3f} {abs(avg_smooth1-avg_smooth2):<15.3f}")
    print(f"{'Desv. Est. Suavidad':<25} {std_smooth1:<15.3f} {std_smooth2:<15.3f} {abs(std_smooth1-std_smooth2):<15.3f}")
    
    # Componentes de suavidad
    if smoothness1 and smoothness2:
        comp1 = {
            'Variación Total': np.mean([data['total_variation'] for data in smoothness1.values()]),
            'Cambios Velocidad': np.mean([data['velocity_sign_changes'] for data in smoothness1.values()]),
            'Cambios aceleración': np.mean([data['accel_sign_changes'] for data in smoothness1.values()]),
        }
        
        comp2 = {
            'Variación Total': np.mean([data['total_variation'] for data in smoothness2.values()]),
            'Cambios Velocidad': np.mean([data['velocity_sign_changes'] for data in smoothness2.values()]),
            'Cambios aceleración': np.mean([data['accel_sign_changes'] for data in smoothness2.values()]),
        }
        
        for metric in comp1:
            print(f"{metric:<25} {comp1[metric]:<15.3f} {comp2[metric]:<15.3f} {abs(comp1[metric]-comp2[metric]):<15.3f}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    if avg_smooth1 > avg_smooth2:
        print(f"   🏆 File 1 tiene funciones Y(t) MÁS SUAVES")
        print(f"   📊 Diferencia: {avg_smooth1 - avg_smooth2:.3f} puntos de suavidad")
    elif avg_smooth2 > avg_smooth1:
        print(f"   🏆 File 2 tiene funciones Y(t) MÁS SUAVES")
        print(f"   📊 Diferencia: {avg_smooth2 - avg_smooth1:.3f} puntos de suavidad")
    else:
        print(f"   🤝 Ambos archivos muestran niveles similares de suavidad")
    
    print(f"\n🔍 COMPONENTES DE SUAVIDAD:")
    print(f"   • Variación Total: Medida general de 'zigzag' en Y(t)")
    print(f"   • Cambios Velocidad: Número de cambios de dirección")
    print(f"   • Cambios aceleración: Cambios en la curvatura")
    
    print("="*80)

def process_data_and_save_csv(results_dir, output_dir):
    """
    Procesa todos los datos y guarda los resultados en un CSV consolidado.
    """
    print("🔍 PROCESAMIENTO DE DATOS - ANÁLISIS DE SUAVIDAD")
    print("="*60)
    
    # Cargar datos de múltiples deltaq
    print(f"\n📂 Cargando datos desde: {results_dir}")
    deltaq_data = load_multiple_deltaq_data(results_dir)
    
    if not deltaq_data:
        print("❌ No se encontraron datos válidos de DeltaQ. Terminando.")
        return None
    
    print(f"\n✅ Cargados {len(deltaq_data)} experimentos de DeltaQ exitosamente")
    
    # Escribir resultados consolidados en CSV
    csv_file = os.path.join(output_dir, 'smoothness_analysis_consolidated.csv')
    write_analysis_to_csv(deltaq_data, csv_file)
    
    return csv_file

def generate_plots_from_csv(csv_file, output_dir):
    """
    Genera todos los gráficos a partir del CSV consolidado.
    """
    print("\n📊 GENERACIÓN DE GRÁFICOS DESDE CSV CONSOLIDADO")
    print("="*60)
    
    # Cargar datos desde CSV
    deltaq_data = load_analysis_from_csv(csv_file)
    
    if not deltaq_data:
        print("❌ No se pudieron cargar los datos desde el CSV. Terminando.")
        return
    
    # Generar gráficos multi-deltaq
    print("\n📊 Generando gráficos de análisis de suavidad multi-DeltaQ...")
    
    print("  - Generando gráfico de distribución de suavidad multi-DeltaQ...")
    plot_smoothness_distribution_multi(deltaq_data, output_dir)
    
    print("  - Generando gráfico de componentes de suavidad multi-DeltaQ...")
    plot_smoothness_components_multi(deltaq_data, output_dir)
    
    print("  - Generando gráfico de resumen multi-DeltaQ...")
    plot_smoothness_summary_multi(deltaq_data, output_dir)
    
    # Imprimir reporte resumido
    print("\n📊 REPORTE RESUMIDO DE SUAVIDAD POR DELTAQ")
    print("="*60)
    
    for deltaq in sorted(deltaq_data.keys()):
        data = deltaq_data[deltaq]
        smoothness = data['smoothness']
        if smoothness:
            scores = [d['smoothness_score'] for d in smoothness.values()]
            avg_smooth = np.mean(scores)
            std_smooth = np.std(scores)
            print(f"ΔQ=1e{float(deltaq):>2}: Suavidad={avg_smooth:.3f}±{std_smooth:.3f} (n={len(scores)})")
    
    # Encontrar el mejor y peor DeltaQ
    best_deltaq = max(deltaq_data.keys(), 
                     key=lambda x: np.mean([d['smoothness_score'] for d in deltaq_data[x]['smoothness'].values()]) 
                     if deltaq_data[x]['smoothness'] else 0)
    worst_deltaq = min(deltaq_data.keys(), 
                      key=lambda x: np.mean([d['smoothness_score'] for d in deltaq_data[x]['smoothness'].values()]) 
                      if deltaq_data[x]['smoothness'] else 1)
    
    print(f"\n🏆 MEJOR DeltaQ: 1e{float(best_deltaq)} (más suave)")
    print(f"⚠️  PEOR DeltaQ: 1e{float(worst_deltaq)} (más zigzag)")
    
    print(f"\n✅ Gráficos generados y guardados en: {output_dir}")
    print("📁 Archivos generados:")
    print("  - smoothness_analysis_consolidated.csv")
    print("  - smoothness_distribution_multi_deltaq.png") 
    print("  - smoothness_components_multi_deltaq.png")
    print("  - smoothness_summary_multi_deltaq.png")

def main():
    parser = argparse.ArgumentParser(description='Analizar suavidad de la función Y(t) para múltiples DeltaQ')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/deltaq/results',
                       help='Directorio con los resultados de experimentos DeltaQ')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/deltaq', 
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