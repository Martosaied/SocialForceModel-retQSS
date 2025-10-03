#!/usr/bin/env python3
"""
Script para analizar la suavidad de la función Y(t) de cada partícula para diferentes valores de MOTIVATION_UPDATE_DT.
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
        
        # 1. Variación total de la función (medida de "zigzag")
        total_variation = np.sum(np.abs(np.diff(y_values)))
        
        # 2. Número de cambios de signo en la velocidad (zigzag)
        velocity_sign_changes = np.sum(np.diff(np.sign(ped_data['y_velocity'].dropna())) != 0)
        
        # 3. Número de cambios de signo en la aceleración (cambios de curvatura)
        accel_sign_changes = np.sum(np.diff(np.sign(ped_data['y_acceleration'].dropna())) != 0)
        
        # 4. Suavidad basada en la regularidad de las derivadas
        velocity_std = ped_data['y_velocity'].std()
        acceleration_std = ped_data['y_acceleration'].std()

        # 5. Puntuación de suavidad compuesta
        # Valores más altos = más suave, valores más bajos = más zigzag
        smoothness_score = calculate_smoothness_score(
            total_variation, velocity_sign_changes, accel_sign_changes,
            velocity_std, acceleration_std,
            len(ped_data)
        )
        
        smoothness_analysis[ped_id] = {
            'total_variation': total_variation,
            'velocity_sign_changes': velocity_sign_changes,
            'accel_sign_changes': accel_sign_changes,
            'velocity_std': velocity_std,
            'acceleration_std': acceleration_std,
            'smoothness_score': smoothness_score,
            'total_steps': len(ped_data),
            'y_range': np.max(y_values) - np.min(y_values)
        }
    
    return smoothness_analysis

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
    Carga datos de múltiples experimentos de motivation_update_dt.
    """
    motivation_dt_data = {}
    
    # Buscar directorios de resultados
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith('motivation_dt_'):
            try:
                motivation_dt = float(item.split('motivation_dt_')[1])
                latest_dir = os.path.join(item_path, 'latest')
                
                if os.path.exists(latest_dir):
                    result_file = os.path.join(latest_dir, 'result_1.csv')
                    if os.path.exists(result_file):
                        df = load_result_file(result_file)
                        if df is not None:
                            smoothness = analyze_function_smoothness(df)
                            motivation_dt_data[motivation_dt] = {
                                'data': df,
                                'smoothness': smoothness
                            }
                            print(f"✅ Cargado motivation_dt={motivation_dt}: {len(smoothness)} peatones analizados")
            except Exception as e:
                print(f"❌ Error procesando {item}: {e}")
                continue
    
    return motivation_dt_data

def plot_y_functions_multi(motivation_dt_data, output_dir):
    """
    Genera gráfico de las funciones Y(t) de muestra para múltiples motivation_update_dt.
    """
    plt.figure(figsize=(14, 10))
    plt.title('Funciones Y(t) de Muestra - Análisis de Suavidad por Motivation Update DT', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motivation_dt_data)))
    
    for i, (motivation_dt, data) in enumerate(sorted(motivation_dt_data.items())):
        df = data['data']
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
                label = f'DT={motivation_dt:.3f}' if j == 0 else ""
                plt.plot(ped_data['time'], ped_data['y'], color=colors[i], 
                        alpha=alpha, linewidth=linewidth, label=label)
    
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición Y (m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'y_functions_multi_motivation_dt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_distribution_multi(motivation_dt_data, output_dir):
    """
    Genera gráfico de distribución de puntuaciones de suavidad para múltiples motivation_update_dt.
    """
    plt.figure(figsize=(12, 8))
    plt.title('Distribución de Puntuaciones de Suavidad por Motivation Update DT', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motivation_dt_data)))
    
    for i, (motivation_dt, data) in enumerate(sorted(motivation_dt_data.items())):
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            plt.hist(scores, bins=15, alpha=0.6, color=colors[i], 
                    label=f'DT={motivation_dt:.3f} (n={len(scores)})', density=True)
    
    plt.xlabel('Puntuación de Suavidad (0=Zigzag, 1=Suave)')
    plt.ylabel('Densidad')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_distribution_multi_motivation_dt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_components_multi(motivation_dt_data, output_dir):
    """
    Genera gráfico de componentes de suavidad para múltiples motivation_update_dt.
    """
    plt.figure(figsize=(14, 8))
    plt.title('Componentes de Suavidad por Motivation Update DT', fontsize=14, fontweight='bold')
    
    # Calcular promedios de componentes para cada motivation_dt
    motivation_dt_values = []
    components_data = {
        'Variación Total': [],
        'Cambios Velocidad': [],
        'Cambios Aceleración': [],
    }
    
    for motivation_dt, data in sorted(motivation_dt_data.items()):
        smoothness = data['smoothness']
        if smoothness:
            motivation_dt_values.append(motivation_dt)
            components_data['Variación Total'].append(np.mean([d['total_variation'] for d in smoothness.values()]))
            components_data['Cambios Velocidad'].append(np.mean([d['velocity_sign_changes'] for d in smoothness.values()]))
            components_data['Cambios Aceleración'].append(np.mean([d['accel_sign_changes'] for d in smoothness.values()]))
        
    if motivation_dt_values:
        x = np.arange(len(motivation_dt_values))
        width = 0.2
        
        for i, (component, values) in enumerate(components_data.items()):
            plt.bar(x + i * width, values, width, label=component, alpha=0.8)
        
        plt.xlabel('Motivation Update DT')
        plt.ylabel('Valor Promedio')
        # Format motivation_dt values
        motivation_dt_labels = [f'{dt:.3f}' for dt in motivation_dt_values]
        plt.xticks(x + width, motivation_dt_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_components_multi_motivation_dt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_summary_multi(motivation_dt_data, output_dir):
    """
    Genera gráfico de resumen estadístico para múltiples motivation_update_dt.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Resumen de Análisis de Suavidad por Motivation Update DT', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Suavidad promedio por Motivation Update DT
    motivation_dt_values = []
    avg_smoothness = []
    std_smoothness = []
    
    for motivation_dt, data in sorted(motivation_dt_data.items()):
        smoothness = data['smoothness']
        if smoothness:
            scores = [d['smoothness_score'] for d in smoothness.values()]
            motivation_dt_values.append(motivation_dt)
            avg_smoothness.append(np.mean(scores))
            std_smoothness.append(np.std(scores))
    
    if motivation_dt_values:
        ax1.errorbar(motivation_dt_values, avg_smoothness, yerr=std_smoothness, 
                    marker='o', capsize=5, linewidth=2)
        ax1.set_xlabel('Motivation Update DT')
        ax1.set_ylabel('Suavidad Promedio')
        ax1.set_title('(a) Suavidad Promedio por Motivation Update DT')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de puntuaciones
        all_scores = []
        all_labels = []
        
        for motivation_dt, data in sorted(motivation_dt_data.items()):
            smoothness = data['smoothness']
            if smoothness:
                scores = [d['smoothness_score'] for d in smoothness.values()]
                all_scores.extend(scores)
                all_labels.extend([f'DT={motivation_dt:.3f}'] * len(scores))
        
        if all_scores:
            df_plot = pd.DataFrame({
                'Motivation DT': all_labels,
                'Smoothness Score': all_scores
            })
            
            df_plot.boxplot(column='Smoothness Score', by='Motivation DT', ax=ax2)
            ax2.set_title('(b) Distribución de Puntuaciones de Suavidad')
            ax2.set_xlabel('Motivation Update DT')
            ax2.set_ylabel('Puntuación de Suavidad')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_summary_multi_motivation_dt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_smoothness_report(smoothness1, smoothness2, file1_name, file2_name):
    """
    Imprime un reporte comparativo de suavidad.
    """
    print("\n" + "="*80)
    print("REPORTE DE SUAVIDAD - COMPARACIÓN")
    print("="*80)
    
    # Calcular estadísticas para cada archivo
    stats1 = calculate_smoothness_stats(smoothness1)
    stats2 = calculate_smoothness_stats(smoothness2)
    
    print(f"\n📊 ESTADÍSTICAS GENERALES")
    print(f"{'Métrica':<25} {'Archivo 1':<15} {'Archivo 2':<15} {'Diferencia':<15}")
    print("-" * 80)
    
    for metric in ['smoothness_score', 'direction_changes', 'acceleration_std', 'max_acceleration']:
        val1 = stats1[metric]['mean']
        val2 = stats2[metric]['mean']
        diff = val2 - val1
        print(f"{metric:<25} {val1:<15.4f} {val2:<15.4f} {diff:<15.4f}")
    
    print(f"\n📈 ANÁLISIS DE SUAVIDAD")
    print(f"Archivo 1 ({file1_name}):")
    print(f"  - Peatones analizados: {stats1['n_pedestrians']}")
    print(f"  - Smoothness promedio: {stats1['smoothness_score']['mean']:.4f} ± {stats1['smoothness_score']['std']:.4f}")
    print(f"  - Cambios de dirección promedio: {stats1['direction_changes']['mean']:.2f} ± {stats1['direction_changes']['std']:.2f}")
    
    print(f"\nArchivo 2 ({file2_name}):")
    print(f"  - Peatones analizados: {stats2['n_pedestrians']}")
    print(f"  - Smoothness promedio: {stats2['smoothness_score']['mean']:.4f} ± {stats2['smoothness_score']['std']:.4f}")
    print(f"  - Cambios de dirección promedio: {stats2['direction_changes']['mean']:.2f} ± {stats2['direction_changes']['std']:.2f}")
    
    # Determinar cuál es más suave
    if stats1['smoothness_score']['mean'] > stats2['smoothness_score']['mean']:
        print(f"\n✅ {file1_name} produce funciones Y(t) MÁS SUAVES")
    elif stats2['smoothness_score']['mean'] > stats1['smoothness_score']['mean']:
        print(f"\n✅ {file2_name} produce funciones Y(t) MÁS SUAVES")
    else:
        print(f"\n🤝 Ambos archivos producen funciones Y(t) con suavidad similar")
    
    print("="*80)

def calculate_smoothness_stats(smoothness_data):
    """
    Calcula estadísticas de suavidad.
    """
    if not smoothness_data:
        return {}
    
    stats = {}
    for metric in ['smoothness_score', 'direction_changes', 'acceleration_std', 'max_acceleration']:
        values = [s[metric] for s in smoothness_data.values()]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    stats['n_pedestrians'] = len(smoothness_data)
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analizar suavidad de la función Y(t) para múltiples Motivation Update DT')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt/results',
                       help='Directorio con los resultados de experimentos Motivation Update DT')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/motivation_update_dt', 
                       help='Directorio de salida para los resultados')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    print("🔍 ANÁLISIS DE SUAVIDAD DE LA FUNCIÓN Y(t) - MÚLTIPLES MOTIVATION UPDATE DT")
    print("="*70)
    
    # Cargar datos de múltiples motivation_dt
    print(f"\n📂 Cargando datos desde: {args.results_dir}")
    motivation_dt_data = load_multiple_motivation_dt_data(args.results_dir)
    
    if not motivation_dt_data:
        print("❌ No se encontraron datos válidos de Motivation Update DT. Terminando.")
        return
    
    print(f"\n✅ Cargados {len(motivation_dt_data)} experimentos de Motivation Update DT exitosamente")
    
    # Generar gráficos multi-motivation_dt
    print("\n📊 Generando gráficos de análisis de suavidad multi-Motivation DT...")
    
    print("  - Generando gráfico de funciones Y(t) multi-Motivation DT...")
    plot_y_functions_multi(motivation_dt_data, args.output)
    
    print("  - Generando gráfico de distribución de suavidad multi-Motivation DT...")
    plot_smoothness_distribution_multi(motivation_dt_data, args.output)
    
    print("  - Generando gráfico de componentes de suavidad multi-Motivation DT...")
    plot_smoothness_components_multi(motivation_dt_data, args.output)
    
    print("  - Generando gráfico de resumen multi-Motivation DT...")
    plot_smoothness_summary_multi(motivation_dt_data, args.output)
    
    # Imprimir reporte resumido
    print("\n📊 REPORTE RESUMIDO DE SUAVIDAD POR MOTIVATION UPDATE DT")
    print("="*60)
    
    for motivation_dt in sorted(motivation_dt_data.keys()):
        smoothness = motivation_dt_data[motivation_dt]['smoothness']
        stats = calculate_smoothness_stats(smoothness)
        
        print(f"\nMotivation DT = {motivation_dt:.3f}:")
        print(f"  - Peatones: {stats['n_pedestrians']}")
        print(f"  - Smoothness: {stats['smoothness_score']['mean']:.4f} ± {stats['smoothness_score']['std']:.4f}")
        print(f"  - Cambios de dirección: {stats['direction_changes']['mean']:.2f} ± {stats['direction_changes']['std']:.2f}")
    
    print("\n✅ Análisis completado. Gráficos guardados en:", args.output)
    print("="*70)

if __name__ == '__main__':
    main()
