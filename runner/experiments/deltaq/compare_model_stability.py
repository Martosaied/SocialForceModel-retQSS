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

    # Only return deltaq_-1, deltaq_-2, deltaq_-3, deltaq_-8
    return [x for x in deltaq_dirs if x[0] in [-1, -2, -3, -4, -5, -6, -7, -8]]

def load_multiple_deltaq_data(results_dir):
    """
    Carga datos de múltiples experimentos de deltaq.
    """
    deltaq_data = {}
    deltaq_dirs = find_deltaq_directories(results_dir)
    
    print(f"🔍 Encontrados {len(deltaq_dirs)} experimentos de DeltaQ:")
    
    for deltaq_value, deltaq_dir in deltaq_dirs:
        result_file = os.path.join(results_dir, deltaq_dir, 'latest', 'result_0.csv')
        
        if os.path.exists(result_file):
            print(f"  - Cargando ΔQ={deltaq_value}...")
            df = load_result_file(result_file)
            
            if df is not None:
                smoothness = analyze_function_smoothness(df)
                deltaq_data[deltaq_value] = {
                    'df': df,
                    'smoothness': smoothness,
                    'file_path': result_file
                }
                print(f"    ✅ Cargado exitosamente ({len(smoothness)} peatones analizados)")
            else:
                print(f"    ❌ Error al cargar datos")
        else:
            print(f"  - ⚠️  No se encontró result_0.csv para ΔQ={deltaq_value}")
    
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
                label = f'ΔQ={deltaq}' if j == 0 else ""
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
    Genera gráfico de distribución de puntuaciones de suavidad para múltiples deltaq.
    """
    plt.figure(figsize=(12, 8))
    plt.title('Distribución de Puntuaciones de Suavidad por DeltaQ', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(deltaq_data)))
    
    for i, (deltaq, data) in enumerate(deltaq_data.items()):
        smoothness = data['smoothness']
        scores = [data['smoothness_score'] for data in smoothness.values()] if smoothness else []
        
        if scores:
            plt.hist(scores, bins=15, alpha=0.6, color=colors[i], 
                    label=f'ΔQ={deltaq} (n={len(scores)})', density=True)
    
    plt.xlabel('Puntuación de Suavidad (0=Zigzag, 1=Suave)')
    plt.ylabel('Densidad')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_distribution_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_components_multi(deltaq_data, output_dir):
    """
    Genera gráfico de componentes de suavidad para múltiples deltaq.
    """
    plt.figure(figsize=(14, 8))
    plt.title('Componentes de Suavidad por DeltaQ', fontsize=14, fontweight='bold')
    
    # Calcular promedios de componentes para cada deltaq
    deltaq_values = []
    components_data = {
        'Variación Total': [],
        'Cambios Velocidad': [],
        'Cambios Aceleración': [],
        'Desv. Est. Velocidad': [],
        'Desv. Est. Aceleración': [],
    }
    
    for deltaq, data in deltaq_data.items():
        smoothness = data['smoothness']
        if smoothness:
            deltaq_values.append(deltaq)
            components_data['Variación Total'].append(np.mean([d['total_variation'] for d in smoothness.values()]))
            components_data['Cambios Velocidad'].append(np.mean([d['velocity_sign_changes'] for d in smoothness.values()]))
            components_data['Cambios Aceleración'].append(np.mean([d['accel_sign_changes'] for d in smoothness.values()]))
            components_data['Desv. Est. Velocidad'].append(np.mean([d['velocity_std'] for d in smoothness.values()]))
            components_data['Desv. Est. Aceleración'].append(np.mean([d['acceleration_std'] for d in smoothness.values()]))
        
    if deltaq_values:
        x = np.arange(len(deltaq_values))
        width = 0.2
        
        for i, (component, values) in enumerate(components_data.items()):
            plt.bar(x + i * width, values, width, label=component, alpha=0.8)
        
        plt.xlabel('DeltaQ')
        plt.ylabel('Valor Promedio')
        plt.xticks(x + width, deltaq_values, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_components_multi_deltaq.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothness_summary_multi(deltaq_data, output_dir):
    """
    Genera gráfico de resumen estadístico para múltiples deltaq.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Resumen de Análisis de Suavidad por DeltaQ', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Suavidad promedio por DeltaQ
    deltaq_values = []
    avg_smoothness = []
    std_smoothness = []
    
    for deltaq, data in deltaq_data.items():
        smoothness = data['smoothness']
        if smoothness:
            scores = [d['smoothness_score'] for d in smoothness.values()]
            deltaq_values.append(deltaq)
            avg_smoothness.append(np.mean(scores))
            std_smoothness.append(np.std(scores))
    
    if deltaq_values:
        ax1.errorbar(deltaq_values, avg_smoothness, yerr=std_smoothness, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_xlabel('DeltaQ')
        ax1.set_ylabel('Suavidad Promedio')
        ax1.set_title('Suavidad Promedio por DeltaQ')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Colorear puntos según suavidad
        colors = plt.cm.RdYlGn([s for s in avg_smoothness])
        for i, (x, y) in enumerate(zip(deltaq_values, avg_smoothness)):
            ax1.scatter(x, y, c=[colors[i]], s=100, zorder=5)
    
    # Gráfico 2: Tabla de estadísticas
    ax2.axis('off')
    ax2.set_title('Estadísticas Detalladas')
    
    # Crear tabla de datos
    table_data = []
    for i, deltaq in enumerate(deltaq_values):
        scores = [d['smoothness_score'] for d in deltaq_data[deltaq]['smoothness'].values()]
        table_data.append([
            f'ΔQ={deltaq}',
            f'{avg_smoothness[i]:.3f}',
            f'{std_smoothness[i]:.3f}',
            f'{len(scores)}'
        ])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['DeltaQ', 'Suavidad Promedio', 'Desv. Est.', 'Peatones'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colorear filas según suavidad
    for i in range(len(table_data)):
        color = colors[i] if i < len(colors) else 'white'
        for j in range(4):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
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
            'Cambios Aceleración': np.mean([data['accel_sign_changes'] for data in smoothness1.values()]),
        }
        
        comp2 = {
            'Variación Total': np.mean([data['total_variation'] for data in smoothness2.values()]),
            'Cambios Velocidad': np.mean([data['velocity_sign_changes'] for data in smoothness2.values()]),
            'Cambios Aceleración': np.mean([data['accel_sign_changes'] for data in smoothness2.values()]),
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
    print(f"   • Cambios Aceleración: Cambios en la curvatura")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analizar suavidad de la función Y(t) para múltiples DeltaQ')
    parser.add_argument('--results-dir', '-r', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/deltaq/results',
                       help='Directorio con los resultados de experimentos DeltaQ')
    parser.add_argument('--output', '-o', 
                       default='/home/martin/Documents/UBA/Tesis/runner/experiments/deltaq', 
                       help='Directorio de salida para los resultados')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    print("🔍 ANÁLISIS DE SUAVIDAD DE LA FUNCIÓN Y(t) - MÚLTIPLES DELTAQ")
    print("="*70)
    
    # Cargar datos de múltiples deltaq
    print(f"\n📂 Cargando datos desde: {args.results_dir}")
    deltaq_data = load_multiple_deltaq_data(args.results_dir)
    
    if not deltaq_data:
        print("❌ No se encontraron datos válidos de DeltaQ. Terminando.")
        return
    
    print(f"\n✅ Cargados {len(deltaq_data)} experimentos de DeltaQ exitosamente")
    
    # Generar gráficos multi-deltaq
    print("\n📊 Generando gráficos de análisis de suavidad multi-DeltaQ...")
    
    print("  - Generando gráfico de funciones Y(t) multi-DeltaQ...")
    plot_y_functions_multi(deltaq_data, args.output)
    
    print("  - Generando gráfico de distribución de suavidad multi-DeltaQ...")
    plot_smoothness_distribution_multi(deltaq_data, args.output)
    
    print("  - Generando gráfico de componentes de suavidad multi-DeltaQ...")
    plot_smoothness_components_multi(deltaq_data, args.output)
    
    print("  - Generando gráfico de resumen multi-DeltaQ...")
    plot_smoothness_summary_multi(deltaq_data, args.output)
    
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
            print(f"ΔQ={deltaq:>6}: Suavidad={avg_smooth:.3f}±{std_smooth:.3f} (n={len(scores)})")
    
    # Encontrar el mejor y peor DeltaQ
    best_deltaq = max(deltaq_data.keys(), 
                     key=lambda x: np.mean([d['smoothness_score'] for d in deltaq_data[x]['smoothness'].values()]) 
                     if deltaq_data[x]['smoothness'] else 0)
    worst_deltaq = min(deltaq_data.keys(), 
                      key=lambda x: np.mean([d['smoothness_score'] for d in deltaq_data[x]['smoothness'].values()]) 
                      if deltaq_data[x]['smoothness'] else 1)
    
    print(f"\n🏆 MEJOR DeltaQ: {best_deltaq} (más suave)")
    print(f"⚠️  PEOR DeltaQ: {worst_deltaq} (más zigzag)")
    
    print(f"\n✅ Análisis completado. Gráficos guardados en: {args.output}")
    print("📁 Archivos generados:")
    print("  - y_functions_multi_deltaq.png")
    print("  - smoothness_distribution_multi_deltaq.png") 
    print("  - smoothness_components_multi_deltaq.png")
    print("  - smoothness_summary_multi_deltaq.png")

if __name__ == '__main__':
    main()