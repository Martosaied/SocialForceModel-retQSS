# Runner - Sistema de Ejecución de Experimentos

## Descripción General

El proyecto **Runner** es un sistema de automatización para ejecutar experimentos de simulación de peatones utilizando retQSS. Está diseñado para facilitar la ejecución de múltiples iteraciones de simulaciones con diferentes parámetros, así como para generar visualizaciones y análisis de los resultados.

## Estructura del Proyecto

```
runner/
├── main.py                 # Punto de entrada principal
├── config.json            # Configuración por defecto
├── src/                   # Código fuente del runner
│   ├── runner.py          # Lógica principal de ejecución
│   ├── config_manager.py  # Gestión de configuración
│   ├── utils.py           # Utilidades generales
│   ├── plotter.py         # Generación de gráficos
│   └── math/              # Cálculos matemáticos
├── experiments/           # Experimentos predefinidos
│   ├── performance_n_pedestrians/
│   ├── average_velocity/
│   ├── breaking_lanes/
│   └── ... (otros experimentos)
└── results/               # Resultados de experimentos standalone y campo de pruebas
```

## Funcionalidad Principal (main.py)

El archivo `main.py` es el punto de entrada del sistema y proporciona tres comandos principales:

### 1. Comando `run`
Ejecuta experimentos con configuración JSON:

```bash
python main.py run <modelo> [opciones]
```

**Parámetros:**
- `modelo`: Nombre del modelo a ejecutar (ej: `social_force_model`)
- `--config`: Archivo de configuración JSON (por defecto: `config.json`)
- `--experiment-name`: Nombre del experimento (crea un subdirectorio)
- `--compile`: Compila el modelo antes de ejecutar
- `--compile-c`: Compila el código C++ antes de ejecutar
- `--output-dir`: Directorio base para resultados (por defecto: `results`)
- `--plot`: Genera gráficos de los resultados
- `--verbose`: Salida detallada
- `--skip-metrics`: Omite cálculos de métricas para ahorrar tiempo
- `--fast-mode`: Modo rápido (omite todos los cálculos)

**Ejemplo:**
```bash
python main.py run social_force_model --experiment-name test_run --compile --plot
```

### 2. Comando `plot`
Genera visualizaciones de resultados existentes:

```bash
python main.py plot <tipo> <archivo_solucion> [opciones]
```

**Tipos de gráficos disponibles:**
- `gif`: Genera un GIF de flujo de peatones
- `grouped_lanes`: Gráfico de carriles agrupados
- `pedestrian_heatmap`: Mapa de calor de densidad de peatones
- `flow_graph_infections`: Gráfico de flujo de peatones con visualización de nivel de contagio en volúmenes

**Ejemplo:**
```bash
python main.py plot gif results/solution.csv --output-dir plots/
```

### 3. Comando `experiments`
Ejecuta experimentos predefinidos:

```bash
python main.py experiments <experimento> [opciones]
```

## Carpeta Experiments

La carpeta `experiments/` contiene experimentos predefinidos que automatizan la ejecución de simulaciones con diferentes parámetros y configuraciones. Cada experimento está organizado en su propio directorio.

### Experimentos Disponibles

#### 1. **performance_n_pedestrians**
Analiza el rendimiento del sistema con diferentes cantidades de peatones (10, 50, 100, 200, 300, 500, 1000, 2000, 3000) y diferentes implementaciones (mmoc, retqss).

**Funcionalidad:**
- Ejecuta simulaciones con diferentes números de peatones
- Compara rendimiento entre implementaciones
- Genera gráficos de tiempo de ejecución vs número de peatones

#### 2. **average_velocity**
Estudia la velocidad promedio de los peatones en diferentes escenarios para analizar la formación de carriles.

**Funcionalidad:**
- Calcula velocidad promedio en 6 momentos diferentes de la simulación
- Analiza diferentes números de volúmenes (1, 2, 3, 5, 10, 20, 50, 100)
- Genera frames de simulación para visualizar formación de carriles

#### 3. **breaking_lanes**
Analiza la formación y ruptura de carriles de peatones.

#### 4. **lanes_by_iterations**
Estudia la evolución de carriles a través de iteraciones individuales.

#### 5. **lanes_by_width**
Analiza la formación de carriles según el ancho del pasillo.

#### 6. **deltaq**
Experimento relacionado con el parámetro deltaq del modelo.

#### 7. **lanes_by_B, lanes_by_R, lanes_by_A**
Experimentos que analizan la formación de carriles según diferentes parámetros del modelo (B, R, A).

#### 8. **lanes_heatmap**
Genera mapas de calor a partir de la combinación de diferentes parámetros del modelo (B, R, A).

#### 9. **breaking_obstacles**
Analiza el comportamiento de peatones ante obstáculos y el minimo de tamanio de obstaculo para que los peatones se detengan.

#### 10. **subway_attack_rate**
Experimento específico para simulaciones de metro con tasas de ataque.

#### 11. **progress_update_dt**
Analiza la actualización de progreso con diferentes pasos de tiempo.

#### 12. **performance_n_volumes, performance_n_peds_m_vols**
Experimentos de rendimiento con diferentes volúmenes y combinaciones de peatones/volúmenes.

### Estructura de un Experimento

Cada experimento sigue una estructura similar:

```
experiments/nombre_experimento/
├── nombre_experimento.py    # Script principal del experimento
├── results/                 # Resultados del experimento
│   ├── n_X_implementation_Y/  # Resultados por configuración
│   └── latest/              # Últimos resultados
└── README.md               # Documentación específica (opcional)
```

### Ejecutar un Experimento

```bash
python main.py experiments performance_n_pedestrians --verbose
```

## Configuración

El sistema utiliza archivos JSON para la configuración. El archivo `config.json` principal contiene:

- **iterations**: Número de iteraciones por experimento
- **parameters**: Lista de parámetros del modelo
- **max_concurrent_processes**: Número máximo de procesos concurrentes

### Parámetros Principales

- `N`: Número de peatones
- `PEDESTRIAN_IMPLEMENTATION`: Implementación de peatones (0: mmoc, 1: retqss)
- `BORDER_IMPLEMENTATION`: Implementación de bordes
- `GRID_SIZE`: Tamaño de la grilla
- `FORCE_TERMINATION_AT`: Tiempo de terminación forzada
- Parámetros del modelo de fuerza social (A, B, R, etc.)

## Características Principales

### 1. **Ejecución Paralela**
- Soporte para múltiples procesos concurrentes
- Optimización automática basada en CPU disponible

### 2. **Gestión de Resultados**
- Organización automática de resultados por experimento
- Copia automática a directorio "latest"
- Métricas de rendimiento (tiempo, memoria, grupos)
- **Carpeta results/**: Contiene resultados de experimentos "standalone" (ejecutados con comando `run`) y campo de pruebas

### 3. **Visualización**
- Generación automática de gráficos
- Diferentes tipos de visualización disponibles
- Soporte para GIFs y mapas de calor

### 4. **Flexibilidad**
- Configuración por archivo JSON
- Parámetros personalizables
- Experimentos predefinidos y personalizados

## Dependencias

El sistema requiere:
- Python 3.x
- Bibliotecas: pandas, numpy, matplotlib
- Compilador C++ (para modelos compilados)
- Modelo RETQSS (en directorio padre)

## Uso Típico

1. **Configurar parámetros** en `config.json`
2. **Ejecutar experimento standalone (campo de pruebas):**
   ```bash
   python main.py run social_force_model --compile --plot
   ```
3. **Ejecutar experimento predefinido (con análisis propio):**
   ```bash
   python main.py experiments performance_n_pedestrians
   ```
4. **Generar visualizaciones:**
   ```bash
   python main.py plot gif results/solution.csv
   ```

## Notas Importantes

- Los experimentos pueden tomar tiempo considerable dependiendo del número de iteraciones y peatones
- Se recomienda usar `--fast-mode` para pruebas rápidas
- Los resultados se guardan automáticamente en directorios organizados por fecha/experimento
- El sistema maneja errores y crea logs de errores para debugging
