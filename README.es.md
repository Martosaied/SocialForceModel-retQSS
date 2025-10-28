# Framework de Simulación de Peatones usando retQSS

Este proyecto implementa modelos de simulación de peatones utilizando el solucionador QSS (Quantized State Systems) retQSS. El framework incluye tanto el motor de simulación (retQSS) como un sistema automatizado de experimentación (runner) para barridos sistemáticos de parámetros y análisis de rendimiento.

## Estructura del Proyecto

```
.
├── retqss/          # Motor de simulación QSS y modelos de peatones
└── runner/          # Sistema de experimentación y análisis automatizado
```

## Descripción de Carpetas

### 📁 retqss/ - Motor de Simulación y Modelos

Esta carpeta contiene el solucionador QSS (Quantized State Systems) y los modelos de simulación de peatones implementados en Modelica.

**Modelos Principales:**

- **`social_force_model`**: Implementación principal del modelo de fuerza social de Helbing para dinámica de peatones. Este modelo simula flujos bidireccionales de peatones utilizando fuerzas continuas (atracción al destino, repulsión de peatones y bordes).

- **`helbing_only_qss`**: Versión simplificada del modelo de fuerza social optimizada para integración QSS. Este modelo se enfoca en cálculos de fuerza basados puramente en QSS sin manejo adicional de eventos discretos.

- **`helbing_school_hallway`**: Escenario especializado que modela la dinámica de pasillos de escuelas durante recreos. Incluye asignaciones de aulas, programación de descansos y patrones de movimiento basados en concentración entre aulas.

Los modelos incluyen características como:
- Flujos bidireccionales de peatones
- Interacciones de fuerza social entre peatones
- Fuerzas de repulsión de bordes
- Cálculos de concentración basados en volúmenes
- Formación de grupos y detección de carriles
- Parámetros configurables (A, B, R - amplitudes y rangos de fuerzas)

Para más detalles sobre el motor de simulación, consulta la [documentación de retQSS](./retqss/README.md).

### 📁 runner/ - Sistema de Experimentación

La carpeta **runner** contiene un framework de experimentación automatizado para ejecutar múltiples simulaciones con diferentes combinaciones de parámetros, analizar resultados y generar visualizaciones.

**Características Principales:**
- Ejecución automática por lotes de simulaciones
- Capacidades de barrido de parámetros
- Análisis de rendimiento y benchmarking
- Visualización automática de resultados (GIFs, mapas de calor, gráficos de flujo)
- Soporte para ejecución en paralelo
- Predefiniciones de experimentos para escenarios comunes

**Comandos Disponibles:**
- `run`: Ejecutar experimentos standalone
- `plot`: Generar visualizaciones de resultados
- `experiments`: Ejecutar protocolos experimentales predefinidos

Para instrucciones detalladas de uso, opciones de configuración y experimentos disponibles, consulta la [documentación del runner](./runner/README.md).

## Inicio Rápido

### Ejecutar una Simulación

```bash
cd runner
python main.py run social_force_model --compile --plot
```

### Ejecutar un Experimento Predefinido

```bash
python main.py experiments performance_n_pedestrians
```

## Dependencias

- Python 3.x
- pandas, numpy, matplotlib
- Compilador C++ (para compilación de modelos)
- Solucionador retQSS (incluido en `retqss/`)

## Licencia

Ver archivos LICENSE individuales en los subdirectorios correspondientes.

## Desarrollo

Este proyecto forma parte de una investigación de tesis de Maestría sobre simulación de peatones y dinámica de evacuación usando sistemas de estados cuantizados.

