# Pedestrian Simulation Framework using retQSS

This project implements pedestrian simulation models using the QSS (Quantized State Systems) solver retQSS. The framework includes both the simulation engine (retQSS) and an automated experimentation system (runner) for systematic parameter sweeps and performance analysis.

## Project Structure

```
.
├── retqss/          # QSS simulation engine and pedestrian models
└── runner/          # Automated experimentation and analysis system
```

## Folder Overview

### 📁 retqss/ - Simulation Engine and Models

This folder contains the QSS (Quantized State Systems) solver and pedestrian simulation models implemented in Modelica.

**Key Models:**

- **`social_force_model`**: Main implementation of Helbing's social force model for pedestrian dynamics. This model simulates bidirectional pedestrian flow using continuous forces (attraction to destination, repulsion from pedestrians and boundaries).

- **`helbing_only_qss`**: Simplified version of the social force model optimized for QSS integration. This model focuses on pure QSS-based force calculations without additional discrete event handling.

- **`helbing_school_hallway`**: Specialized scenario modeling school hallway dynamics during class breaks. Includes classroom assignments, break scheduling, and concentration-based movement patterns between classrooms.

The models include features such as:
- Bidirectional pedestrian flows
- Social force interactions between pedestrians
- Boundary repulsion forces
- Volume-based concentration calculations
- Group formation and lane detection
- Configurable parameters (A, B, R - force amplitudes and ranges)

For more details on the simulation engine, see the [retQSS documentation](./retqss/README.md).

### 📁 runner/ - Experimentation System

The **runner** folder contains an automated experimentation framework for running multiple simulations with different parameter combinations, analyzing results, and generating visualizations.

**Main Features:**
- Automated batch execution of simulations
- Parameter sweep capabilities
- Performance analysis and benchmarking
- Automatic result visualization (GIFs, heatmaps, flow graphs)
- Parallel execution support
- Experiment predefinitions for common scenarios

**Available Commands:**
- `run`: Execute standalone experiments
- `plot`: Generate visualizations from results
- `experiments`: Run predefined experimental protocols

For detailed usage instructions, configuration options, and available experiments, see the [runner documentation](./runner/README.md).

## Quick Start

### Running a Simulation

```bash
cd runner
python main.py run social_force_model --compile --plot
```

### Running a Predefined Experiment

```bash
python main.py experiments performance_n_pedestrians
```

## Dependencies

- Python 3.x
- pandas, numpy, matplotlib
- C++ compiler (for model compilation)
- retQSS solver (included in `retqss/`)

## License

See individual LICENSE files in the respective subdirectories.

## Development

This project is part of a thesis research on pedestrian simulation optimization using quantized state systems.

