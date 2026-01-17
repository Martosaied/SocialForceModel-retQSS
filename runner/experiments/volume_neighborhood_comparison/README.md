# Volume Neighborhood Comparison Experiment

This experiment compares the two volume neighborhood types in the social force model:
- **Type 0**: Face Sharing (default) - `retQSS_volumeNeighborhood_toDefault()`
- **Type 1**: Vertex Sharing - `retQSS_volumeNeighborhood_toVertexSharing()`

## Metrics Measured

1. **Performance**: Execution time (seconds)
2. **Memory Usage**: Memory consumption (MB)
3. **Lane Formation**: Number of lanes formed using clustering analysis
4. **Collisions**: Number of personal space invasions (pedestrians within 2 * PEDESTRIAN_R distance)

## Configuration

- **Pedestrian Count**: 300 (density 0.3 pedestrians/m²)
- **Corridor Width**: 20m
- **Grid Size**: 50m
- **Cell Sizes Tested**: 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 25, 50 meters
- **Iterations per Configuration**: 10

## How to Run

### Quick Test (N=20, single cell size)

To validate the pipeline with a small test:

1. Edit `volume_neighborhood_comparison.py`:
   - Set `PEDESTRIAN_COUNT = 20`
   - Set `CELL_SIZES = [5.0]`
   - Set `RUN_EXPERIMENT = True`

2. Run:
```bash
cd /home/martin/Documents/UBA/Tesis/runner
python experiments/volume_neighborhood_comparison/volume_neighborhood_comparison.py
```

### Full Experiment

To run the complete experiment suite:

1. Ensure `volume_neighborhood_comparison.py` has:
   - `PEDESTRIAN_COUNT = 300`
   - `CELL_SIZES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 25, 50]`
   - `RUN_EXPERIMENT = True`

2. Run:
```bash
cd /home/martin/Documents/UBA/Tesis/runner
python experiments/volume_neighborhood_comparison/volume_neighborhood_comparison.py
```

**Note**: Full experiment runs 22 configurations (11 cell sizes × 2 neighborhood types) × 10 iterations = 220 simulations. This may take several hours.

## Output

### Results Directory Structure
```
experiments/volume_neighborhood_comparison/
├── config.json
├── volume_neighborhood_comparison.py
├── README.md
├── results/
│   ├── cell_size_0.5_neighborhood_0/
│   │   └── latest/
│   │       ├── metrics.csv
│   │       └── result_*.csv
│   ├── cell_size_0.5_neighborhood_1/
│   ├── cell_size_1.0_neighborhood_0/
│   └── ...
└── volume_neighborhood_performance_comparison.png
```

### Visualization

The experiment generates a 2×2 comparison plot:
- **Top-left**: Performance (execution time)
- **Top-right**: Memory usage
- **Bottom-left**: Lanes formed
- **Bottom-right**: Collisions

Each subplot shows bar charts comparing Face Sharing (blue) vs Vertex Sharing (orange) across all cell sizes, with error bars showing standard deviation.

## Metrics CSV Format

The enhanced metrics.csv now includes collision data:
```csv
time,memory_usage,density_based_groups,clustering_based_groups,total_collisions,avg_collision_rate
0.523,45231,0,2.3,145,0.0234
```

## Implementation Details

### Collision Detection

Collisions are detected by:
1. Calculating pairwise distances between all pedestrians at each timestep
2. Counting pairs within threshold distance (2 × PEDESTRIAN_R = 0.6m)
3. Aggregating across all timesteps (excluding initial transient period)

The collision detection is implemented in:
- `runner/src/math/Collisions.py`: Reusable collision detection module
- `runner/src/runner.py`: Integrated into the experiment runner

### Volume Neighborhood Parameter

The experiment uses the `VOLUME_NEIGHBORHOOD_TYPE` parameter added to `social_force_model.mo`:
- `0`: Calls `volumeNeighborhood_toDefault()` (face sharing)
- `1`: Calls `volumeNeighborhood_toVertexSharing()` (vertex sharing)

## Troubleshooting

### No data in plots
- Check that experiments completed successfully
- Verify metrics.csv files exist in results/*/latest/ directories
- Check console output for errors

### Compilation errors
- Ensure retQSS C++ code is compiled: `cd ../retqss/src && make`
- Verify social_force_model.mo has VOLUME_NEIGHBORHOOD_TYPE parameter

### Memory issues
- Reduce number of cell sizes tested
- Reduce iterations per configuration
- Run experiments sequentially instead of in parallel

## Related Experiments

- `breaking_lanes`: Similar structure, tests pedestrian implementations
- `performance_n_pedestrians`: Performance scaling experiments
- `motivation_update_dt`: Parameter sensitivity analysis


