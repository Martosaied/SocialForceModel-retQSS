#!/usr/bin/env python3
"""
CCX Locality Test for AMD Ryzen 5 2600

This script tests whether L3 cache contention (same CCX) or memory bandwidth
is causing the performance degradation when running multiple simulations.

Ryzen 5 2600 topology:
- CCX0: physical cores 0, 1, 2 (logical cores 0, 1, 2, 6, 7, 8 with SMT)
- CCX1: physical cores 3, 4, 5 (logical cores 3, 4, 5, 9, 10, 11 with SMT)

Test 1: Same CCX (cores 0, 1, 2) - shared 8MB L3 cache
Test 2: Cross CCX (cores 0, 3) - separate L3 caches, but cross-CCX communication
Test 3: Sequential baseline - one at a time
"""

import subprocess
import os
import sys
import time
import shutil
import random
import string

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.runner import compile_c_code, compile_model

MODEL_NAME = 'social_force_model'
# Use absolute path
RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(RUNNER_DIR, '..', 'retqss', 'build')
NUM_ITERATIONS = 3  # Run 3 simulations per test


def create_temp_dir(model_name: str, suffix: str) -> str:
    """Create a temporary directory for the simulation."""
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    dir_name = f"{model_name}_{suffix}_{random_str}"
    tmp_dir = os.path.join(BUILD_DIR, dir_name)
    tmp_dir = os.path.abspath(tmp_dir)

    src_dir = os.path.join(BUILD_DIR, model_name)
    src_dir = os.path.abspath(src_dir)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    shutil.copytree(src_dir, tmp_dir)
    # Copy the binary explicitly
    subprocess.run(f"cp {src_dir}/{model_name} {tmp_dir}/", shell=True, check=True)

    # Modify the shell script to use the correct path
    sh_file = os.path.join(tmp_dir, f"{model_name}.sh")
    subprocess.run(['sed', '-i', f's|/build/{model_name}/|/build/{dir_name}/|g', sh_file])

    # Create parameters.config file with default test parameters
    params_file = os.path.join(tmp_dir, "parameters.config")
    with open(params_file, 'w') as f:
        f.write("N=5000\n")
        f.write("PEDESTRIAN_IMPLEMENTATION=1\n")
        f.write("BORDER_IMPLEMENTATION=1\n")
        f.write("FORCE_TERMINATION_AT=80.0\n")
        f.write("GRID_SIZE=129.1\n")
        f.write("GRID_DIVISIONS=43\n")
        f.write("FROM_Y=25.82\n")
        f.write("TO_Y=103.28\n")
        f.write("CONVEYOR_BELT_EFFECT=1\n")
        f.write("PEDESTRIAN_A_1=2.1\n")
        f.write("PEDESTRIAN_A_2=2.1\n")
        f.write("PEDESTRIAN_B_1=0.7\n")
        f.write("PEDESTRIAN_B_2=0.7\n")
        f.write("PEDESTRIAN_R=0.3\n")
        f.write("PEDESTRIAN_LAMBDA=0.3\n")
        f.write("BORDER_A=10.0\n")
        f.write("BORDER_B=0.7\n")
        f.write("BORDER_R=0.6\n")
        f.write("SPEED_MU=1.34\n")
        f.write("SPEED_SIGMA=0.26\n")
        f.write("MOTIVATION_UPDATE_DT=0.1\n")
        f.write(f"RANDOM_SEED={random.randint(0, 1000000)}\n")

    return tmp_dir


def parse_time_output(stderr: str) -> tuple:
    """Parse user and sys time from /usr/bin/time -v output."""
    user_time = 0.0
    sys_time = 0.0

    for line in stderr.split('\n'):
        if 'User time (seconds):' in line:
            try:
                user_time = float(line.split(':')[1].strip())
            except (IndexError, ValueError):
                pass
        elif 'System time (seconds):' in line:
            try:
                sys_time = float(line.split(':')[1].strip())
            except (IndexError, ValueError):
                pass

    return user_time, sys_time


def run_simulation_with_affinity(tmp_dir: str, model_name: str, cpu_core: int) -> dict:
    """Run a single simulation pinned to a specific CPU core."""
    cmd = os.path.join(tmp_dir, f"{model_name}.sh")

    # Use taskset to pin to specific CPU core
    full_cmd = f"taskset -c {cpu_core} /usr/bin/time -v {cmd}"

    start_time = time.monotonic()
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=tmp_dir
    )
    wall_time = time.monotonic() - start_time

    # Parse time output
    user_time, sys_time = parse_time_output(result.stderr)

    if user_time == 0:
        print(f"Warning: Could not parse time from stderr (first 500 chars):")
        print(result.stderr[:500])
        if result.returncode != 0:
            print(f"Process exited with code {result.returncode}")

    return {
        'cpu_core': cpu_core,
        'user_time': user_time,
        'sys_time': sys_time,
        'wall_time': wall_time,
        'total_time': user_time + sys_time
    }


def run_parallel_test(cores: list, test_name: str) -> list:
    """Run simulations in parallel on specified cores."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"Cores: {cores}")
    print(f"{'='*60}")

    # Create temp directories
    tmp_dirs = []
    for i, core in enumerate(cores):
        tmp_dir = create_temp_dir(MODEL_NAME, f"test_{i}")
        tmp_dirs.append((tmp_dir, core))
        print(f"Created temp dir for core {core}: {tmp_dir}")

    # Start all processes in parallel
    processes = []
    start_time = time.monotonic()

    for tmp_dir, core in tmp_dirs:
        cmd = f"{tmp_dir}/{MODEL_NAME}.sh"
        full_cmd = f"taskset -c {core} /usr/bin/time -v {cmd}"

        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_dir
        )
        processes.append((proc, core, tmp_dir))

    # Wait for all to complete and collect results
    results = []
    for proc, core, tmp_dir in processes:
        stdout, stderr = proc.communicate()

        user_time, sys_time = parse_time_output(stderr)

        if user_time == 0:
            print(f"Warning: Could not parse time for core {core}")
            print(f"stderr (first 300 chars): {stderr[:300]}")
            if proc.returncode != 0:
                print(f"Process exited with code {proc.returncode}")

        results.append({
            'cpu_core': core,
            'user_time': user_time,
            'sys_time': sys_time,
            'total_time': user_time + sys_time
        })

        # Cleanup
        shutil.rmtree(tmp_dir)

    total_wall_time = time.monotonic() - start_time

    # Print results
    print(f"\nResults for {test_name}:")
    print(f"{'Core':<8} {'User Time':<12} {'Sys Time':<12} {'Total':<12}")
    print("-" * 44)
    for r in results:
        print(f"{r['cpu_core']:<8} {r['user_time']:<12.2f} {r['sys_time']:<12.2f} {r['total_time']:<12.2f}")

    avg_user_time = sum(r['user_time'] for r in results) / len(results)
    print(f"\nAverage user time: {avg_user_time:.2f}s")
    print(f"Total wall time: {total_wall_time:.2f}s")

    return results, total_wall_time, avg_user_time


def run_sequential_test() -> list:
    """Run simulations sequentially as baseline."""
    print(f"\n{'='*60}")
    print(f"Test: Sequential Baseline (one at a time)")
    print(f"{'='*60}")

    results = []
    total_start = time.monotonic()

    for i in range(NUM_ITERATIONS):
        tmp_dir = create_temp_dir(MODEL_NAME, f"seq_{i}")
        print(f"Running iteration {i+1}/{NUM_ITERATIONS} on core 0...")

        result = run_simulation_with_affinity(tmp_dir, MODEL_NAME, cpu_core=0)
        results.append(result)

        print(f"  User time: {result['user_time']:.2f}s")

        # Cleanup
        shutil.rmtree(tmp_dir)

    total_wall_time = time.monotonic() - total_start
    avg_user_time = sum(r['user_time'] for r in results) / len(results)

    print(f"\nAverage user time: {avg_user_time:.2f}s")
    print(f"Total wall time: {total_wall_time:.2f}s")

    return results, total_wall_time, avg_user_time


def main():
    print("CCX Locality Test for AMD Ryzen 5 2600")
    print("=" * 60)

    # Ensure model is compiled
    print("\nCompiling model...")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
    compile_c_code()
    compile_model(MODEL_NAME)

    # Test 1: Sequential baseline
    seq_results, seq_wall, seq_avg = run_sequential_test()

    # Test 2: Same CCX (cores 0, 1, 2 share L3 cache)
    same_ccx_results, same_ccx_wall, same_ccx_avg = run_parallel_test(
        cores=[0, 1, 2],
        test_name="Same CCX (cores 0, 1, 2 - shared 8MB L3)"
    )

    # Test 3: Cross CCX (cores 0, 3 have separate L3 caches)
    # Using only 2 processes to isolate CCX effect
    cross_ccx_results, cross_ccx_wall, cross_ccx_avg = run_parallel_test(
        cores=[0, 3],
        test_name="Cross CCX (cores 0, 3 - separate L3 caches)"
    )

    # Test 4: Same CCX but only 2 processes for fair comparison
    same_ccx_2_results, same_ccx_2_wall, same_ccx_2_avg = run_parallel_test(
        cores=[0, 1],
        test_name="Same CCX 2 procs (cores 0, 1 - shared 8MB L3)"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Test':<40} {'Avg User Time':<15} {'Overhead':<10}")
    print("-" * 65)

    def calc_overhead(avg, baseline):
        if baseline > 0:
            return f"{((avg/baseline)-1)*100:>+.1f}%"
        return "N/A"

    print(f"{'Sequential (baseline)':<40} {seq_avg:<15.2f} {'0%':<10}")
    print(f"{'Same CCX (3 procs, cores 0,1,2)':<40} {same_ccx_avg:<15.2f} {calc_overhead(same_ccx_avg, seq_avg):<10}")
    print(f"{'Same CCX (2 procs, cores 0,1)':<40} {same_ccx_2_avg:<15.2f} {calc_overhead(same_ccx_2_avg, seq_avg):<10}")
    print(f"{'Cross CCX (2 procs, cores 0,3)':<40} {cross_ccx_avg:<15.2f} {calc_overhead(cross_ccx_avg, seq_avg):<10}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if same_ccx_2_avg == 0 or cross_ccx_avg == 0:
        print("-> Could not collect enough data for interpretation")
        print("-> Check the warnings above for parsing errors")
    elif same_ccx_2_avg > cross_ccx_avg * 1.05:
        print("-> Same CCX is slower than Cross CCX")
        print("-> This indicates L3 CACHE CONTENTION is the primary issue")
        print("-> Processes on same CCX compete for shared 8MB L3 cache")
    elif cross_ccx_avg > same_ccx_2_avg * 1.05:
        print("-> Cross CCX is slower than Same CCX")
        print("-> This indicates MEMORY BANDWIDTH is the primary issue")
        print("-> Cross-CCX adds Infinity Fabric latency but that's not the bottleneck")
    else:
        print("-> Same CCX and Cross CCX have similar performance")
        print("-> This indicates MEMORY BANDWIDTH SATURATION is the issue")
        print("-> Both CCXs share the same memory controller")
        print("-> The simulation is memory-bound, not cache-bound")


if __name__ == '__main__':
    main()
