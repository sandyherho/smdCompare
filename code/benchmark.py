#!/usr/bin/env python
import subprocess
import re
import csv
import os

# Configuration constants
WARMUP_RUNS = 5
NUM_RUNS = 1000

# Script pairs to benchmark
BENCHMARKS = {
    'open': {
        'python': "./open_compute.py",
        'r': "./open_compute.R",
        'output': "../outputs/data/openSim.csv"
    },
    'controlled': {
        'python': "./controlled_compute.py",
        'r': "./controlled_compute.R",
        'output': "../outputs/data/controlledSim.csv"
    }
}

# Regex patterns for parsing time output
TIME_REGEX = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d.:]+)")
MEM_REGEX = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")

def run_with_time(script_cmd):
    """
    Runs the given command using '/usr/bin/time -v' and parses out
    the elapsed time (in seconds) and max memory usage (in kB).
    Returns (execution_time_in_seconds, memory_usage_in_kb).
    """
    command = ["/usr/bin/time", "-v"] + script_cmd
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        _, stderr = proc.communicate()
        
        time_match = TIME_REGEX.search(stderr)
        mem_match = MEM_REGEX.search(stderr)
        
        if not time_match or not mem_match:
            print(f"Debug - Time Output: {stderr}")  # Debug output
            raise RuntimeError(
                f"Unable to parse time or memory from '/usr/bin/time' output:\n{stderr}"
            )
            
        raw_time = time_match.group(1)
        execution_time = convert_time_string_to_seconds(raw_time)
        memory_usage = int(mem_match.group(1))
        
        return execution_time, memory_usage
        
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to execute command: {e}")

def convert_time_string_to_seconds(time_str):
    """
    Converts a time string of the form "mm:ss", "hh:mm:ss", or "ss.sss"
    into seconds (float).
    Examples:
      "1.23"     -> 1.23 seconds
      "0:01.23"  -> 1.23 seconds
      "1:02"     -> 62 seconds
      "1:02:03"  -> 3723 seconds
    """
    parts = time_str.split(":")
    if len(parts) == 1:
        # Just seconds (possibly fractional)
        return float(parts[0])
    elif len(parts) == 2:
        # mm:ss
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # hh:mm:ss
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Unrecognized time format: {time_str}")

def run_benchmark(benchmark_name, benchmark_config):
    """
    Runs benchmarks for a pair of Python and R scripts and saves results to CSV.
    """
    output_path = benchmark_config['output']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nStarting {benchmark_name} benchmark suite...")
    
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["script", "run_index", "execution_time", "memory_usage"])
        
        # Run benchmarks for both Python and R scripts
        scripts = {
            f"{benchmark_name}_compute.py": benchmark_config['python'],
            f"{benchmark_name}_compute.R": benchmark_config['r']
        }
        
        for script_name, script_path in scripts.items():
            print(f"Benchmarking {script_name}...")
            command = ["python" if script_name.endswith('.py') else "Rscript", 
                      script_path]
            
            stable_run_index = 1
            total_runs = WARMUP_RUNS + NUM_RUNS
            
            for i in range(1, total_runs + 1):
                try:
                    exec_time, mem_kb = run_with_time(command)
                    if i > WARMUP_RUNS:
                        writer.writerow([script_name, stable_run_index, 
                                       exec_time, mem_kb])
                        stable_run_index += 1
                        
                        # Print progress every 20 runs
                        if stable_run_index % 20 == 0:
                            print(f"Completed {stable_run_index} of {NUM_RUNS} measurements")
                            
                except Exception as e:
                    print(f"Error on run {i} of {script_name}: {e}")
                    continue
                
            print(f"Completed benchmarking {script_name}")
    
    print(f"Results written to {output_path}")

def main():
    for benchmark_name, config in BENCHMARKS.items():
        run_benchmark(benchmark_name, config)
    print("\nAll benchmarks completed!")

if __name__ == "__main__":
    main()
