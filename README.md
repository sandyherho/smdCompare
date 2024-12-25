# Supporting Material for "Quantitative Performance Analysis of Spring-Mass-Damper Control Systems: A Comparative Implementation in Python and R"


[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)
[![DOI](https://zenodo.org/badge/908233800.svg)](https://doi.org/10.5281/zenodo.14556320)

This repository contains the code, data, and visualization outputs for the analysis and simulation of spring-mass-damper systems.

## Repository Structure

```plaintext
.
├── README.md                     # This file
├── LICENSE.txt                   # License information
├── code/
│   ├── controlled_compute.py     # Controlled system computation in Python
│   ├── open_compute.py           # Open-loop system computation in Python
│   ├── controlled_compute.R      # Controlled system computation in R
│   ├── open_compute.R            # Open-loop system computation in R
│   ├── controlled_vis.py         # Visualization for controlled systems
│   ├── open_vis.py               # Visualization for open-loop systems
│   ├── stats_benchmark.py        # Statistical benchmarking and performance analysis
│   ├── benchmark.py              # Benchmarking execution and memory
├── outputs/
│   ├── data/                     # Processed data from simulations
│   │   ├── controlledSim.csv                         # Controlled simulation results
│   │   ├── openSim.csv                               # Open-loop simulation results
│   │   ├── controlled_smd_stability_metrics_py.csv   # Controlled system stability metrics in Python
│   │   ├── controlled_smd_stability_interpretations_py.csv # Controlled system stability interpretations
│   │   ├── smd_stability_metrics_py.csv              # General stability metrics in Python
│   │   ├── smd_stability_metrics_r.csv               # General stability metrics in R
│   │   ├── controlled_smd_metrics_py.csv            # Metrics for controlled system in Python
│   │   ├── controlled_smd_simulation_py.csv         # Controlled system simulation in Python
│   │   ├── smd_simulation_py.csv                    # General system simulation in Python
│   │   ├── smd_simulation_r.csv                     # General system simulation in R
│   │   ├── memory_usage_performance_stats.csv       # Memory performance statistics
│   │   ├── execution_time_performance_stats.csv     # Execution time performance statistics
│   │   ├── memory_usage_comparison_stats.csv        # Memory comparison statistics between Python and R
│   │   ├── execution_time_comparison_stats.csv      # Execution time comparison statistics between Python and R
│   │   ├── performance_analysis.xlsx                # Excel file with all performance analysis
│   ├── figs/                     # Figures generated from visualizations
│       ├── memory_usage_violin_plots.png
│       ├── execution_time_violin_plots.png
│       ├── phase_portrait_py.png
│       ├── smd_response_py.png
│       ├── controlled_phase_portrait_py.png
│       └── controlled_smd_response_py.png
```

## Citation

If you use this repository in your research, please cite the paper:

```bibtex
@article{herhoKaban25a,
  author = {S. H. S. Herho and S. N. Kaban},
  title = {{Quantitative Performance Analysis of Spring-Mass-Damper Control Systems: A Comparative Implementation in Python and R}},
  journal = {xxxxxx},
  year = {2024},
  doi={xxxxx}
}
```

## Usage

### Prerequisites
Ensure you have the following dependencies installed:

- **Python**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- **R**: `deSolve`, `dplyr`, `tidyverse`

### Running the Simulations

1. **Controlled System**  
   - Python:  
     ```bash
     python ./code/controlled_compute.py
     ```
   - R:  
     ```bash
     Rscript ./code/controlled_compute.R
     ```

2. **Open-loop System**  
   - Python:  
     ```bash
     python ./code/open_compute.py
     ```
   - R:  
     ```bash
     Rscript ./code/open_compute.R
     ```

### Generating Visualizations

Run the visualization scripts after generating simulation data.

- Controlled system visualization:  
  ```bash
  python ./code/controlled_vis.py
  ```
- Open-loop system visualization:  
  ```bash
  python ./code/open_vis.py
  ```

### Benchmarking and Analysis

Execute `benchmark.py` to run performance benchmarks and save results in `outputs/data/`.

### Outputs
Processed data files are saved in `outputs/data/`. Figures are saved in `outputs/figs/`.

### Available Figures

- `memory_usage_violin_plots.png`: Memory usage violin plot comparing Python and R implementations.
- `execution_time_violin_plots.png`: Execution time violin plot comparing Python and R implementations.
- `phase_portrait_py.png`: Phase portrait for the open-loop system in Python.
- `smd_response_py.png`: Response plots for the open-loop system in Python.
- `controlled_phase_portrait_py.png`: Phase portrait for the controlled system in Python.
- `controlled_smd_response_py.png`: Response plots for the controlled system in Python.

## License
This repository is licensed under the WTFPL.







