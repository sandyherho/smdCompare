#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled Spring Mass Damper System Visualization
Sandy H. S. Herho <sandy.herho@email.ucr.edu>
23/12/2024
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting_style():
    """Setup publication quality plot styling"""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("deep")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 12),
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

def create_response_plots(data, output_dir):
    """Create and save system response plots"""
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    
    # Position tracking plot
    axs[0].plot(data['Time'], data['Desired_Position'], 'r--', label='Desired Position')
    axs[0].plot(data['Time'], data['Position'], 'b-', label='Actual Position')
    axs[0].set_ylabel('Position (m)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Control input plot
    axs[1].plot(data['Time'], data['Control_Input'], 'g-', label='Control Force')
    axs[1].set_ylabel('Force (N)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Error plot
    axs[2].plot(data['Time'], data['Error'], 'r-', label='Position Error')
    axs[2].set_ylabel('Error (m)')
    axs[2].grid(True)
    axs[2].legend()
    
    # Speed plot
    axs[3].plot(data['Time'], data['Speed'], 'purple', label='Speed')
    axs[3].set_ylabel('Speed (m/s)')
    axs[3].grid(True)
    axs[3].legend()
    
    # Acceleration plot
    axs[4].plot(data['Time'], data['Acceleration'], 'orange', label='Acceleration')
    axs[4].set_ylabel('Acceleration (m/s²)')
    axs[4].set_xlabel('Time (s)')
    axs[4].grid(True)
    axs[4].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'controlled_smd_response_py.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_phase_portrait(data, output_dir):
    """Create and save phase portrait"""
    plt.figure(figsize=(8, 8))
    plt.plot(data['Position'], data['Speed'], 'b-', label='Phase Portrait')
    plt.xlabel('Position (m)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'controlled_phase_portrait_py.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function"""
    try:
        # Setup paths
        base_dir = Path('../outputs')
        data_dir = base_dir / 'data'
        figs_dir = base_dir / 'figs'
        figs_dir.mkdir(parents=True, exist_ok=True)
        
        # Read simulation data
        data = pd.read_csv(data_dir / 'controlled_smd_simulation_py.csv')
        
        # Setup plotting style
        setup_plotting_style()
        
        # Create plots
        create_response_plots(data, figs_dir)
        create_phase_portrait(data, figs_dir)
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

if __name__ == '__main__':
    main()
