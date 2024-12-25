#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spring Mass Damper System Analysis with Mathematical Stability Criteria
Author: Sandy Herho 
Date: 12/23/2024

This script implements a Spring Mass Damper (SMD) system simulation
with comprehensive mathematical stability analysis using:
- State space representation
- Eigenvalue analysis
- Routh-Hurwitz stability criterion
- BIBO stability analysis
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pathlib import Path

class SMDSystem:
    """
    Spring Mass Damper System Class
    
    Implements a second-order mechanical system with:
    - Mass (m)
    - Spring stiffness (k)
    - Damping coefficient (d)
    
    System equation: mẍ + dẋ + kx = F(t)
    State space form: ẋ = Ax + Bu
    """
    
    def __init__(self, mass, stiffness, damping):
        """
        Initialize system parameters
        
        Args:
            mass (float): System mass in kg
            stiffness (float): Spring stiffness in N/m
            damping (float): Damping coefficient in Ns/m
        """
        self.m = mass          
        self.k = stiffness     
        self.d = damping       
        
        # Create state space matrices
        self.A = np.array([[0, 1], 
                          [-self.k/self.m, -self.d/self.m]])
        self.B = np.array([[0], 
                          [1/self.m]])

    @staticmethod
    def create_step_input(t, y_in, t_start, y_start, t_end, y_end):
        """
        Create a step input function with linear transition
        
        Args:
            t (ndarray): Time vector
            y_in (ndarray): Input vector to modify
            t_start (float): Start time of transition
            y_start (float): Initial value
            t_end (float): End time of transition
            y_end (float): Final value
            
        Returns:
            ndarray: Modified input vector with step transition
        """
        y = y_in.copy()
        step_size = t[1] - t[0]
        
        idx_start = np.where(t >= t_start)[0][0]
        idx_end = np.where(t >= t_end)[0][0]
        
        slope = (y_end - y_start) / (t_end - t_start)
        
        # Vectorized operations for efficiency
        transition_indices = np.arange(idx_start, idx_end + 1)
        y[transition_indices] = y_start + slope * (transition_indices - idx_start) * step_size
        y[idx_end:] = y_end
        
        return y

    def system_model(self, X, t, F_in):
        """
        Define system dynamics
        
        Implements the state space equation: ẋ = Ax + Bu
        
        Args:
            X (list): State vector [position, velocity]
            t (float): Current time
            F_in (float): Input force
            
        Returns:
            list: State derivatives [velocity, acceleration]
        """
        x1, x2 = X  # x1: position, x2: velocity
        
        # State equations
        dx1dt = x2
        dx2dt = (F_in - self.k * x1 - self.d * x2) / self.m
        
        return [dx1dt, dx2dt]

    def analyze_stability(self):
        """
        Perform mathematical stability analysis
        
        Analyzes system stability using:
        1. Eigenvalue analysis
        2. Characteristic equation
        3. Routh-Hurwitz criterion
        4. BIBO stability
        
        Returns:
            tuple: (metrics, interpretations) dictionaries
        """
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(self.A)
        is_stable = np.all(np.real(eigenvals) < 0)
        
        # Characteristic equation coefficients: s² + (d/m)s + (k/m) = 0
        char_poly = [1, self.d/self.m, self.k/self.m]
        
        # Routh-Hurwitz array first column
        routh_array = np.array([
            [char_poly[0], char_poly[2]],
            [char_poly[1], 0],
        ])
        routh_stable = np.all(routh_array[:, 0] > 0)
        
        # BIBO stability check
        bibo_stable = all(coef > 0 for coef in char_poly)
        
        metrics = {
            'Eigenvalues': [round(complex(ev).real, 3) + round(complex(ev).imag, 3)*1j 
                           for ev in eigenvals],
            'Characteristic Equation': char_poly,
            'Damping Factor': round(self.d/(2*np.sqrt(self.m*self.k)), 3),
            'Asymptotic Stability': 'Stable' if is_stable else 'Unstable',
            'Routh Stability': 'Stable' if routh_stable else 'Unstable',
            'BIBO Stability': 'Stable' if bibo_stable else 'Unstable'
        }
        
        interpretations = {
            'Eigenvalues': f"System poles at {metrics['Eigenvalues']}",
            'Characteristic Equation': (f"System characteristic equation: "
                                      f"s² + {char_poly[1]}s + {char_poly[2]} = 0"),
            'Damping Factor': f"System damping factor ζ = {metrics['Damping Factor']}",
            'Asymptotic Stability': (f"System is {metrics['Asymptotic Stability'].lower()} "
                                   f"(eigenvalue criterion)"),
            'Routh Stability': (f"System is {metrics['Routh Stability'].lower()} "
                               f"(Routh criterion)"),
            'BIBO Stability': (f"System is {metrics['BIBO Stability'].lower()} "
                              f"(BIBO criterion)")
        }
        
        return metrics, interpretations

    def simulate(self, sim_params):
        """
        Run time-domain simulation
        
        Args:
            sim_params (dict): Simulation parameters including:
                - start_time: Simulation start time
                - end_time: Simulation end time
                - step_size: Integration step size
                - input_start: Input transition start time
                - input_end: Input transition end time
                - input_start_val: Initial input value
                - input_end_val: Final input value
                
        Returns:
            dict: Simulation results containing time, force, position, speed, acceleration
        """
        # Create time vector
        t_span = np.linspace(
            sim_params['start_time'], 
            sim_params['end_time'],
            int((sim_params['end_time'] - sim_params['start_time'])/sim_params['step_size'] + 1)
        )
        
        num_steps = len(t_span)
        state_var = np.zeros((num_steps, 2))  # [position, velocity]
        acc = np.zeros(num_steps)             # acceleration
        
        # Create input force profile
        force_in = self.create_step_input(
            t_span, np.zeros(num_steps),
            sim_params['input_start'], sim_params['input_start_val'],
            sim_params['input_end'], sim_params['input_end_val']
        )
        
        # Simulate system response
        for i in range(num_steps - 1):
            t_current = [t_span[i], t_span[i+1]]
            solution = odeint(self.system_model, state_var[i], t_current, args=(force_in[i],))
            state_var[i+1] = solution[-1]
            # Calculate acceleration
            acc[i+1] = (force_in[i] - self.k * state_var[i+1,0] - 
                       self.d * state_var[i+1,1]) / self.m
        
        return {
            'time': t_span,
            'force': force_in,
            'position': state_var[:,0],
            'speed': state_var[:,1],
            'acceleration': acc
        }

    def save_results(self, results, metrics, output_dir='../outputs'):
        """
        Save simulation results and stability metrics
        
        Args:
            results (dict): Simulation results
            metrics (dict): Stability metrics
            output_dir (str): Output directory path
        """
        data_dir = Path(output_dir) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save simulation results
        sim_df = pd.DataFrame({
            'Time': results['time'],
            'Force': results['force'],
            'Position': results['position'],
            'Speed': results['speed'],
            'Acceleration': results['acceleration']
        })
        sim_df.to_csv(data_dir / 'smd_simulation_py.csv', index=False)
        
        # Save stability metrics
        metrics_for_csv = metrics.copy()
        metrics_for_csv['Eigenvalues'] = str(metrics_for_csv['Eigenvalues'])
        metrics_for_csv['Characteristic Equation'] = str(metrics_for_csv['Characteristic Equation'])
        
        metrics_df = pd.DataFrame(list(metrics_for_csv.items()), 
                                columns=['Criterion', 'Value'])
        metrics_df.to_csv(data_dir / 'smd_stability_metrics_py.csv', index=False)

def main():
    """Main execution function"""
    # System parameters
    smd_params = {
        'mass': 100,          # kg
        'stiffness': 50,      # N/m
        'damping': 50         # Ns/m
    }
    
    # Simulation parameters
    sim_params = {
        'start_time': 0,      # s
        'end_time': 100,      # s
        'step_size': 0.01,    # s
        'input_start': 4,     # s
        'input_end': 5,       # s
        'input_start_val': 0, # N
        'input_end_val': 50   # N
    }
    
    try:
        # Create system and run analysis
        system = SMDSystem(**smd_params)
        
        # Perform stability analysis
        metrics, interpretations = system.analyze_stability()
        
        # Run time-domain simulation
        results = system.simulate(sim_params)
        
        # Save all results
        system.save_results(results, metrics)
        
        # Print stability analysis
        print("\nMathematical Stability Analysis:")
        print("=" * 50)
        for criterion, value in metrics.items():
            print(f"\n{criterion}:")
            print(f"Value: {value}")
            print(f"Interpretation: {interpretations[criterion]}")
        
        # Print final simulation values
        print("\nSimulation Results:")
        print("=" * 50)
        print(f"Final position: {results['position'][-1]:.3f} m")
        print(f"Final velocity: {results['speed'][-1]:.3f} m/s")
        print(f"Final acceleration: {results['acceleration'][-1]:.3f} m/s²")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()
