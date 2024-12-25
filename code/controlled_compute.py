#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spring Mass Damper System with Simple PID Control
Author: Sandy Herho (Modified)
Date: 12/23/2024
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pathlib import Path

class SMDSystem:
    """Base Spring Mass Damper System Class"""
    
    def __init__(self, mass, stiffness, damping):
        """Initialize system parameters"""
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
        """Create a step input function with linear transition"""
        y = y_in.copy()
        step_size = t[1] - t[0]
        
        idx_start = np.where(t >= t_start)[0][0]
        idx_end = np.where(t >= t_end)[0][0]
        
        slope = (y_end - y_start) / (t_end - t_start)
        
        transition_indices = np.arange(idx_start, idx_end + 1)
        y[transition_indices] = y_start + slope * (transition_indices - idx_start) * step_size
        y[idx_end:] = y_end
        
        return y

    def system_model(self, X, t, F_in):
        """Define system dynamics"""
        x1, x2 = X
        dx1dt = x2
        dx2dt = (F_in - self.k * x1 - self.d * x2) / self.m
        return [dx1dt, dx2dt]

class PIDController:
    """Simple PID Controller"""
    
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
        
    def reset(self):
        """Reset controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def compute(self, setpoint, measurement, time):
        """Compute PID control action"""
        error = setpoint - measurement
        
        if self.last_time is None:
            self.last_time = time
            self.last_error = error
            return 0.0
        
        dt = time - self.last_time
        if dt <= 0:
            return 0.0
            
        # Integral term
        self.integral += error * dt
        
        # Derivative term
        derivative = (error - self.last_error) / dt
        
        # PID output
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # Update states
        self.last_error = error
        self.last_time = time
        
        return output

class SMDSystemWithPID(SMDSystem):
    """Controlled Spring Mass Damper System"""
    
    def __init__(self, mass, stiffness, damping):
        super().__init__(mass, stiffness, damping)
        
        # Initialize PID controller with adjusted gains
        self.controller = PIDController(
            kp=200.0,    # Proportional gain
            ki=50.0,     # Integral gain
            kd=100.0     # Derivative gain
        )

    def analyze_closed_loop_stability(self):
        """Perform mathematical stability analysis of the closed-loop system"""
        char_poly = [
            1,  # s³ coefficient
            (self.d + self.controller.kd)/self.m,  # s² coefficient
            (self.k + self.controller.kp)/self.m,  # s¹ coefficient
            self.controller.ki/self.m  # s⁰ coefficient
        ]
        
        A_cl = np.array([
            [0, 1, 0],
            [-self.k/self.m, -self.d/self.m, 0],
            [-self.controller.kp/self.m, -self.controller.kd/self.m, -self.controller.ki/self.m]
        ])
        
        eigenvals = np.linalg.eigvals(A_cl)
        is_stable = np.all(np.real(eigenvals) < 0)
        
        routh_array = np.array([
            [char_poly[0], char_poly[2]],
            [char_poly[1], char_poly[3]],
            [(char_poly[1]*char_poly[2] - char_poly[0]*char_poly[3])/char_poly[1], 0],
            [char_poly[3], 0]
        ])
        routh_stable = np.all(routh_array[:, 0] > 0)
        
        bibo_stable = all(coef > 0 for coef in char_poly)
        
        metrics = {
            'Eigenvalues': [round(complex(ev).real, 3) + round(complex(ev).imag, 3)*1j 
                           for ev in eigenvals],
            'Characteristic Equation': char_poly,
            'Natural Frequency': round(np.sqrt((self.k + self.controller.kp)/self.m), 3),
            'Damping Ratio': round((self.d + self.controller.kd)/(2*np.sqrt(self.m*(self.k + self.controller.kp))), 3),
            'Asymptotic Stability': 'Stable' if is_stable else 'Unstable',
            'Routh Stability': 'Stable' if routh_stable else 'Unstable',
            'BIBO Stability': 'Stable' if bibo_stable else 'Unstable'
        }
        
        interpretations = {
            'Eigenvalues': f"Closed-loop poles at {metrics['Eigenvalues']}",
            'Characteristic Equation': (f"Characteristic equation: "
                                      f"s³ + {char_poly[1]:.3f}s² + {char_poly[2]:.3f}s + {char_poly[3]:.3f} = 0"),
            'Natural Frequency': f"Natural frequency ωn = {metrics['Natural Frequency']} rad/s",
            'Damping Ratio': f"Damping ratio ζ = {metrics['Damping Ratio']}",
            'Asymptotic Stability': f"System is {metrics['Asymptotic Stability'].lower()} (eigenvalue criterion)",
            'Routh Stability': f"System is {metrics['Routh Stability'].lower()} (Routh criterion)",
            'BIBO Stability': f"System is {metrics['BIBO Stability'].lower()} (BIBO criterion)"
        }
        
        return metrics, interpretations
        
    def system_model_with_control(self, X, t, target_position):
        """System dynamics with control"""
        position, velocity = X
        
        control_force = self.controller.compute(
            setpoint=target_position,
            measurement=position,
            time=t
        )
        
        dx1dt = velocity
        dx2dt = (control_force - self.k * position - self.d * velocity) / self.m
        
        return [dx1dt, dx2dt]
        
    def simulate_with_control(self, sim_params):
        """Run controlled simulation"""
        self.controller.reset()
        
        t_span = np.linspace(
            sim_params['start_time'],
            sim_params['end_time'],
            int((sim_params['end_time'] - sim_params['start_time'])/sim_params['step_size'] + 1)
        )
        
        num_steps = len(t_span)
        state_var = np.zeros((num_steps, 2))
        control_actions = np.zeros(num_steps)
        accelerations = np.zeros(num_steps)
        
        desired_positions = self.create_step_input(
            t_span, np.zeros(num_steps),
            sim_params['input_start'], sim_params['input_start_val'],
            sim_params['input_end'], sim_params['input_end_val']
        )
        
        for i in range(num_steps - 1):
            t_current = [t_span[i], t_span[i+1]]
            
            control_actions[i] = self.controller.compute(
                desired_positions[i], state_var[i,0], t_span[i]
            )
            
            solution = odeint(
                self.system_model, 
                state_var[i], 
                t_current, 
                args=(control_actions[i],)
            )
            state_var[i+1] = solution[-1]
            
            accelerations[i] = (control_actions[i] - self.k * state_var[i,0] - 
                              self.d * state_var[i,1]) / self.m
        
        control_actions[-1] = self.controller.compute(
            desired_positions[-1], state_var[-1,0], t_span[-1]
        )
        accelerations[-1] = (control_actions[-1] - self.k * state_var[-1,0] - 
                           self.d * state_var[-1,1]) / self.m
        
        position_error = desired_positions - state_var[:,0]
        
        return {
            'time': t_span,
            'desired_position': desired_positions,
            'position': state_var[:,0],
            'speed': state_var[:,1],
            'acceleration': accelerations,
            'control_input': control_actions,
            'error': position_error
        }

    def calculate_performance_metrics(self, results):
        """Calculate performance metrics"""
        final_error = abs(results['error'][-1])
        max_overshoot = max(0, max(results['position']) - results['desired_position'][-1])
        max_control = max(abs(results['control_input']))
        rms_error = np.sqrt(np.mean(results['error']**2))
        
        steady_state = results['desired_position'][-1]
        settling_threshold = 0.02 * steady_state
        settling_time = None
        
        for i, pos in enumerate(results['position']):
            if abs(pos - steady_state) <= settling_threshold:
                if all(abs(p - steady_state) <= settling_threshold 
                       for p in results['position'][i:]):
                    settling_time = results['time'][i]
                    break
        
        metrics = {
            'Final Position Error (m)': final_error,
            'Maximum Overshoot (m)': max_overshoot,
            'Settling Time (s)': settling_time if settling_time else 'Not reached',
            'Maximum Control Force (N)': max_control,
            'RMS Error (m)': rms_error,
            'Steady State Value (m)': results['position'][-1],
            'Maximum Speed (m/s)': max(abs(results['speed'])),
            'Maximum Acceleration (m/s²)': max(abs(results['acceleration']))
        }
        
        return metrics

    def save_results(self, results, metrics, stability_metrics, stability_interpretations, output_dir='../outputs'):
        """Save all results"""
        data_dir = Path(output_dir) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        sim_df = pd.DataFrame({
            'Time': results['time'],
            'Desired_Position': results['desired_position'],
            'Position': results['position'],
            'Speed': results['speed'],
            'Acceleration': results['acceleration'],
            'Control_Input': results['control_input'],
            'Error': results['error']
        })
        sim_df.to_csv(data_dir / 'controlled_smd_simulation_py.csv', index=False)
        
        metrics_df = pd.DataFrame(list(metrics.items()), 
                                columns=['Metric', 'Value'])
        metrics_df.to_csv(data_dir / 'controlled_smd_metrics_py.csv', index=False)
        
        stability_metrics_df = pd.DataFrame(list(stability_metrics.items()),
                                          columns=['Criterion', 'Value'])
        stability_metrics_df.to_csv(data_dir / 'controlled_smd_stability_metrics_py.csv', index=False)
        
        stability_interp_df = pd.DataFrame(list(stability_interpretations.items()),
                                         columns=['Criterion', 'Interpretation'])
        stability_interp_df.to_csv(data_dir / 'controlled_smd_stability_interpretations_py.csv', index=False)

def main():
    """Main execution function"""
    smd_params = {
        'mass': 100,          # kg
        'stiffness': 50,      # N/m
        'damping': 50         # Ns/m
    }
    
    sim_params = {
        'start_time': 0,      # s
        'end_time': 20,       # s
        'step_size': 0.01,    # s
        'input_start': 2,     # s
        'input_end': 2.5,     # s
        'input_start_val': 0, # m (desired position)
        'input_end_val': 1    # m (desired position)
    }
    
    try:
        # Create controlled system
        system = SMDSystemWithPID(**smd_params)
        
        # Perform stability analysis
        stability_metrics, stability_interpretations = system.analyze_closed_loop_stability()
        
        # Run simulation
        results = system.simulate_with_control(sim_params)
        
        # Calculate performance metrics
        performance_metrics = system.calculate_performance_metrics(results)
        
        # Save all results
        system.save_results(results, performance_metrics, 
                          stability_metrics, stability_interpretations)
        
        # Print stability analysis
        print("\nClosed-Loop Stability Analysis:")
        print("=" * 50)
        for criterion, interpretation in stability_interpretations.items():
            print(f"\n{criterion}:")
            print(f"Value: {stability_metrics[criterion]}")
            print(f"Interpretation: {interpretation}")
            
        # Print performance metrics
        print("\nControl System Performance:")
        print("=" * 50)
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == '__main__':
    main()
