#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import shapiro, kstest, levene, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
Path("../outputs/figs").mkdir(parents=True, exist_ok=True)
Path("../outputs/data").mkdir(parents=True, exist_ok=True)

# Set pandas display options for 3 decimal places
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def convert_to_mb(kb_value):
    """Convert KB to MB."""
    return round(kb_value / 1024.0, 3)

def load_data():
    """Load and preprocess the benchmark data."""
    try:
        controlled = pd.read_csv("../outputs/data/controlledSim.csv")
        open_data = pd.read_csv("../outputs/data/openSim.csv")
        
        # Convert memory from KB to MB
        controlled['memory_usage'] = controlled['memory_usage'].apply(convert_to_mb)
        open_data['memory_usage'] = open_data['memory_usage'].apply(convert_to_mb)
        
        # Round execution time to 3 decimals
        controlled['execution_time'] = controlled['execution_time'].round(3)
        open_data['execution_time'] = open_data['execution_time'].round(3)
        
        return {
            'controlled': {
                'py': controlled[controlled['script'].str.contains('py')],
                'r': controlled[controlled['script'].str.contains('R')]
            },
            'open': {
                'py': open_data[open_data['script'].str.contains('py')],
                'r': open_data[open_data['script'].str.contains('R')]
            }
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_violin_plots(data, metric):
    """Create violin plots with proper units."""
    try:
        plt.style.use('bmh')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ylabel = 'Execution Time (seconds)' if metric == 'execution_time' else 'Memory Usage (MB)'
        
        # Controlled plot
        controlled_data = pd.DataFrame({
            'Python': data['controlled']['py'][metric].round(3),
            'R': data['controlled']['r'][metric].round(3)
        })
        sns.violinplot(data=controlled_data, ax=ax1)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel('Controlled Implementation')
        
        # Format y-axis to 3 decimal places
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        # Open plot
        open_data = pd.DataFrame({
            'Python': data['open']['py'][metric].round(3),
            'R': data['open']['r'][metric].round(3)
        })
        sns.violinplot(data=open_data, ax=ax2)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel('Open Implementation')
        
        # Format y-axis to 3 decimal places
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        plt.tight_layout()
        plt.savefig(f'../outputs/figs/{metric}_violin_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating violin plots: {str(e)}")
        raise

def analyze_performance(data, metric):
    """Comprehensive performance analysis with interpretations."""
    try:
        results = []
        
        for impl in ['controlled', 'open']:
            for lang in ['py', 'r']:
                values = data[impl][lang][metric]
                
                # Basic statistics with 3 decimal rounding
                cv = round(np.std(values, ddof=1) / np.mean(values) * 100, 3)
                skewness = round(stats.skew(values), 3)
                kurtosis = round(stats.kurtosis(values), 3)
                
                basic_stats = {
                    'implementation': impl,
                    'language': lang,
                    'metric': metric,
                    'mean': round(np.mean(values), 3),
                    'std': round(np.std(values, ddof=1), 3),
                    'median': round(np.median(values), 3),
                    'cv': cv,
                    'cv_interpretation': interpret_coefficient_variation(cv),
                    'min': round(np.min(values), 3),
                    'max': round(np.max(values), 3),
                    'q1': round(np.percentile(values, 25), 3),
                    'q3': round(np.percentile(values, 75), 3),
                    'iqr': round(np.percentile(values, 75) - np.percentile(values, 25), 3),
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'distribution_shape': interpret_skewness_kurtosis(skewness, kurtosis)
                }
                
                # Distribution tests
                shapiro_stat, shapiro_p = shapiro(values)
                ks_norm_stat, ks_norm_p = kstest(values, 'norm')
                
                distribution_stats = {
                    'shapiro_stat': round(shapiro_stat, 3),
                    'shapiro_p': round(shapiro_p, 3),
                    'ks_norm_stat': round(ks_norm_stat, 3),
                    'ks_norm_p': round(ks_norm_p, 3),
                    'normality_interpretation': interpret_normality(shapiro_p, ks_norm_p)
                }
                
                # Anomaly detection with 3 decimal rounding
                z_scores = np.abs(stats.zscore(values))
                mad = stats.median_abs_deviation(values)
                modified_z = 0.6745 * (values - np.median(values)) / mad
                
                Q1, Q3 = np.percentile(values, [25, 75])
                IQR = Q3 - Q1
                
                zscore_mask = z_scores > 3
                iqr_mask = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
                mod_zscore_mask = np.abs(modified_z) > 3.5
                
                combined_mask = zscore_mask | iqr_mask | mod_zscore_mask
                anomaly_indices = np.where(combined_mask)[0]
                
                anomaly_stats = {
                    'anomaly_count': len(anomaly_indices),
                    'anomaly_percentage': round((len(anomaly_indices) / len(values) * 100), 3),
                    'anomaly_indices': json.dumps(sorted(anomaly_indices.tolist())),
                    'anomaly_interpretation': (
                        f"Found {len(anomaly_indices)} anomalies ({round(len(anomaly_indices) / len(values) * 100, 3)}% of data). "
                        f"This suggests {'unstable' if len(anomaly_indices) > len(values) * 0.05 else 'stable'} performance."
                    )
                }
                
                results.append({**basic_stats, **distribution_stats, **anomaly_stats})
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error in performance analysis: {str(e)}")
        raise

def compare_implementations(data, metric):
    """Compare Python vs R implementations with interpretations."""
    try:
        results = []
        
        for impl in ['controlled', 'open']:
            py_values = data[impl]['py'][metric]
            r_values = data[impl]['r'][metric]
            
            wilcox_stat, wilcox_p = wilcoxon(py_values, r_values)
            levene_stat, levene_p = levene(py_values, r_values)
            
            cohens_d = round((np.mean(py_values) - np.mean(r_values)) / np.sqrt(
                (np.var(py_values) + np.var(r_values)) / 2
            ), 3)
            
            mean_diff_percent = round(((np.mean(py_values) - np.mean(r_values)) / 
                               np.mean(r_values) * 100), 3)
            
            results.append({
                'implementation': impl,
                'metric': metric,
                'wilcoxon_stat': round(wilcox_stat, 3),
                'wilcoxon_p': round(wilcox_p, 3),
                'wilcoxon_interpretation': interpret_wilcoxon(wilcox_p, mean_diff_percent),
                'levene_stat': round(levene_stat, 3),
                'levene_p': round(levene_p, 3),
                'levene_interpretation': interpret_levene(levene_p),
                'cohens_d': cohens_d,
                'effect_size_interpretation': interpret_effect_size(cohens_d),
                'mean_diff_percent': mean_diff_percent,
                'overall_conclusion': f"""
                {'Statistically significant' if wilcox_p <= 0.05 else 'No significant'} difference between Python and R.
                Effect size is {interpret_effect_size(cohens_d).split()[0].lower()}.
                Performance variance is {'different' if levene_p <= 0.05 else 'similar'} between implementations.
                Python is {abs(mean_diff_percent):.3f}% {'faster' if mean_diff_percent < 0 else 'slower'} than R.
                """.strip()
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error in implementation comparison: {str(e)}")
        raise

def interpret_coefficient_variation(cv):
    """Interpret Coefficient of Variation."""
    if cv < 5:
        return "Very consistent performance (CV < 5%)"
    elif cv < 10:
        return "Good consistency (5% ≤ CV < 10%)"
    elif cv < 15:
        return "Moderate variability (10% ≤ CV < 15%)"
    else:
        return "High variability (CV ≥ 15%)"

def interpret_normality(shapiro_p, ks_p):
    """Interpret normality test results."""
    interpretations = []
    
    if shapiro_p > 0.05:
        interpretations.append("Shapiro-Wilk test suggests normal distribution (p > 0.05)")
    else:
        interpretations.append("Shapiro-Wilk test suggests non-normal distribution (p ≤ 0.05)")
        
    if ks_p > 0.05:
        interpretations.append("KS test confirms normality (p > 0.05)")
    else:
        interpretations.append("KS test suggests non-normal distribution (p ≤ 0.05)")
        
    return "; ".join(interpretations)

def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "Negligible effect size (|d| < 0.2)"
    elif abs_d < 0.5:
        return "Small effect size (0.2 ≤ |d| < 0.5)"
    elif abs_d < 0.8:
        return "Medium effect size (0.5 ≤ |d| < 0.8)"
    else:
        return "Large effect size (|d| ≥ 0.8)"

def interpret_wilcoxon(p_value, mean_diff_percent):
    """Interpret Wilcoxon test results."""
    if p_value > 0.05:
        return "No significant difference between Python and R implementations (p > 0.05)"
    else:
        direction = "faster/lower" if mean_diff_percent < 0 else "slower/higher"
        return f"Significant difference detected (p ≤ 0.05). Python is {abs(mean_diff_percent):.3f}% {direction} than R"

def interpret_levene(p_value):
    """Interpret Levene's test results."""
    if p_value > 0.05:
        return "Variances are homogeneous (p > 0.05)"
    else:
        return "Variances are significantly different (p ≤ 0.05)"

def interpret_skewness_kurtosis(skewness, kurtosis):
    """Interpret distribution shape metrics."""
    interpretations = []
    
    if abs(skewness) < 0.5:
        interpretations.append("Approximately symmetric distribution (|skewness| < 0.5)")
    else:
        direction = "right" if skewness > 0 else "left"
        magnitude = "moderately" if abs(skewness) < 1 else "heavily"
        interpretations.append(f"{magnitude.capitalize()} skewed to the {direction} (skewness = {skewness:.3f})")
    
    if abs(kurtosis) < 0.5:
        interpretations.append("Normal tail weight (|kurtosis| < 0.5)")
    elif kurtosis > 0:
        interpretations.append("Heavy-tailed distribution (kurtosis > 0.5)")
    else:
        interpretations.append("Light-tailed distribution (kurtosis < -0.5)")
    
    return "; ".join(interpretations)

def main():
    try:
        print("Starting performance analysis...")
        data = load_data()
        metrics = ['execution_time', 'memory_usage']
        
        for metric in metrics:
            print(f"\nAnalyzing {metric}...")
            
            # Create violin plots
            create_violin_plots(data, metric)
            print(f"Created violin plots for {metric}")
            
            # Perform analyses with interpretations
            performance_stats = analyze_performance(data, metric)
            comparison_stats = compare_implementations(data, metric)
            
            # Save results as separate CSVs with float_format for 3 decimals
            performance_stats.to_csv(
                f'../outputs/data/{metric}_performance_stats.csv', 
                index=False,
                float_format='%.3f'
            )
            comparison_stats.to_csv(
                f'../outputs/data/{metric}_comparison_stats.csv', 
                index=False,
                float_format='%.3f'
            )
            
            # Save to Excel with float_format for 3 decimals
            with pd.ExcelWriter(
                '../outputs/data/performance_analysis.xlsx',
                engine='openpyxl'
            ) as writer:
                performance_stats.to_excel(
                    writer, 
                    sheet_name=f'{metric}_performance',
                    index=False,
                    float_format='%.3f'
                )
                comparison_stats.to_excel(
                    writer, 
                    sheet_name=f'{metric}_comparison',
                    index=False,
                    float_format='%.3f'
                )
        
        print("\nAnalysis completed successfully. Results saved in:")
        print("1. Visualizations: ../outputs/figs/")
        print("2. CSV files:")
        print("   - execution_time_performance_stats.csv")
        print("   - execution_time_comparison_stats.csv")
        print("   - memory_usage_performance_stats.csv")
        print("   - memory_usage_comparison_stats.csv")
        print("3. Excel file: performance_analysis.xlsx")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
