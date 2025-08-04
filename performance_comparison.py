#!/usr/bin/env python3
"""
Performance comparison between PyTorch and NumPy implementations of BP-based multi-target tracking.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add implementation directories to path
sys.path.append(os.path.join(os.getcwd(), 'Torch_Implementation'))
sys.path.append(os.path.join(os.getcwd(), 'NumPy_Implementation'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, will only test NumPy implementation")


def get_parameters(device='cpu'):
    """Get simulation parameters."""
    num_steps = 100  # Reduced for faster comparison
    
    parameters = {
        'surveillance_region': np.array([[-3000, 3000], [-3000, 3000]], dtype=np.float32),
        'prior_velocity_covariance': np.diag([10**2, 10**2]).astype(np.float32),
        'driving_noise_variance': 0.010,
        'length_steps': np.ones(num_steps, dtype=np.float32),
        'survival_probability': 0.999,
        
        'num_sensors': 2,
        'measurement_variance_range': 25**2,
        'measurement_variance_bearing': 0.5**2,
        'detection_probability': 0.9,
        'measurement_range': 2 * 3000,
        'mean_clutter': 5,
        'clutter_distribution': 1 / (360 * 2 * 3000),
        
        'detection_threshold': 0.5,
        'threshold_pruning': 1e-4,
        'minimum_track_length': 1,
        'num_particles': 5000  # Reduced for faster comparison
    }
    
    return parameters, num_steps


def run_numpy_implementation(parameters: Dict, num_steps: int, num_runs: int = 3) -> Dict:
    """Run NumPy implementation and measure performance."""
    print("Testing NumPy implementation...")
    
    from NumPy_Implementation.data_generator import Data_Generator as NumpyDataGenerator
    from NumPy_Implementation.bp_filter import BP_Filter as NumpyBPFilter
    
    times = {'data_gen': [], 'tracking': [], 'total': []}
    results = []
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        
        # Set seed for reproducibility
        np.random.seed(run)
        
        # Initialize
        data_gen = NumpyDataGenerator(random_seed=run)
        parameters['target_start_states'] = data_gen.get_start_states(5, 1000, 10)
        parameters['target_appearance_from_to'] = np.array([
            [5, 55], [10, 60], [15, 65], [20, 70], [25, 75]
        ], dtype=np.float32).T
        parameters['sensor_positions'] = data_gen.get_sensor_positions(parameters['num_sensors'], 5000)
        
        # Data generation
        start_time = time.time()
        true_tracks, all_measurements = data_gen.generate_simulation_data(parameters, num_steps)
        data_gen_time = time.time() - start_time
        times['data_gen'].append(data_gen_time)
        
        # Tracking
        bp_filter = NumpyBPFilter(parameters)
        unknown_number = 0.01
        unknown_particles = np.random.rand(2, parameters['num_particles']) * 6000 - 3000
        
        start_time = time.time()
        estimates = []
        estimated_cardinalities = []
        
        for step in range(num_steps):
            result = bp_filter.track_step(all_measurements[step], unknown_number, unknown_particles, step)
            estimates.append(result['estimates'])
            estimated_cardinalities.append(result['estimated_cardinality'])
        
        tracking_time = time.time() - start_time
        times['tracking'].append(tracking_time)
        times['total'].append(data_gen_time + tracking_time)
        
        results.append({
            'estimates': estimates,
            'estimated_cardinalities': estimated_cardinalities,
            'true_tracks': true_tracks
        })
    
    # Compute statistics
    stats = {}
    for key in times:
        stats[key] = {
            'mean': np.mean(times[key]),
            'std': np.std(times[key]),
            'min': np.min(times[key]),
            'max': np.max(times[key])
        }
    
    return {'times': times, 'stats': stats, 'results': results}


def run_torch_implementation(parameters: Dict, num_steps: int, num_runs: int = 3, device: str = 'cpu') -> Dict:
    """Run PyTorch implementation and measure performance."""
    if not TORCH_AVAILABLE:
        return None
        
    print(f"Testing PyTorch implementation on {device}...")
    
    from Torch_Implementation.data_generator import Data_Generator as TorchDataGenerator
    from Torch_Implementation.bp_filter import BP_Filter as TorchBPFilter
    
    times = {'data_gen': [], 'tracking': [], 'total': []}
    results = []
    
    # Convert parameters to torch tensors
    torch_params = parameters.copy()
    torch_params['surveillance_region'] = torch.tensor(parameters['surveillance_region'], device=device)
    torch_params['prior_velocity_covariance'] = torch.tensor(parameters['prior_velocity_covariance'], device=device)
    torch_params['length_steps'] = torch.tensor(parameters['length_steps'], device=device)
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        
        # Set seed for reproducibility
        torch.manual_seed(run)
        np.random.seed(run)
        
        # Initialize
        data_gen = TorchDataGenerator(device=device)
        torch_params['target_start_states'] = data_gen.get_start_states(5, 1000, 10)
        torch_params['target_appearance_from_to'] = torch.tensor([
            [5, 55], [10, 60], [15, 65], [20, 70], [25, 75]
        ], dtype=torch.float32, device=device).T
        torch_params['sensor_positions'] = data_gen.get_sensor_positions(torch_params['num_sensors'], 5000)
        
        # Data generation
        start_time = time.time()
        true_tracks, all_measurements = data_gen.generate_simulation_data(torch_params, num_steps)
        if device == 'cuda':
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        data_gen_time = time.time() - start_time
        times['data_gen'].append(data_gen_time)
        
        # Tracking
        bp_filter = TorchBPFilter(torch_params, device=device)
        unknown_number = 0.01
        unknown_particles = torch.rand(2, torch_params['num_particles'], device=device) * 6000 - 3000
        
        start_time = time.time()
        estimates = []
        estimated_cardinalities = []
        
        for step in range(num_steps):
            result = bp_filter.track_step(all_measurements[step], unknown_number, unknown_particles, step)
            estimates.append(result['estimates'])
            estimated_cardinalities.append(result['estimated_cardinality'])
        
        if device == 'cuda':
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        tracking_time = time.time() - start_time
        times['tracking'].append(tracking_time)
        times['total'].append(data_gen_time + tracking_time)
        
        # Convert results to CPU for comparison
        cpu_true_tracks = true_tracks.cpu().numpy() if hasattr(true_tracks, 'cpu') else true_tracks
        results.append({
            'estimates': estimates,
            'estimated_cardinalities': estimated_cardinalities,
            'true_tracks': cpu_true_tracks
        })
    
    # Compute statistics
    stats = {}
    for key in times:
        stats[key] = {
            'mean': np.mean(times[key]),
            'std': np.std(times[key]),
            'min': np.min(times[key]),
            'max': np.max(times[key])
        }
    
    return {'times': times, 'stats': stats, 'results': results}


def compare_accuracy(numpy_results: Dict, torch_results: Dict) -> Dict:
    """Compare accuracy between implementations."""
    if torch_results is None:
        return {}
        
    print("Comparing accuracy...")
    
    numpy_cards = []
    torch_cards = []
    
    # Collect cardinalities from all runs
    for result in numpy_results['results']:
        numpy_cards.extend(result['estimated_cardinalities'])
    
    for result in torch_results['results']:
        torch_cards.extend(result['estimated_cardinalities'])
    
    # Compute metrics
    mse = np.mean((np.array(numpy_cards) - np.array(torch_cards))**2)
    mae = np.mean(np.abs(np.array(numpy_cards) - np.array(torch_cards)))
    correlation = np.corrcoef(numpy_cards, torch_cards)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'numpy_mean_card': np.mean(numpy_cards),
        'torch_mean_card': np.mean(torch_cards)
    }


def plot_results(numpy_results: Dict, torch_results: Dict = None, accuracy: Dict = None):
    """Plot comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Performance comparison
    ax = axes[0, 0]
    categories = ['Data Gen', 'Tracking', 'Total']
    numpy_means = [numpy_results['stats']['data_gen']['mean'],
                   numpy_results['stats']['tracking']['mean'],
                   numpy_results['stats']['total']['mean']]
    numpy_stds = [numpy_results['stats']['data_gen']['std'],
                  numpy_results['stats']['tracking']['std'],
                  numpy_results['stats']['total']['std']]
    
    x_pos = np.arange(len(categories))
    bars1 = ax.bar(x_pos - 0.2, numpy_means, 0.4, yerr=numpy_stds, label='NumPy', alpha=0.7)
    
    if torch_results:
        torch_means = [torch_results['stats']['data_gen']['mean'],
                       torch_results['stats']['tracking']['mean'],
                       torch_results['stats']['total']['mean']]
        torch_stds = [torch_results['stats']['data_gen']['std'],
                      torch_results['stats']['tracking']['std'],
                      torch_results['stats']['total']['std']]
        bars2 = ax.bar(x_pos + 0.2, torch_means, 0.4, yerr=torch_stds, label='PyTorch', alpha=0.7)
    
    ax.set_xlabel('Operation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup ratio (if both implementations available)
    if torch_results:
        ax = axes[0, 1]
        speedups = [numpy_results['stats'][key]['mean'] / torch_results['stats'][key]['mean'] 
                   for key in ['data_gen', 'tracking', 'total']]
        bars = ax.bar(categories, speedups, alpha=0.7, color='green')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax.set_ylabel('Speedup Ratio (NumPy/PyTorch)')
        ax.set_title('PyTorch Speedup over NumPy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{speedup:.2f}x', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'PyTorch not available', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Speedup Comparison')
    
    # Time per step comparison
    ax = axes[1, 0]
    numpy_time_per_step = [t / 100 for t in numpy_results['times']['tracking']]  # 100 steps
    ax.hist(numpy_time_per_step, alpha=0.7, label='NumPy', bins=10)
    
    if torch_results:
        torch_time_per_step = [t / 100 for t in torch_results['times']['tracking']]
        ax.hist(torch_time_per_step, alpha=0.7, label='PyTorch', bins=10)
    
    ax.set_xlabel('Time per Step (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Time per Step Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy comparison (if available)
    if accuracy and torch_results:
        ax = axes[1, 1]
        numpy_cards = []
        torch_cards = []
        for result in numpy_results['results']:
            numpy_cards.extend(result['estimated_cardinalities'])
        for result in torch_results['results']:
            torch_cards.extend(result['estimated_cardinalities'])
        
        ax.scatter(numpy_cards, torch_cards, alpha=0.6)
        min_val = min(min(numpy_cards), min(torch_cards))
        max_val = max(max(numpy_cards), max(torch_cards))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
        ax.set_xlabel('NumPy Estimated Cardinality')
        ax.set_ylabel('PyTorch Estimated Cardinality')
        ax.set_title(f'Accuracy Comparison (r={accuracy["correlation"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Accuracy comparison\nnot available', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Accuracy Comparison')
    
    plt.tight_layout()
    plt.savefig('/Users/lipingb/Desktop/BP-MTT/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function."""
    print("=== BP Multi-Target Tracking Performance Comparison ===")
    print()
    
    # Get parameters
    parameters, num_steps = get_parameters()
    num_runs = 3
    
    print(f"Configuration:")
    print(f"  Steps: {num_steps}")
    print(f"  Particles: {parameters['num_particles']}")
    print(f"  Sensors: {parameters['num_sensors']}")
    print(f"  Runs: {num_runs}")
    print()
    
    # Run NumPy implementation
    numpy_results = run_numpy_implementation(parameters, num_steps, num_runs)
    
    # Run PyTorch implementation(s)
    torch_cpu_results = None
    torch_gpu_results = None
    
    if TORCH_AVAILABLE:
        torch_cpu_results = run_torch_implementation(parameters, num_steps, num_runs, device='cpu')
        
        if torch.cuda.is_available():
            torch_gpu_results = run_torch_implementation(parameters, num_steps, num_runs, device='cuda')
    
    # Print results
    print("\n=== PERFORMANCE RESULTS ===")
    print(f"NumPy Implementation:")
    print(f"  Data Generation: {numpy_results['stats']['data_gen']['mean']:.3f} ± {numpy_results['stats']['data_gen']['std']:.3f}s")
    print(f"  Tracking:        {numpy_results['stats']['tracking']['mean']:.3f} ± {numpy_results['stats']['tracking']['std']:.3f}s")
    print(f"  Total:           {numpy_results['stats']['total']['mean']:.3f} ± {numpy_results['stats']['total']['std']:.3f}s")
    print(f"  Time per step:   {numpy_results['stats']['tracking']['mean']/num_steps:.5f}s")
    
    if torch_cpu_results:
        print(f"\nPyTorch CPU Implementation:")
        print(f"  Data Generation: {torch_cpu_results['stats']['data_gen']['mean']:.3f} ± {torch_cpu_results['stats']['data_gen']['std']:.3f}s")
        print(f"  Tracking:        {torch_cpu_results['stats']['tracking']['mean']:.3f} ± {torch_cpu_results['stats']['tracking']['std']:.3f}s")
        print(f"  Total:           {torch_cpu_results['stats']['total']['mean']:.3f} ± {torch_cpu_results['stats']['total']['std']:.3f}s")
        print(f"  Time per step:   {torch_cpu_results['stats']['tracking']['mean']/num_steps:.5f}s")
        print(f"  Speedup:         {numpy_results['stats']['total']['mean']/torch_cpu_results['stats']['total']['mean']:.2f}x")
    
    if torch_gpu_results:
        print(f"\nPyTorch GPU Implementation:")
        print(f"  Data Generation: {torch_gpu_results['stats']['data_gen']['mean']:.3f} ± {torch_gpu_results['stats']['data_gen']['std']:.3f}s")
        print(f"  Tracking:        {torch_gpu_results['stats']['tracking']['mean']:.3f} ± {torch_gpu_results['stats']['tracking']['std']:.3f}s")
        print(f"  Total:           {torch_gpu_results['stats']['total']['mean']:.3f} ± {torch_gpu_results['stats']['total']['std']:.3f}s")
        print(f"  Time per step:   {torch_gpu_results['stats']['tracking']['mean']/num_steps:.5f}s")
        print(f"  Speedup:         {numpy_results['stats']['total']['mean']/torch_gpu_results['stats']['total']['mean']:.2f}x")
    
    # Compare accuracy
    accuracy_cpu = compare_accuracy(numpy_results, torch_cpu_results) if torch_cpu_results else {}
    accuracy_gpu = compare_accuracy(numpy_results, torch_gpu_results) if torch_gpu_results else {}
    
    if accuracy_cpu:
        print(f"\nAccuracy Comparison (NumPy vs PyTorch CPU):")
        print(f"  MSE:              {accuracy_cpu['mse']:.6f}")
        print(f"  MAE:              {accuracy_cpu['mae']:.6f}")
        print(f"  Correlation:      {accuracy_cpu['correlation']:.6f}")
    
    if accuracy_gpu:
        print(f"\nAccuracy Comparison (NumPy vs PyTorch GPU):")
        print(f"  MSE:              {accuracy_gpu['mse']:.6f}")
        print(f"  MAE:              {accuracy_gpu['mae']:.6f}")
        print(f"  Correlation:      {accuracy_gpu['correlation']:.6f}")
    
    # Create visualizations
    if torch_gpu_results:
        plot_results(numpy_results, torch_gpu_results, accuracy_gpu)
    elif torch_cpu_results:
        plot_results(numpy_results, torch_cpu_results, accuracy_cpu)
    else:
        plot_results(numpy_results)
    
    print(f"\nComparison complete! Results saved to performance_comparison.png")
    
    return {
        'numpy': numpy_results,
        'torch_cpu': torch_cpu_results,
        'torch_gpu': torch_gpu_results,
        'accuracy_cpu': accuracy_cpu,
        'accuracy_gpu': accuracy_gpu
    }


if __name__ == "__main__":
    results = main()