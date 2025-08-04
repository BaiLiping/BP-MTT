#!/usr/bin/env python3
"""
Performance comparison including the optimized PyTorch implementation.
Tests NumPy, Original PyTorch, and Optimized PyTorch versions.
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
sys.path.append(os.path.join(os.getcwd(), 'Torch_Optimized'))

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
        'surveillance_region': np.array([[-3000, -3000], [3000, 3000]], dtype=np.float32),
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


def run_torch_implementation(parameters: Dict, num_steps: int, num_runs: int = 3, 
                           device: str = 'cpu', optimized: bool = False) -> Dict:
    """Run PyTorch implementation and measure performance."""
    if not TORCH_AVAILABLE:
        return None
        
    impl_name = "Optimized PyTorch" if optimized else "Original PyTorch"
    print(f"Testing {impl_name} implementation on {device}...")
    
    if optimized:
        from Torch_Optimized.data_generator import Data_Generator as TorchDataGenerator
        from Torch_Optimized.bp_filter import BP_Filter as TorchBPFilter
    else:
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
        if optimized:
            true_tracks, all_measurements = data_gen.generate_simulation_data(torch_params, num_steps, batch_size=20)
        else:
            true_tracks, all_measurements = data_gen.generate_simulation_data(torch_params, num_steps)
        
        if device in ['cuda', 'mps']:
            if device == 'cuda':
                torch.cuda.synchronize()
            # MPS doesn't have synchronize, but operations should be complete
        
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
        
        if device in ['cuda', 'mps']:
            if device == 'cuda':
                torch.cuda.synchronize()
        
        tracking_time = time.time() - start_time
        times['tracking'].append(tracking_time)
        times['total'].append(data_gen_time + tracking_time)
        
        # Convert results to CPU for comparison
        cpu_true_tracks = true_tracks.cpu().numpy() if hasattr(true_tracks, 'cpu') else true_tracks
        cpu_cardinalities = [c.cpu().item() if hasattr(c, 'cpu') else c for c in estimated_cardinalities]
        
        results.append({
            'estimates': estimates,
            'estimated_cardinalities': cpu_cardinalities,
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


def plot_results(results_dict: Dict):
    """Plot comprehensive comparison results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract results
    implementations = list(results_dict.keys())
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Performance comparison
    ax = axes[0, 0]
    categories = ['Data Gen', 'Tracking', 'Total']
    x_pos = np.arange(len(categories))
    width = 0.15
    
    for i, (impl_name, result) in enumerate(results_dict.items()):
        if result is None:
            continue
        means = [result['stats']['data_gen']['mean'],
                result['stats']['tracking']['mean'],
                result['stats']['total']['mean']]
        stds = [result['stats']['data_gen']['std'],
               result['stats']['tracking']['std'],
               result['stats']['total']['std']]
        
        offset = (i - len(results_dict)/2) * width
        ax.bar(x_pos + offset, means, width, yerr=stds, 
               label=impl_name, alpha=0.7, color=colors[i])
    
    ax.set_xlabel('Operation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup comparison (relative to NumPy)
    ax = axes[0, 1]
    if 'NumPy' in results_dict:
        numpy_total = results_dict['NumPy']['stats']['total']['mean']
        speedups = []
        labels = []
        
        for impl_name, result in results_dict.items():
            if result is None or impl_name == 'NumPy':
                continue
            speedup = numpy_total / result['stats']['total']['mean']
            speedups.append(speedup)
            labels.append(impl_name.replace(' ', '\n'))
        
        bars = ax.bar(labels, speedups, alpha=0.7, color=colors[1:len(speedups)+1])
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='NumPy baseline')
        ax.set_ylabel('Speedup vs NumPy')
        ax.set_title('Relative Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{speedup:.2f}x', ha='center', va='bottom')
    
    # Memory transfer analysis (time per step)
    ax = axes[0, 2]
    for i, (impl_name, result) in enumerate(results_dict.items()):
        if result is None:
            continue
        time_per_step = [t / 100 for t in result['times']['tracking']]  # 100 steps
        ax.hist(time_per_step, alpha=0.6, label=impl_name, bins=10, color=colors[i])
    
    ax.set_xlabel('Time per Step (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Time per Step Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Variance analysis
    ax = axes[1, 0]
    variances = []
    labels = []
    for impl_name, result in results_dict.items():
        if result is None:
            continue
        variance = result['stats']['total']['std']
        variances.append(variance)
        labels.append(impl_name.replace(' ', '\n'))
    
    bars = ax.bar(labels, variances, alpha=0.7, color=colors[:len(variances)])
    ax.set_ylabel('Standard Deviation (seconds)')
    ax.set_title('Performance Consistency')
    ax.grid(True, alpha=0.3)
    
    # Efficiency metrics
    ax = axes[1, 1]
    if len(results_dict) >= 2:
        # Show data generation vs tracking time ratio
        ratios = []
        labels = []
        for impl_name, result in results_dict.items():
            if result is None:
                continue
            ratio = result['stats']['data_gen']['mean'] / result['stats']['tracking']['mean']
            ratios.append(ratio)
            labels.append(impl_name.replace(' ', '\n'))
        
        ax.bar(labels, ratios, alpha=0.7, color=colors[:len(ratios)])
        ax.set_ylabel('Data Gen / Tracking Time Ratio')
        ax.set_title('Resource Allocation')
        ax.grid(True, alpha=0.3)
    
    # Overall performance summary
    ax = axes[1, 2]
    if 'NumPy' in results_dict:
        summary_data = []
        summary_labels = []
        
        for impl_name, result in results_dict.items():
            if result is None:
                continue
            total_time = result['stats']['total']['mean']
            summary_data.append(total_time)
            summary_labels.append(impl_name)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(summary_labels))
        ax.barh(y_pos, summary_data, alpha=0.7, color=colors[:len(summary_data)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary_labels)
        ax.set_xlabel('Total Time (seconds)')
        ax.set_title('Overall Performance Summary')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (label, time) in enumerate(zip(summary_labels, summary_data)):
            ax.text(time + 0.01, i, f'{time:.3f}s', va='center')
    
    plt.tight_layout()
    plt.savefig('/Users/lipingb/Desktop/BP-MTT/performance_comparison_optimized.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function."""
    print("=== Comprehensive BP Multi-Target Tracking Performance Comparison ===")
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
    
    results_dict = {}
    
    # Run NumPy implementation
    results_dict['NumPy'] = run_numpy_implementation(parameters, num_steps, num_runs)
    
    # Run PyTorch implementations
    if TORCH_AVAILABLE:
        # Original PyTorch CPU
        results_dict['PyTorch CPU'] = run_torch_implementation(parameters, num_steps, num_runs, device='cpu', optimized=False)
        
        # Optimized PyTorch CPU
        results_dict['Optimized PyTorch CPU'] = run_torch_implementation(parameters, num_steps, num_runs, device='cpu', optimized=True)
        
        # GPU implementations
        if torch.cuda.is_available():
            results_dict['PyTorch CUDA'] = run_torch_implementation(parameters, num_steps, num_runs, device='cuda', optimized=False)
            results_dict['Optimized PyTorch CUDA'] = run_torch_implementation(parameters, num_steps, num_runs, device='cuda', optimized=True)
        elif torch.backends.mps.is_available():
            results_dict['PyTorch MPS'] = run_torch_implementation(parameters, num_steps, num_runs, device='mps', optimized=False)
            results_dict['Optimized PyTorch MPS'] = run_torch_implementation(parameters, num_steps, num_runs, device='mps', optimized=True)
    
    # Print results
    print("\n=== COMPREHENSIVE PERFORMANCE RESULTS ===")
    
    baseline_time = results_dict['NumPy']['stats']['total']['mean']
    
    for impl_name, result in results_dict.items():
        if result is None:
            continue
            
        print(f"\n{impl_name}:")
        print(f"  Data Generation: {result['stats']['data_gen']['mean']:.3f} ± {result['stats']['data_gen']['std']:.3f}s")
        print(f"  Tracking:        {result['stats']['tracking']['mean']:.3f} ± {result['stats']['tracking']['std']:.3f}s")
        print(f"  Total:           {result['stats']['total']['mean']:.3f} ± {result['stats']['total']['std']:.3f}s")
        print(f"  Time per step:   {result['stats']['tracking']['mean']/num_steps:.5f}s")
        
        if impl_name != 'NumPy':
            speedup = baseline_time / result['stats']['total']['mean']
            print(f"  Speedup vs NumPy: {speedup:.2f}x")
    
    # Create comprehensive visualization
    plot_results(results_dict)
    
    print(f"\nComprehensive comparison complete! Results saved to performance_comparison_optimized.png")
    
    return results_dict


if __name__ == "__main__":
    results = main()