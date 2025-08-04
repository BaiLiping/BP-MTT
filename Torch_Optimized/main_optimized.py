#!/usr/bin/env python3
"""
Memory-optimized PyTorch implementation of BP-based multitarget tracking.
Minimizes CPU-GPU transfers and uses vectorized operations for better performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_generator import Data_Generator
from bp_filter import BP_Filter
import time


def main():
    """Main function to test the optimized Data_Generator and BP_Filter classes."""
    
    # Set random seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    # Device selection with preference for available accelerators
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = 'mps'  
        print("Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Model parameters (keep as much on GPU as possible)
    num_steps = 200
    parameters = {
        'surveillance_region': torch.tensor([[-3000, -3000], [3000, 3000]], dtype=torch.float32, device=device),
        'prior_velocity_covariance': torch.diag(torch.tensor([10**2, 10**2], dtype=torch.float32, device=device)),
        'driving_noise_variance': 0.010,
        'length_steps': torch.ones(num_steps, dtype=torch.float32, device=device),
        'survival_probability': 0.999,
        
        'num_sensors': 2,
        'measurement_variance_range': 25**2,
        'measurement_variance_bearing': 0.5**2,
        'detection_probability': 0.9,
        'measurement_range': 2 * 3000,
        'mean_clutter': 5,
        'clutter_distribution': 1 / (360 * 2 * 3000),
        
        # Algorithm parameters
        'detection_threshold': 0.5,
        'threshold_pruning': 1e-4,
        'minimum_track_length': 1,
        'num_particles': 10000
    }
    
    # Initialize data generator
    data_gen = Data_Generator(device=device)
    
    # Generate target start states and appearance times (keep on GPU)
    parameters['target_start_states'] = data_gen.get_start_states(5, 1000, 10)
    parameters['target_appearance_from_to'] = torch.tensor([
        [5, 155], [10, 160], [15, 165], [20, 170], [25, 175]
    ], dtype=torch.float32, device=device).T
    
    # Generate sensor positions
    parameters['sensor_positions'] = data_gen.get_sensor_positions(parameters['num_sensors'], 5000)
    
    print("Generating simulation data...")
    start_time = time.time()
    
    # Generate true tracks and measurements with batched processing
    true_tracks, all_measurements = data_gen.generate_simulation_data(parameters, num_steps, batch_size=20)
    
    if device in ['cuda', 'mps']:
        torch.cuda.synchronize() if device == 'cuda' else None
    
    data_gen_time = time.time() - start_time
    print(f"Data generation completed in {data_gen_time:.2f} seconds")
    
    # Initialize BP filter
    bp_filter = BP_Filter(parameters, device=device)
    
    # Initialize unknown target parameters (keep on GPU)
    unknown_number = 0.01
    unknown_particles = torch.zeros(2, parameters['num_particles'], device=device, dtype=torch.float32)
    surveillance_region = parameters['surveillance_region']
    
    # Use device-specific random generation
    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    
    unknown_particles[0, :] = (torch.rand(parameters['num_particles'], device=device, generator=generator) * 
                              (surveillance_region[1, 0] - surveillance_region[0, 0]) + 
                              surveillance_region[0, 0])
    unknown_particles[1, :] = (torch.rand(parameters['num_particles'], device=device, generator=generator) * 
                              (surveillance_region[1, 1] - surveillance_region[0, 1]) + 
                              surveillance_region[0, 1])
    
    # Tracking loop
    print("Starting optimized BP tracking...")
    start_time = time.time()
    
    estimates = []
    estimated_cardinalities = []
    
    # Pre-allocate result lists to avoid repeated resizing
    estimates.reserve = num_steps
    estimated_cardinalities.reserve = num_steps
    
    for step in range(num_steps):
        if step % 50 == 0:
            print(f"Processing step {step}/{num_steps}")
        
        # Perform tracking step
        result = bp_filter.track_step(
            all_measurements[step], 
            unknown_number, 
            unknown_particles, 
            step
        )
        
        estimates.append(result['estimates'])
        # Keep cardinality on GPU until final collection
        estimated_cardinalities.append(result['estimated_cardinality'])
    
    if device in ['cuda', 'mps']:
        torch.cuda.synchronize() if device == 'cuda' else None
    
    tracking_time = time.time() - start_time
    print(f"Tracking completed in {tracking_time:.2f} seconds")
    
    # Convert GPU results to CPU only when needed for analysis
    cpu_cardinalities = [card.cpu().item() if hasattr(card, 'cpu') else card for card in estimated_cardinalities]
    
    # Analysis and visualization
    print("\n=== Tracking Results ===")
    print(f"Total simulation steps: {num_steps}")
    print(f"Number of sensors: {parameters['num_sensors']}")
    print(f"Number of particles: {parameters['num_particles']}")
    print(f"Device: {device}")
    print(f"Mean estimated cardinality: {np.mean(cpu_cardinalities):.2f}")
    print(f"Final estimated cardinality: {cpu_cardinalities[-1]:.2f}")
    
    # Count detections per step
    detections_per_step = []
    for est in estimates:
        if 'state' in est and len(est['state']) > 0:
            if hasattr(est['state'], 'shape'):
                detections_per_step.append(est['state'].shape[1])
            else:
                detections_per_step.append(0)
        else:
            detections_per_step.append(0)
    
    print(f"Mean detections per step: {np.mean(detections_per_step):.2f}")
    print(f"Max detections in single step: {np.max(detections_per_step)}")
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    print(f"Data generation time: {data_gen_time:.2f}s")
    print(f"Tracking time: {tracking_time:.2f}s")
    print(f"Time per step: {tracking_time/num_steps:.4f}s")
    print(f"Total runtime: {data_gen_time + tracking_time:.2f}s")
    
    # Memory usage information
    if device == 'cuda':
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Simple visualization (convert to CPU only for plotting)
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot estimated cardinality over time
        plt.subplot(2, 2, 1)
        plt.plot(cpu_cardinalities)
        plt.title('Estimated Number of Targets')
        plt.xlabel('Time Step')
        plt.ylabel('Cardinality')
        plt.grid(True)
        
        # Plot detections per step
        plt.subplot(2, 2, 2)
        plt.plot(detections_per_step)
        plt.title('Detections per Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Detections')
        plt.grid(True)
        
        # Plot true tracks (convert to CPU for plotting)
        plt.subplot(2, 2, 3)
        true_tracks_cpu = true_tracks.cpu().numpy() if hasattr(true_tracks, 'cpu') else true_tracks
        for target in range(min(5, true_tracks_cpu.shape[1])):
            track = true_tracks_cpu[:2, target, :]
            valid_mask = ~np.isnan(track[0, :])
            if np.any(valid_mask):
                plt.plot(track[0, valid_mask], track[1, valid_mask], 'o-', 
                        markersize=2, linewidth=1, label=f'Target {target+1}')
        
        plt.title('True Target Trajectories')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot sensor positions
        plt.subplot(2, 2, 4)
        sensor_pos = parameters['sensor_positions'].cpu().numpy() if hasattr(parameters['sensor_positions'], 'cpu') else parameters['sensor_positions']
        plt.scatter(sensor_pos[0, :], sensor_pos[1, :], s=100, c='red', marker='s', label='Sensors')
        
        # Plot surveillance region
        surv_region = parameters['surveillance_region'].cpu().numpy() if hasattr(parameters['surveillance_region'], 'cpu') else parameters['surveillance_region']
        plt.plot([surv_region[0,0], surv_region[1,0], surv_region[1,0], surv_region[0,0], surv_region[0,0]],
                [surv_region[0,1], surv_region[0,1], surv_region[1,1], surv_region[1,1], surv_region[0,1]], 
                'k--', label='Surveillance Region')
        
        plt.title(f'Sensor Configuration ({device.upper()})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'/Users/lipingb/Desktop/BP-MTT/Torch_Optimized/tracking_results_{device}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Results saved to variables for manual inspection")
    
    return {
        'true_tracks': true_tracks,
        'estimates': estimates,
        'estimated_cardinalities': cpu_cardinalities,
        'parameters': parameters,
        'performance': {
            'data_gen_time': data_gen_time,
            'tracking_time': tracking_time,
            'total_time': data_gen_time + tracking_time,
            'device': device
        }
    }


if __name__ == "__main__":
    results = main()
    print("\\nOptimized tracking simulation completed successfully!")
    print("Results are stored in the 'results' dictionary.")