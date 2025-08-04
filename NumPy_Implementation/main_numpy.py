#!/usr/bin/env python3
"""
NumPy implementation of BP-based multitarget tracking.
Converted from PyTorch implementation for performance comparison.

Implementation of BP-based multitarget tracking using particles as presented in [A] and [B].

[A] F. Meyer, P. Braca, P. Willett, and F. Hlawatsch, "A scalable algorithm for tracking an unknown number of targets using multiple sensors," IEEE Trans. Signal Process., vol. 65, pp. 3478–3493, Jul. 2017.
[B] F. Meyer, T. Kropfreiter, J. L. Williams, R. A. Lau, F. Hlawatsch, P. Braca, and M. Z. Win, "Message passing algorithms for scalable multitarget tracking," Proc. IEEE, vol. 106, pp. 221–259, Feb. 2018.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import Data_Generator
from bp_filter import BP_Filter
import time


def main():
    """Main function to test the Data_Generator and BP_Filter classes."""
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    print("Using NumPy implementation")
    
    # Model parameters
    num_steps = 200
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
        'measurement_range': 2 * 3000,  # 2 * surveillance_region max
        'mean_clutter': 5,
        'clutter_distribution': 1 / (360 * 2 * 3000),
        
        # Algorithm parameters
        'detection_threshold': 0.5,
        'threshold_pruning': 1e-4,
        'minimum_track_length': 1,
        'num_particles': 10000
    }
    
    # Initialize data generator with fixed seed
    data_gen = Data_Generator(random_seed=1)
    
    # Generate target start states and appearance times
    parameters['target_start_states'] = data_gen.get_start_states(5, 1000, 10)
    parameters['target_appearance_from_to'] = np.array([
        [5, 155], [10, 160], [15, 165], [20, 170], [25, 175]
    ], dtype=np.float32).T
    
    # Generate sensor positions
    parameters['sensor_positions'] = data_gen.get_sensor_positions(parameters['num_sensors'], 5000)
    
    print("Generating simulation data...")
    start_time = time.time()
    
    # Generate true tracks and measurements
    true_tracks, all_measurements = data_gen.generate_simulation_data(parameters, num_steps)
    
    data_gen_time = time.time() - start_time
    print(f"Data generation completed in {data_gen_time:.2f} seconds")
    
    # Initialize BP filter
    bp_filter = BP_Filter(parameters)
    
    # Initialize unknown target parameters
    unknown_number = 0.01
    unknown_particles = np.zeros((2, parameters['num_particles']))
    surveillance_region = parameters['surveillance_region']
    unknown_particles[0, :] = (np.random.rand(parameters['num_particles']) * 
                              (surveillance_region[1, 0] - surveillance_region[0, 0]) + 
                              surveillance_region[0, 0])
    unknown_particles[1, :] = (np.random.rand(parameters['num_particles']) * 
                              (surveillance_region[1, 1] - surveillance_region[0, 1]) + 
                              surveillance_region[0, 1])
    
    # Tracking loop
    print("Starting BP tracking...")
    start_time = time.time()
    
    estimates = []
    estimated_cardinalities = []
    
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
        estimated_cardinalities.append(result['estimated_cardinality'])
    
    tracking_time = time.time() - start_time
    print(f"Tracking completed in {tracking_time:.2f} seconds")
    
    # Analysis and visualization
    print("\n=== Tracking Results ===")
    print(f"Total simulation steps: {num_steps}")
    print(f"Number of sensors: {parameters['num_sensors']}")
    print(f"Number of particles: {parameters['num_particles']}")
    print(f"Mean estimated cardinality: {np.mean(estimated_cardinalities):.2f}")
    print(f"Final estimated cardinality: {estimated_cardinalities[-1]:.2f}")
    
    # Count detections per step
    detections_per_step = []
    for est in estimates:
        if 'state' in est and len(est['state']) > 0:
            if isinstance(est['state'], np.ndarray):
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
    
    # Simple visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot estimated cardinality over time
        plt.subplot(2, 2, 1)
        plt.plot(estimated_cardinalities)
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
        
        # Plot true tracks (first few targets)
        plt.subplot(2, 2, 3)
        for target in range(min(5, true_tracks.shape[1])):
            track = true_tracks[:2, target, :]
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
        sensor_pos = parameters['sensor_positions']
        plt.scatter(sensor_pos[0, :], sensor_pos[1, :], s=100, c='red', marker='s', label='Sensors')
        
        # Plot surveillance region
        surv_region = parameters['surveillance_region']
        plt.plot([surv_region[0,0], surv_region[1,0], surv_region[1,0], surv_region[0,0], surv_region[0,0]],
                [surv_region[0,1], surv_region[0,1], surv_region[1,1], surv_region[1,1], surv_region[0,1]], 
                'k--', label='Surveillance Region')
        
        plt.title('Sensor Configuration')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('/Users/lipingb/Desktop/BP-MTT/NumPy_Implementation/tracking_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Results saved to variables for manual inspection")
    
    return {
        'true_tracks': true_tracks,
        'estimates': estimates,
        'estimated_cardinalities': estimated_cardinalities,
        'parameters': parameters,
        'performance': {
            'data_gen_time': data_gen_time,
            'tracking_time': tracking_time,
            'total_time': data_gen_time + tracking_time
        }
    }


if __name__ == "__main__":
    results = main()
    print("\\nTracking simulation completed successfully!")
    print("Results are stored in the 'results' dictionary.")