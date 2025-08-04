import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class Data_Generator:
    """
    NumPy implementation of data generator for multi-target tracking simulations.
    Converted from PyTorch implementation for performance comparison.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
        
    def get_start_states(self, num_targets: int, max_position: float, max_velocity: float) -> np.ndarray:
        """Generate random initial states for targets."""
        states = np.zeros((4, num_targets))
        states[:2] = (np.random.rand(2, num_targets) - 0.5) * 2 * max_position
        states[2:] = (np.random.rand(2, num_targets) - 0.5) * 2 * max_velocity
        return states
    
    def get_sensor_positions(self, num_sensors: int, max_range: float) -> np.ndarray:
        """Generate sensor positions."""
        return (np.random.rand(2, num_sensors) - 0.5) * 2 * max_range
    
    def get_transition_matrices(self, scan_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get state transition matrices for constant velocity model."""
        A = np.eye(4)
        A[0, 2] = scan_time
        A[1, 3] = scan_time
        
        W = np.zeros((4, 2))
        W[0, 0] = 0.5 * scan_time**2
        W[1, 1] = 0.5 * scan_time**2
        W[2, 0] = scan_time
        W[3, 1] = scan_time
        
        return A, W
    
    def generate_true_tracks(self, parameters: Dict, num_steps: int) -> np.ndarray:
        """Generate true target trajectories."""
        target_start_states = parameters['target_start_states']
        target_appearance = parameters['target_appearance_from_to']
        length_steps = parameters['length_steps']
        driving_noise_variance = parameters['driving_noise_variance']
        
        num_targets = target_start_states.shape[1]
        target_tracks = np.full((4, num_targets, num_steps), np.nan)
        
        for target in range(num_targets):
            current_state = target_start_states[:, target].copy()
            
            for step in range(1, num_steps):
                A, W = self.get_transition_matrices(length_steps[step])
                noise = np.random.randn(2) * math.sqrt(driving_noise_variance)
                current_state = A @ current_state + W @ noise
                
                if target_appearance[0, target] <= step + 1 <= target_appearance[1, target]:
                    target_tracks[:, target, step] = current_state
                    
        return target_tracks
    
    def generate_true_measurements(self, target_trajectory: np.ndarray, parameters: Dict) -> np.ndarray:
        """Generate true measurements from target positions."""
        measurement_var_range = parameters['measurement_variance_range']
        measurement_var_bearing = parameters['measurement_variance_bearing']
        num_sensors = parameters['num_sensors']
        sensor_positions = parameters['sensor_positions']
        
        num_targets = target_trajectory.shape[1]
        measurements = np.zeros((2, num_targets, num_sensors))
        
        for sensor in range(num_sensors):
            for target in range(num_targets):
                if not np.isnan(target_trajectory[0, target]):
                    # Range measurement
                    dx = target_trajectory[0, target] - sensor_positions[0, sensor]
                    dy = target_trajectory[1, target] - sensor_positions[1, sensor]
                    true_range = np.sqrt(dx**2 + dy**2)
                    range_noise = np.random.randn() * math.sqrt(measurement_var_range)
                    measurements[0, target, sensor] = true_range + range_noise
                    
                    # Bearing measurement (in degrees)
                    true_bearing = np.arctan2(dx, dy) * 180 / math.pi
                    bearing_noise = np.random.randn() * math.sqrt(measurement_var_bearing)
                    measurements[1, target, sensor] = true_bearing + bearing_noise
                else:
                    measurements[:, target, sensor] = np.nan
                    
        return measurements
    
    def generate_cluttered_measurements(self, track_measurements: np.ndarray, parameters: Dict) -> List[np.ndarray]:
        """Generate measurements with clutter."""
        detection_probability = parameters['detection_probability']
        num_sensors = parameters['num_sensors']
        mean_clutter = parameters['mean_clutter']
        measurement_range = parameters['measurement_range']
        
        num_targets = track_measurements.shape[1]
        all_measurements_list = []
        
        for sensor in range(num_sensors):
            # Determine detected targets
            exists_mask = ~np.isnan(track_measurements[0, :, sensor])
            detection_mask = np.random.rand(num_targets) < detection_probability
            detected_mask = exists_mask & detection_mask
            
            detected_measurements = track_measurements[:, detected_mask, sensor]
            
            # Generate clutter
            num_clutter = np.random.poisson(mean_clutter)
            if num_clutter > 0:
                clutter = np.zeros((2, num_clutter))
                clutter[0, :] = np.random.rand(num_clutter) * measurement_range
                clutter[1, :] = np.random.rand(num_clutter) * 360 - 180
            else:
                clutter = np.zeros((2, 0))
            
            # Combine measurements and permute
            all_measurements = np.concatenate([clutter, detected_measurements], axis=1)
            if all_measurements.shape[1] > 0:
                perm = np.random.permutation(all_measurements.shape[1])
                all_measurements = all_measurements[:, perm]
            
            all_measurements_list.append(all_measurements)
            
        return all_measurements_list
    
    def generate_simulation_data(self, parameters: Dict, num_steps: int) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """Generate complete simulation dataset."""
        # Generate true tracks
        true_tracks = self.generate_true_tracks(parameters, num_steps)
        
        # Generate measurements for each time step
        all_measurements = []
        for step in range(num_steps):
            true_measurements = self.generate_true_measurements(
                true_tracks[:, :, step], parameters
            )
            cluttered_measurements = self.generate_cluttered_measurements(
                true_measurements, parameters
            )
            all_measurements.append(cluttered_measurements)
            
        return true_tracks, all_measurements