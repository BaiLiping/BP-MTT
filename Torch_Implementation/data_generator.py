import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class Data_Generator:
    """
    PyTorch optimized data generator for multi-target tracking simulations.
    Converts MATLAB implementation to efficient tensor operations.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def get_start_states(self, num_targets: int, max_position: float, max_velocity: float) -> torch.Tensor:
        """Generate random initial states for targets."""
        states = torch.zeros(4, num_targets, device=self.device)
        states[:2] = (torch.rand(2, num_targets, device=self.device) - 0.5) * 2 * max_position
        states[2:] = (torch.rand(2, num_targets, device=self.device) - 0.5) * 2 * max_velocity
        return states
    
    def get_sensor_positions(self, num_sensors: int, max_range: float) -> torch.Tensor:
        """Generate sensor positions."""
        return (torch.rand(2, num_sensors, device=self.device) - 0.5) * 2 * max_range
    
    def get_transition_matrices(self, scan_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state transition matrices for constant velocity model."""
        A = torch.eye(4, device=self.device)
        A[0, 2] = scan_time
        A[1, 3] = scan_time
        
        W = torch.zeros(4, 2, device=self.device)
        W[0, 0] = 0.5 * scan_time**2
        W[1, 1] = 0.5 * scan_time**2
        W[2, 0] = scan_time
        W[3, 1] = scan_time
        
        return A, W
    
    def generate_true_tracks(self, parameters: Dict, num_steps: int) -> torch.Tensor:
        """Generate true target trajectories."""
        target_start_states = parameters['target_start_states']
        target_appearance = parameters['target_appearance_from_to']
        length_steps = parameters['length_steps']
        driving_noise_variance = parameters['driving_noise_variance']
        
        num_targets = target_start_states.shape[1]
        target_tracks = torch.full((4, num_targets, num_steps), float('nan'), device=self.device)
        
        for target in range(num_targets):
            current_state = target_start_states[:, target].clone()
            
            for step in range(1, num_steps):
                A, W = self.get_transition_matrices(length_steps[step])
                noise = torch.randn(2, device=self.device) * math.sqrt(driving_noise_variance)
                current_state = A @ current_state + W @ noise
                
                if target_appearance[0, target] <= step + 1 <= target_appearance[1, target]:
                    target_tracks[:, target, step] = current_state
                    
        return target_tracks
    
    def generate_true_measurements(self, target_trajectory: torch.Tensor, parameters: Dict) -> torch.Tensor:
        """Generate true measurements from target positions."""
        measurement_var_range = parameters['measurement_variance_range']
        measurement_var_bearing = parameters['measurement_variance_bearing']
        num_sensors = parameters['num_sensors']
        sensor_positions = parameters['sensor_positions']
        
        num_targets = target_trajectory.shape[1]
        measurements = torch.zeros(2, num_targets, num_sensors, device=self.device)
        
        for sensor in range(num_sensors):
            for target in range(num_targets):
                if not torch.isnan(target_trajectory[0, target]):
                    # Range measurement
                    dx = target_trajectory[0, target] - sensor_positions[0, sensor]
                    dy = target_trajectory[1, target] - sensor_positions[1, sensor]
                    true_range = torch.sqrt(dx**2 + dy**2)
                    range_noise = torch.randn(1, device=self.device) * math.sqrt(measurement_var_range)
                    measurements[0, target, sensor] = true_range + range_noise
                    
                    # Bearing measurement (in degrees)
                    true_bearing = torch.atan2(dx, dy) * 180 / math.pi
                    bearing_noise = torch.randn(1, device=self.device) * math.sqrt(measurement_var_bearing)
                    measurements[1, target, sensor] = true_bearing + bearing_noise
                else:
                    measurements[:, target, sensor] = float('nan')
                    
        return measurements
    
    def generate_cluttered_measurements(self, track_measurements: torch.Tensor, parameters: Dict) -> List[torch.Tensor]:
        """Generate measurements with clutter."""
        detection_probability = parameters['detection_probability']
        num_sensors = parameters['num_sensors']
        mean_clutter = parameters['mean_clutter']
        measurement_range = parameters['measurement_range']
        
        num_targets = track_measurements.shape[1]
        all_measurements_list = []
        
        for sensor in range(num_sensors):
            # Determine detected targets
            exists_mask = ~torch.isnan(track_measurements[0, :, sensor])
            detection_mask = torch.rand(num_targets, device=self.device) < detection_probability
            detected_mask = exists_mask & detection_mask
            
            detected_measurements = track_measurements[:, detected_mask, sensor]
            
            # Generate clutter (use CPU for poisson if MPS doesn't support it)
            if self.device == 'mps':
                # MPS doesn't support poisson yet, use CPU fallback
                num_clutter = torch.poisson(torch.tensor(mean_clutter, dtype=torch.float32)).int().item()
            else:
                num_clutter = torch.poisson(torch.tensor(mean_clutter, dtype=torch.float32, device=self.device)).int().item()
            if num_clutter > 0:
                clutter = torch.zeros(2, num_clutter, device=self.device)
                clutter[0, :] = torch.rand(num_clutter, device=self.device) * measurement_range
                clutter[1, :] = torch.rand(num_clutter, device=self.device) * 360 - 180
            else:
                clutter = torch.zeros(2, 0, device=self.device)
            
            # Combine measurements and permute
            all_measurements = torch.cat([clutter, detected_measurements], dim=1)
            if all_measurements.shape[1] > 0:
                perm = torch.randperm(all_measurements.shape[1], device=self.device)
                all_measurements = all_measurements[:, perm]
            
            all_measurements_list.append(all_measurements)
            
        return all_measurements_list
    
    def generate_simulation_data(self, parameters: Dict, num_steps: int) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
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