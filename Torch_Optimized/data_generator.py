import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class Data_Generator:
    """
    Memory-optimized PyTorch data generator for multi-target tracking simulations.
    Minimizes CPU-GPU transfers and uses vectorized operations.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Pre-generate random seeds for reproducible results without CPU fallbacks
        self._setup_random_state()
        
    def _setup_random_state(self):
        """Setup random state generators to avoid repeated CPU-GPU transfers."""
        # Use torch generators for consistent device-specific randomness
        self.generator = torch.Generator(device=self.device)
        
    def get_start_states(self, num_targets: int, max_position: float, max_velocity: float) -> torch.Tensor:
        """Generate random initial states for targets."""
        states = torch.zeros(4, num_targets, device=self.device, dtype=torch.float32)
        states[:2] = (torch.rand(2, num_targets, device=self.device, generator=self.generator) - 0.5) * 2 * max_position
        states[2:] = (torch.rand(2, num_targets, device=self.device, generator=self.generator) - 0.5) * 2 * max_velocity
        return states
    
    def get_sensor_positions(self, num_sensors: int, max_range: float) -> torch.Tensor:
        """Generate sensor positions."""
        return (torch.rand(2, num_sensors, device=self.device, generator=self.generator) - 0.5) * 2 * max_range
    
    def get_transition_matrices(self, scan_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state transition matrices for constant velocity model."""
        A = torch.eye(4, device=self.device, dtype=torch.float32)
        A[0, 2] = scan_time
        A[1, 3] = scan_time
        
        W = torch.zeros(4, 2, device=self.device, dtype=torch.float32)
        W[0, 0] = 0.5 * scan_time**2
        W[1, 1] = 0.5 * scan_time**2
        W[2, 0] = scan_time
        W[3, 1] = scan_time
        
        return A, W
    
    def generate_true_tracks(self, parameters: Dict, num_steps: int) -> torch.Tensor:
        """Generate true target trajectories using vectorized operations."""
        target_start_states = parameters['target_start_states']
        target_appearance = parameters['target_appearance_from_to']
        length_steps = parameters['length_steps']
        driving_noise_variance = parameters['driving_noise_variance']
        
        num_targets = target_start_states.shape[1]
        target_tracks = torch.full((4, num_targets, num_steps), float('nan'), 
                                  device=self.device, dtype=torch.float32)
        
        # Vectorized approach: process all targets simultaneously
        current_states = target_start_states.clone()
        noise_std = math.sqrt(driving_noise_variance)
        
        for step in range(1, num_steps):
            A, W = self.get_transition_matrices(length_steps[step])
            
            # Generate noise for all targets at once
            noise = torch.randn(2, num_targets, device=self.device, generator=self.generator) * noise_std
            current_states = A @ current_states + W @ noise
            
            # Vectorized appearance check
            step_tensor = torch.tensor(step + 1, device=self.device)
            appear_mask = (target_appearance[0, :] <= step_tensor) & (step_tensor <= target_appearance[1, :])
            target_tracks[:, appear_mask, step] = current_states[:, appear_mask]
                    
        return target_tracks
    
    def generate_true_measurements(self, target_trajectory: torch.Tensor, parameters: Dict) -> torch.Tensor:
        """Generate true measurements from target positions using vectorized operations."""
        measurement_var_range = parameters['measurement_variance_range']
        measurement_var_bearing = parameters['measurement_variance_bearing']
        num_sensors = parameters['num_sensors']
        sensor_positions = parameters['sensor_positions']
        
        num_targets = target_trajectory.shape[1]
        measurements = torch.zeros(2, num_targets, num_sensors, device=self.device, dtype=torch.float32)
        
        # Vectorize across all sensors and targets
        for sensor in range(num_sensors):
            valid_mask = ~torch.isnan(target_trajectory[0, :])
            
            if torch.any(valid_mask):
                valid_positions = target_trajectory[:2, valid_mask]
                sensor_pos = sensor_positions[:, sensor:sensor+1]
                
                # Vectorized range and bearing computation
                dx = valid_positions[0:1, :] - sensor_pos[0:1, :]
                dy = valid_positions[1:2, :] - sensor_pos[1:2, :]
                
                true_range = torch.sqrt(dx**2 + dy**2)
                true_bearing = torch.atan2(dx, dy) * 180 / math.pi
                
                # Add noise
                range_noise = torch.randn(true_range.shape, device=self.device, generator=self.generator) * math.sqrt(measurement_var_range)
                bearing_noise = torch.randn(true_bearing.shape, device=self.device, generator=self.generator) * math.sqrt(measurement_var_bearing)
                
                measurements[0, valid_mask, sensor] = (true_range + range_noise).squeeze()
                measurements[1, valid_mask, sensor] = (true_bearing + bearing_noise).squeeze()
                
                # Set invalid measurements to nan
                measurements[:, ~valid_mask, sensor] = float('nan')
                    
        return measurements
    
    def generate_cluttered_measurements(self, track_measurements: torch.Tensor, parameters: Dict) -> List[torch.Tensor]:
        """Generate measurements with clutter using GPU-optimized operations."""
        detection_probability = parameters['detection_probability']
        num_sensors = parameters['num_sensors']
        mean_clutter = parameters['mean_clutter']
        measurement_range = parameters['measurement_range']
        
        num_targets = track_measurements.shape[1]
        all_measurements_list = []
        
        # Pre-generate all random numbers to minimize CPU-GPU transfers
        detection_probs = torch.rand(num_targets, num_sensors, device=self.device, generator=self.generator)
        
        # Generate Poisson samples (use approximation for GPU compatibility)
        clutter_counts = self._gpu_poisson_approx(mean_clutter, num_sensors)
        
        for sensor in range(num_sensors):
            # Determine detected targets
            exists_mask = ~torch.isnan(track_measurements[0, :, sensor])
            detection_mask = detection_probs[:, sensor] < detection_probability
            detected_mask = exists_mask & detection_mask
            
            detected_measurements = track_measurements[:, detected_mask, sensor]
            
            # Generate clutter
            num_clutter = clutter_counts[sensor].item()
            if num_clutter > 0:
                clutter = torch.zeros(2, num_clutter, device=self.device, dtype=torch.float32)
                clutter[0, :] = torch.rand(num_clutter, device=self.device, generator=self.generator) * measurement_range
                clutter[1, :] = torch.rand(num_clutter, device=self.device, generator=self.generator) * 360 - 180
            else:
                clutter = torch.empty(2, 0, device=self.device, dtype=torch.float32)
            
            # Combine measurements and permute
            all_measurements = torch.cat([clutter, detected_measurements], dim=1)
            if all_measurements.shape[1] > 0:
                perm = torch.randperm(all_measurements.shape[1], device=self.device, generator=self.generator)
                all_measurements = all_measurements[:, perm]
            
            all_measurements_list.append(all_measurements)
            
        return all_measurements_list
    
    def _gpu_poisson_approx(self, lam: float, num_samples: int) -> torch.Tensor:
        """GPU-compatible Poisson approximation using normal distribution for large lambda."""
        if lam > 30:
            # Use normal approximation for large lambda
            samples = torch.randn(num_samples, device=self.device, generator=self.generator)
            samples = samples * math.sqrt(lam) + lam
            samples = torch.clamp(samples, min=0)
            return samples.round().int()
        else:
            # Use CPU fallback for small lambda (unavoidable but infrequent)
            cpu_samples = torch.poisson(torch.full((num_samples,), lam, dtype=torch.float32)).int()
            return cpu_samples.to(self.device)
    
    def generate_simulation_data(self, parameters: Dict, num_steps: int, 
                               batch_size: int = 10) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Generate complete simulation dataset with batched processing."""
        # Generate true tracks
        true_tracks = self.generate_true_tracks(parameters, num_steps)
        
        # Process measurements in batches to reduce memory usage
        all_measurements = []
        
        for batch_start in range(0, num_steps, batch_size):
            batch_end = min(batch_start + batch_size, num_steps)
            batch_measurements = []
            
            for step in range(batch_start, batch_end):
                true_measurements = self.generate_true_measurements(
                    true_tracks[:, :, step], parameters
                )
                cluttered_measurements = self.generate_cluttered_measurements(
                    true_measurements, parameters
                )
                batch_measurements.append(cluttered_measurements)
            
            all_measurements.extend(batch_measurements)
            
        return true_tracks, all_measurements