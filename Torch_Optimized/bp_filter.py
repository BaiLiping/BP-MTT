import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class BP_Filter:
    """
    Memory-optimized PyTorch Belief Propagation filter for multi-target tracking.
    Minimizes CPU-GPU transfers and uses vectorized operations for better GPU utilization.
    """
    
    def __init__(self, parameters: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.parameters = parameters
        
        # Algorithm parameters (keep on GPU as tensors when possible)
        self.detection_threshold = torch.tensor(parameters['detection_threshold'], device=device)
        self.threshold_pruning = torch.tensor(parameters['threshold_pruning'], device=device)
        self.num_particles = parameters['num_particles']
        
        # Initialize state
        self.posterior_beliefs = torch.empty(4, self.num_particles, 0, device=device, dtype=torch.float32)
        self.posterior_existences = torch.empty(0, device=device, dtype=torch.float32)
        self.posterior_labels = torch.empty(3, 0, device=device, dtype=torch.float32)
        
        # Cache frequently used constants
        self._cache_constants()
        
        # Random generator for consistent randomness
        self.generator = torch.Generator(device=device)
        
    def _cache_constants(self):
        """Cache frequently used constants to avoid repeated tensor creation."""
        self.sqrt_2pi = torch.tensor(math.sqrt(2 * math.pi), device=self.device)
        self.detection_prob = torch.tensor(self.parameters['detection_probability'], device=self.device)
        self.survival_prob = torch.tensor(self.parameters['survival_probability'], device=self.device)
        self.driving_noise_var = torch.tensor(self.parameters['driving_noise_variance'], device=self.device)
        self.meas_var_range = torch.tensor(self.parameters['measurement_variance_range'], device=self.device)
        self.meas_var_bearing = torch.tensor(self.parameters['measurement_variance_bearing'], device=self.device)
        
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
    
    def perform_prediction(self, old_particles: torch.Tensor, old_existences: torch.Tensor, 
                          scan_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform time prediction step with vectorized operations."""
        if old_particles.shape[2] == 0:
            return old_particles, old_existences
            
        A, W = self.get_transition_matrices(scan_time)
        num_targets = old_particles.shape[2]
        
        # Vectorized prediction for all targets at once
        new_particles = old_particles.clone()
        
        if num_targets > 0:
            # Generate noise for all targets simultaneously
            noise_std = torch.sqrt(self.driving_noise_var)
            noise = torch.randn(2, self.num_particles, num_targets, 
                              device=self.device, generator=self.generator) * noise_std
            
            # Apply prediction to all targets
            for target in range(num_targets):
                new_particles[:, :, target] = A @ old_particles[:, :, target] + W @ noise[:, :, target]
            
        new_existences = self.survival_prob * old_existences
        
        return new_particles, new_existences
    
    def evaluate_measurements(self, alphas: torch.Tensor, alphas_existence: torch.Tensor,
                            measurements: torch.Tensor, sensor_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate measurement likelihood factors with vectorized operations."""
        if measurements.shape[1] == 0 or alphas.shape[2] == 0:
            return (torch.empty(1, 0, device=self.device, dtype=torch.float32),
                   torch.empty(1, 0, self.num_particles, device=self.device, dtype=torch.float32))
            
        clutter_intensity = self.parameters['mean_clutter'] * self.parameters['clutter_distribution']
        sensor_position = self.parameters['sensor_positions'][:, sensor_index]
        
        num_measurements = measurements.shape[1]
        num_targets = alphas.shape[2]
        
        beta_messages = torch.zeros(num_measurements + 1, num_targets, device=self.device, dtype=torch.float32)
        v_factors1 = torch.zeros(num_measurements + 1, num_targets, self.num_particles, 
                                device=self.device, dtype=torch.float32)
        
        # Non-detection factor
        v_factors1[0, :, :] = 1 - self.detection_prob
        
        # Constant factor
        constant_factor = (1 / (2 * math.pi * torch.sqrt(self.meas_var_bearing * self.meas_var_range)) *
                          self.detection_prob / clutter_intensity)
        
        # Vectorized computation for all targets
        if num_targets > 0:
            # Compute predicted measurements for all targets and particles at once
            dx = alphas[0, :, :] - sensor_position[0]  # [particles, targets]
            dy = alphas[1, :, :] - sensor_position[1]  # [particles, targets]
            predicted_range = torch.sqrt(dx**2 + dy**2)  # [particles, targets]
            predicted_bearing = torch.atan2(dx, dy) * 180 / math.pi  # [particles, targets]
            
            for measurement_idx in range(num_measurements):
                meas_range = measurements[0, measurement_idx]
                meas_bearing = measurements[1, measurement_idx]
                
                # Vectorized likelihood computation
                range_diff = meas_range - predicted_range  # [particles, targets]
                bearing_diff = self.wrap_to_180(meas_bearing - predicted_bearing)  # [particles, targets]
                
                range_likelihood = torch.exp(-0.5 * range_diff**2 / self.meas_var_range)
                bearing_likelihood = torch.exp(-0.5 * bearing_diff**2 / self.meas_var_bearing)
                
                v_factors1[measurement_idx + 1, :, :] = (constant_factor * 
                                                        range_likelihood.T * bearing_likelihood.T)
        
        # Compute beta messages
        v_factors0 = torch.zeros(num_measurements + 1, num_targets, device=self.device)
        v_factors0[0, :] = 1
        
        existence_expanded = alphas_existence.unsqueeze(0).expand(num_measurements + 1, -1)
        beta_messages = (existence_expanded * torch.mean(v_factors1, dim=2) + 
                        (1 - existence_expanded) * v_factors0)
        
        return beta_messages, v_factors1
    
    def wrap_to_180(self, angles: torch.Tensor) -> torch.Tensor:
        """Wrap angles to [-180, 180] degrees using GPU operations."""
        return ((angles + 180) % 360) - 180
    
    def introduce_new_pts(self, measurements: torch.Tensor, sensor: int, step: int,
                         unknown_number: float, unknown_particles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Introduce new particle tracks with vectorized operations."""
        if measurements.shape[1] == 0:
            return (torch.empty(4, self.num_particles, 0, device=self.device),
                   torch.empty(3, 0, device=self.device),
                   torch.empty(0, device=self.device), 
                   torch.empty(0, device=self.device))
            
        clutter_intensity = self.parameters['mean_clutter'] * self.parameters['clutter_distribution']
        sensor_position = self.parameters['sensor_positions'][:, sensor]
        surveillance_region = self.parameters['surveillance_region']
        
        num_measurements = measurements.shape[1]
        
        # Compute unknown intensity
        area = ((surveillance_region[1, 0] - surveillance_region[0, 0]) *
                (surveillance_region[1, 1] - surveillance_region[0, 1]))
        unknown_intensity = unknown_number / area
        unknown_intensity *= (1 - self.detection_prob) ** (sensor - 1)
        
        # Calculate constants
        constants = self.calculate_constants_uniform(sensor_position, measurements, unknown_particles)
        
        # Initialize new particle tracks
        new_pts = torch.zeros(4, self.num_particles, num_measurements, device=self.device, dtype=torch.float32)
        new_labels = torch.zeros(3, num_measurements, device=self.device, dtype=torch.float32)
        xi_messages = torch.zeros(num_measurements, device=self.device, dtype=torch.float32)
        
        # Vectorized particle generation
        if num_measurements > 0:
            # Generate all particles at once
            all_particles = self.sample_from_likelihood_vectorized(measurements, sensor, self.num_particles)
            new_pts = all_particles
            
            # Labels
            step_tensor = torch.tensor(step, device=self.device, dtype=torch.float32)
            sensor_tensor = torch.tensor(sensor, device=self.device, dtype=torch.float32)
            measurement_indices = torch.arange(num_measurements, device=self.device, dtype=torch.float32)
            
            new_labels[0, :] = step_tensor
            new_labels[1, :] = sensor_tensor
            new_labels[2, :] = measurement_indices
            
            # Xi messages
            xi_messages = 1 + (constants * unknown_intensity * self.detection_prob) / clutter_intensity
        
        new_existences = xi_messages - 1
        
        return new_pts, new_labels, new_existences, xi_messages
    
    def calculate_constants_uniform(self, sensor_position: torch.Tensor, measurements: torch.Tensor,
                                  particles: torch.Tensor) -> torch.Tensor:
        """Calculate constants with vectorized operations."""
        surveillance_region = self.parameters['surveillance_region']
        
        num_measurements = measurements.shape[1]
        
        area = ((surveillance_region[1, 0] - surveillance_region[0, 0]) *
                (surveillance_region[1, 1] - surveillance_region[0, 1]))
        constant_weight = 1 / area
        
        # Vectorized prediction for all particles
        dx = particles[0, :] - sensor_position[0]
        dy = particles[1, :] - sensor_position[1]
        predicted_range = torch.sqrt(dx**2 + dy**2)
        predicted_bearing = torch.atan2(dx, dy) * 180 / math.pi
        
        constant_likelihood = 1 / (2 * math.pi * torch.sqrt(self.meas_var_bearing * self.meas_var_range))
        
        # Vectorized computation for all measurements
        constants = torch.zeros(num_measurements, device=self.device, dtype=torch.float32)
        
        if num_measurements > 0:
            # Expand measurements and predictions for vectorized computation
            meas_range = measurements[0, :].unsqueeze(1)  # [measurements, 1]
            meas_bearing = measurements[1, :].unsqueeze(1)  # [measurements, 1]
            pred_range = predicted_range.unsqueeze(0)  # [1, particles]
            pred_bearing = predicted_bearing.unsqueeze(0)  # [1, particles]
            
            # Compute likelihoods for all measurement-particle pairs
            range_likelihood = torch.exp(-0.5 * (meas_range - pred_range)**2 / self.meas_var_range)
            bearing_likelihood = torch.exp(-0.5 * (meas_bearing - pred_bearing)**2 / self.meas_var_bearing)
            
            # Sum over particles for each measurement
            constants = torch.mean(constant_likelihood * range_likelihood * bearing_likelihood, dim=1)
        
        constants = constants / constant_weight
        return constants
    
    def sample_from_likelihood_vectorized(self, measurements: torch.Tensor, sensor_index: int, 
                                        num_particles: int) -> torch.Tensor:
        """Sample particles from measurement likelihood with vectorized operations."""
        sensor_position = self.parameters['sensor_positions'][:, sensor_index]
        prior_velocity_covariance = self.parameters['prior_velocity_covariance']
        
        num_measurements = measurements.shape[1]
        samples = torch.zeros(4, num_particles, num_measurements, device=self.device, dtype=torch.float32)
        
        if num_measurements > 0:
            # Sample range and bearing noise for all measurements at once
            range_noise = torch.randn(num_particles, num_measurements, device=self.device, generator=self.generator) * torch.sqrt(self.meas_var_range)
            bearing_noise = torch.randn(num_particles, num_measurements, device=self.device, generator=self.generator) * torch.sqrt(self.meas_var_bearing)
            
            # Add noise to measurements
            random_range = measurements[0, :].unsqueeze(0) + range_noise  # [particles, measurements]
            random_bearing = measurements[1, :].unsqueeze(0) + bearing_noise  # [particles, measurements]
            
            # Convert to Cartesian coordinates
            bearing_rad = random_bearing * math.pi / 180
            samples[0, :, :] = sensor_position[0] + random_range * torch.sin(bearing_rad)
            samples[1, :, :] = sensor_position[1] + random_range * torch.cos(bearing_rad)
            
            # Sample velocities
            velocity_samples = torch.randn(2, num_particles, num_measurements, device=self.device, generator=self.generator)
            chol = torch.linalg.cholesky(prior_velocity_covariance)
            for m in range(num_measurements):
                samples[2:4, :, m] = chol @ velocity_samples[:, :, m]
        
        return samples
    
    def perform_data_association_bp(self, input_legacy: torch.Tensor, input_new: torch.Tensor,
                                   check_convergence: int = 20, threshold: float = 1e-5,
                                   num_iterations: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform belief propagation with GPU-optimized operations."""
        if input_legacy.shape[1] == 0 or input_legacy.shape[0] <= 1:
            num_measurements = input_new.shape[0] if input_new.numel() > 0 else 0
            num_objects = input_legacy.shape[1] if input_legacy.numel() > 0 else 0
            return (torch.ones(num_measurements, num_objects, device=self.device, dtype=torch.float32),
                   torch.ones(num_measurements, device=self.device, dtype=torch.float32))
        
        num_measurements = input_legacy.shape[0] - 1
        num_objects = input_legacy.shape[1]
        
        output_legacy = torch.ones(num_measurements, num_objects, device=self.device, dtype=torch.float32)
        output_new = torch.ones(num_measurements, device=self.device, dtype=torch.float32)
        
        if num_objects == 0 or num_measurements == 0:
            return output_legacy, output_new
        
        messages2 = torch.ones(num_measurements, num_objects, device=self.device, dtype=torch.float32)
        
        # Optimize iterations with early stopping
        for iteration in range(num_iterations):
            messages2_old = messages2.clone()
            
            # Vectorized message passing
            product1 = messages2 * input_legacy[1:, :]
            sum1 = input_legacy[0, :].unsqueeze(0) + torch.sum(product1, dim=0, keepdim=True)
            
            # Add small epsilon to avoid division by zero
            eps = torch.finfo(messages2.dtype).eps
            messages1 = input_legacy[1:, :] / torch.clamp(sum1.expand_as(product1) - product1, min=eps)
            
            sum2 = input_new.unsqueeze(1) + torch.sum(messages1, dim=1, keepdim=True)
            messages2 = 1 / torch.clamp(sum2.expand_as(messages1) - messages1, min=eps)
            
            # Check convergence less frequently to reduce overhead
            if iteration % check_convergence == 0 and iteration > 0:
                # Use max absolute difference instead of log for stability
                distance = torch.max(torch.abs(messages2 - messages2_old))
                if distance < threshold:
                    break
        
        output_legacy = messages2
        
        output_new_full = torch.cat([torch.ones(num_measurements, 1, device=self.device), messages1], dim=1)
        row_sums = torch.sum(output_new_full, dim=1, keepdim=True)
        output_new_full = output_new_full / torch.clamp(row_sums, min=eps)
        output_new = output_new_full[:, 0]
        
        return output_legacy, output_new
    
    def resample_systematic_vectorized(self, weights: torch.Tensor, num_particles: int) -> torch.Tensor:
        """Vectorized systematic resampling."""
        # Normalize weights
        weight_sum = torch.sum(weights)
        if weight_sum <= 0:
            return torch.arange(num_particles, device=self.device) % len(weights)
            
        normalized_weights = weights / weight_sum
        cum_weights = torch.cumsum(normalized_weights, dim=0)
        
        # Generate systematic grid
        u = torch.rand(1, device=self.device, generator=self.generator) / num_particles
        grid = torch.linspace(0, (num_particles - 1) / num_particles, num_particles, device=self.device) + u
        
        # Vectorized search
        indices = torch.searchsorted(cum_weights, grid, right=False)
        indices = torch.clamp(indices, 0, len(cum_weights) - 1)
        
        return indices
    
    def update_pts(self, kappas: torch.Tensor, iotas: torch.Tensor, legacy_pts: torch.Tensor,
                  new_pts: torch.Tensor, legacy_existences: torch.Tensor, new_existences: torch.Tensor,
                  legacy_labels: torch.Tensor, new_labels: torch.Tensor, v_factors1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update particle tracks with vectorized operations."""
        num_targets = v_factors1.shape[1]
        num_measurements = v_factors1.shape[0] - 1
        
        if num_targets > 0:
            # Vectorized weight computation
            weights = v_factors1[0, :, :].clone()  # [targets, particles]
            
            for measurement in range(num_measurements):
                weights += kappas[measurement, :].unsqueeze(1) * v_factors1[measurement + 1, :, :]
            
            # Vectorized existence update
            sum_weights = torch.sum(weights, dim=1)  # [targets]
            
            is_alive = legacy_existences * sum_weights / self.num_particles
            is_dead = 1 - legacy_existences
            legacy_existences = is_alive / (is_alive + is_dead)
            
            # Vectorized resampling for all targets
            for target in range(num_targets):
                if sum_weights[target] > 0:
                    indices = self.resample_systematic_vectorized(weights[target, :], self.num_particles)
                    legacy_pts[:, :, target] = legacy_pts[:, indices, target]
        
        # Merge new and legacy particle tracks
        if new_pts.shape[2] > 0:
            legacy_pts = torch.cat([legacy_pts, new_pts], dim=2)
            
            # Avoid division by zero
            eps = torch.finfo(new_existences.dtype).eps
            denominator = torch.clamp(iotas * new_existences + 1, min=eps)
            new_existences = iotas * new_existences / denominator
            
            legacy_existences = torch.cat([legacy_existences, new_existences])
            legacy_labels = torch.cat([legacy_labels, new_labels], dim=1)
        
        return legacy_pts, legacy_existences, legacy_labels
    
    def track_step(self, cluttered_measurements: List[torch.Tensor], unknown_number: float,
                  unknown_particles: torch.Tensor, step: int) -> Dict:
        """Perform one tracking step with optimized GPU operations."""
        num_sensors = self.parameters['num_sensors']
        length_step = self.parameters['length_steps'][step]
        
        # Prediction
        self.posterior_beliefs, self.posterior_existences = self.perform_prediction(
            self.posterior_beliefs, self.posterior_existences, length_step
        )
        
        # Process each sensor
        for sensor in range(num_sensors):
            measurements = cluttered_measurements[sensor]
            
            if measurements.shape[1] == 0:
                continue
                
            # Introduce new particle tracks
            new_pts, new_labels, new_existences, xi_messages = self.introduce_new_pts(
                measurements, sensor, step, unknown_number, unknown_particles
            )
            
            # Evaluate measurements
            beta_messages, v_factors1 = self.evaluate_measurements(
                self.posterior_beliefs, self.posterior_existences, measurements, sensor
            )
            
            # Perform data association
            kappas, iotas = self.perform_data_association_bp(beta_messages, xi_messages)
            
            # Update particle tracks
            self.posterior_beliefs, self.posterior_existences, self.posterior_labels = self.update_pts(
                kappas, iotas, self.posterior_beliefs, new_pts,
                self.posterior_existences, new_existences,
                self.posterior_labels, new_labels, v_factors1
            )
            
            # Pruning with vectorized operations
            if self.posterior_existences.shape[0] > 0:
                keep_mask = self.posterior_existences >= self.threshold_pruning
                if torch.any(keep_mask):
                    self.posterior_beliefs = self.posterior_beliefs[:, :, keep_mask]
                    self.posterior_existences = self.posterior_existences[keep_mask]
                    self.posterior_labels = self.posterior_labels[:, keep_mask]
                else:
                    # Reset to empty if all pruned
                    self.posterior_beliefs = torch.empty(4, self.num_particles, 0, device=self.device)
                    self.posterior_existences = torch.empty(0, device=self.device)
                    self.posterior_labels = torch.empty(3, 0, device=self.device)
        
        # Generate estimates (keep on GPU until needed)
        estimated_cardinality = torch.sum(self.posterior_existences)
        estimates = self.generate_estimates()
        
        return {
            'estimates': estimates,
            'estimated_cardinality': estimated_cardinality,
        }
    
    def generate_estimates(self) -> Dict:
        """Generate state estimates with GPU operations."""
        estimates = {'state': [], 'label': []}
        
        if self.posterior_existences.shape[0] == 0:
            return estimates
            
        detected_mask = self.posterior_existences > self.detection_threshold
        
        if torch.any(detected_mask):
            detected_beliefs = self.posterior_beliefs[:, :, detected_mask]
            detected_labels = self.posterior_labels[:, detected_mask]
            
            # Compute mean states on GPU
            mean_states = torch.mean(detected_beliefs, dim=1)
            
            estimates['state'] = mean_states
            estimates['label'] = detected_labels
            
        return estimates