import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class BP_Filter:
    """
    PyTorch optimized Belief Propagation filter for multi-target tracking.
    Converts MATLAB implementation to efficient tensor operations.
    """
    
    def __init__(self, parameters: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.parameters = parameters
        
        # Algorithm parameters
        self.detection_threshold = parameters['detection_threshold']
        self.threshold_pruning = parameters['threshold_pruning']
        self.num_particles = parameters['num_particles']
        
        # Initialize state
        self.posterior_beliefs = torch.zeros(4, self.num_particles, 0, device=device)
        self.posterior_existences = torch.zeros(0, device=device)
        self.posterior_labels = torch.zeros(3, 0, device=device)
        
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
    
    def perform_prediction(self, old_particles: torch.Tensor, old_existences: torch.Tensor, 
                          scan_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform time prediction step."""
        if old_particles.shape[2] == 0:
            return old_particles, old_existences
            
        driving_noise_variance = self.parameters['driving_noise_variance']
        survival_probability = self.parameters['survival_probability']
        
        A, W = self.get_transition_matrices(scan_time)
        
        new_particles = old_particles.clone()
        num_targets = old_particles.shape[2]
        
        for target in range(num_targets):
            noise = torch.randn(2, self.num_particles, device=self.device) * math.sqrt(driving_noise_variance)
            new_particles[:, :, target] = A @ old_particles[:, :, target] + W @ noise
            
        new_existences = survival_probability * old_existences
        
        return new_particles, new_existences
    
    def evaluate_measurements(self, alphas: torch.Tensor, alphas_existence: torch.Tensor,
                            measurements: torch.Tensor, sensor_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate measurement likelihood factors."""
        clutter_distribution = self.parameters['clutter_distribution']
        mean_clutter = self.parameters['mean_clutter']
        detection_probability = self.parameters['detection_probability']
        measurement_var_range = self.parameters['measurement_variance_range']
        measurement_var_bearing = self.parameters['measurement_variance_bearing']
        sensor_positions = self.parameters['sensor_positions']
        
        num_measurements = measurements.shape[1]
        num_targets = alphas.shape[2]
        
        beta_messages = torch.zeros(num_measurements + 1, num_targets, device=self.device)
        v_factors1 = torch.zeros(num_measurements + 1, num_targets, self.num_particles, device=self.device)
        
        if num_targets == 0:
            return beta_messages, v_factors1
            
        # Non-detection factor
        v_factors1[0, :, :] = 1 - detection_probability
        
        constant_factor = (1 / (2 * math.pi * math.sqrt(measurement_var_bearing * measurement_var_range)) *
                          detection_probability / (mean_clutter * clutter_distribution))
        
        for target in range(num_targets):
            # Predicted measurements
            dx = alphas[0, :, target] - sensor_positions[0, sensor_index]
            dy = alphas[1, :, target] - sensor_positions[1, sensor_index]
            predicted_range = torch.sqrt(dx**2 + dy**2)
            predicted_bearing = torch.atan2(dx, dy) * 180 / math.pi
            
            for measurement in range(num_measurements):
                range_diff = measurements[0, measurement] - predicted_range
                bearing_diff = self.wrap_to_180(measurements[1, measurement] - predicted_bearing)
                
                range_likelihood = torch.exp(-0.5 * range_diff**2 / measurement_var_range)
                bearing_likelihood = torch.exp(-0.5 * bearing_diff**2 / measurement_var_bearing)
                
                v_factors1[measurement + 1, target, :] = constant_factor * range_likelihood * bearing_likelihood
        
        # Compute beta messages
        v_factors0 = torch.zeros(num_measurements + 1, num_targets, device=self.device)
        v_factors0[0, :] = 1
        
        existence = alphas_existence.unsqueeze(0).expand(num_measurements + 1, -1)
        beta_messages = existence * torch.mean(v_factors1, dim=2) + (1 - existence) * v_factors0
        
        return beta_messages, v_factors1
    
    def wrap_to_180(self, angles: torch.Tensor) -> torch.Tensor:
        """Wrap angles to [-180, 180] degrees."""
        return ((angles + 180) % 360) - 180
    
    def introduce_new_pts(self, measurements: torch.Tensor, sensor: int, step: int,
                         unknown_number: float, unknown_particles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Introduce new particle tracks for measurements."""
        detection_probability = self.parameters['detection_probability']
        clutter_intensity = self.parameters['mean_clutter'] * self.parameters['clutter_distribution']
        sensor_position = self.parameters['sensor_positions'][:, sensor]
        surveillance_region = self.parameters['surveillance_region']
        
        num_measurements = measurements.shape[1]
        
        # Compute unknown intensity
        area = ((surveillance_region[1, 0] - surveillance_region[0, 0]) *
                (surveillance_region[1, 1] - surveillance_region[0, 1]))
        unknown_intensity = unknown_number / area
        unknown_intensity *= (1 - detection_probability) ** (sensor - 1)
        
        # Calculate constants for uniform distribution
        if num_measurements > 0:
            constants = self.calculate_constants_uniform(sensor_position, measurements, unknown_particles)
        else:
            constants = torch.zeros(0, device=self.device)
        
        # Initialize new particle tracks
        new_pts = torch.zeros(4, self.num_particles, num_measurements, device=self.device)
        new_labels = torch.zeros(3, num_measurements, device=self.device)
        xi_messages = torch.zeros(num_measurements, device=self.device)
        
        for measurement in range(num_measurements):
            new_pts[:, :, measurement] = self.sample_from_likelihood(
                measurements[:, measurement], sensor, self.num_particles
            )
            new_labels[:, measurement] = torch.tensor([step, sensor, measurement], device=self.device)
            xi_messages[measurement] = 1 + (constants[measurement] * unknown_intensity * detection_probability) / clutter_intensity
        
        new_existences = xi_messages - 1
        
        return new_pts, new_labels, new_existences, xi_messages
    
    def calculate_constants_uniform(self, sensor_position: torch.Tensor, measurements: torch.Tensor,
                                  particles: torch.Tensor) -> torch.Tensor:
        """Calculate constants for uniform distribution."""
        measurement_var_range = self.parameters['measurement_variance_range']
        measurement_var_bearing = self.parameters['measurement_variance_bearing']
        surveillance_region = self.parameters['surveillance_region']
        
        num_measurements = measurements.shape[1]
        num_particles = particles.shape[1]
        
        area = ((surveillance_region[1, 0] - surveillance_region[0, 0]) *
                (surveillance_region[1, 1] - surveillance_region[0, 1]))
        constant_weight = 1 / area
        
        dx = particles[0, :] - sensor_position[0]
        dy = particles[1, :] - sensor_position[1]
        predicted_range = torch.sqrt(dx**2 + dy**2)
        predicted_bearing = torch.atan2(dx, dy) * 180 / math.pi
        
        constant_likelihood = 1 / (2 * math.pi * math.sqrt(measurement_var_bearing * measurement_var_range))
        
        constants = torch.zeros(num_measurements, device=self.device)
        for measurement in range(num_measurements):
            range_likelihood = torch.exp(-0.5 * (measurements[0, measurement] - predicted_range)**2 / measurement_var_range)
            bearing_likelihood = torch.exp(-0.5 * (measurements[1, measurement] - predicted_bearing)**2 / measurement_var_bearing)
            
            constants[measurement] = torch.sum(constant_likelihood * range_likelihood * bearing_likelihood) / num_particles
        
        constants = constants / constant_weight
        return constants
    
    def sample_from_likelihood(self, measurement: torch.Tensor, sensor_index: int, num_particles: int) -> torch.Tensor:
        """Sample particles from measurement likelihood."""
        sensor_position = self.parameters['sensor_positions'][:, sensor_index]
        measurement_var_range = self.parameters['measurement_variance_range']
        measurement_var_bearing = self.parameters['measurement_variance_bearing']
        prior_velocity_covariance = self.parameters['prior_velocity_covariance']
        
        samples = torch.zeros(4, num_particles, device=self.device)
        
        # Sample range and bearing
        random_range = measurement[0] + torch.randn(num_particles, device=self.device) * math.sqrt(measurement_var_range)
        random_bearing = measurement[1] + torch.randn(num_particles, device=self.device) * math.sqrt(measurement_var_bearing)
        
        # Convert to Cartesian coordinates
        samples[0, :] = sensor_position[0] + random_range * torch.sin(random_bearing * math.pi / 180)
        samples[1, :] = sensor_position[1] + random_range * torch.cos(random_bearing * math.pi / 180)
        
        # Sample velocities
        velocity_samples = torch.randn(2, num_particles, device=self.device)
        chol = torch.linalg.cholesky(prior_velocity_covariance)
        samples[2:4, :] = chol @ velocity_samples
        
        return samples
    
    def perform_data_association_bp(self, input_legacy: torch.Tensor, input_new: torch.Tensor,
                                   check_convergence: int = 20, threshold: float = 1e-5,
                                   num_iterations: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform belief propagation for data association."""
        if input_legacy.shape[1] == 0 or input_legacy.shape[0] == 1:
            return torch.ones_like(input_legacy[1:, :]), torch.ones_like(input_new)
        
        num_measurements = input_legacy.shape[0] - 1
        num_objects = input_legacy.shape[1]
        
        output_legacy = torch.ones(num_measurements, num_objects, device=self.device)
        output_new = torch.ones(num_measurements, device=self.device)
        
        if num_objects == 0 or num_measurements == 0:
            return output_legacy, output_new
        
        messages2 = torch.ones(num_measurements, num_objects, device=self.device)
        
        for iteration in range(num_iterations):
            messages2_old = messages2.clone()
            
            product1 = messages2 * input_legacy[1:, :]
            sum1 = input_legacy[0, :].unsqueeze(0) + torch.sum(product1, dim=0, keepdim=True)
            messages1 = input_legacy[1:, :] / (sum1.expand_as(product1) - product1)
            
            sum2 = input_new.unsqueeze(1) + torch.sum(messages1, dim=1, keepdim=True)
            messages2 = 1 / (sum2.expand_as(messages1) - messages1)
            
            if iteration % check_convergence == 0 and iteration > 0:
                distance = torch.max(torch.abs(torch.log(messages2 / messages2_old)))
                if distance < threshold:
                    break
        
        output_legacy = messages2
        
        output_new_full = torch.cat([torch.ones(num_measurements, 1, device=self.device), messages1], dim=1)
        output_new_full = output_new_full / torch.sum(output_new_full, dim=1, keepdim=True)
        output_new = output_new_full[:, 0]
        
        return output_legacy, output_new
    
    def resample_systematic(self, weights: torch.Tensor, num_particles: int) -> torch.Tensor:
        """Systematic resampling."""
        cum_weights = torch.cumsum(weights, dim=0)
        
        # Generate systematic grid
        u = torch.rand(1, device=self.device) / num_particles
        grid = torch.linspace(0, (num_particles - 1) / num_particles, num_particles, device=self.device) + u
        
        # Find indices
        indices = torch.searchsorted(cum_weights, grid, right=False)
        indices = torch.clamp(indices, 0, len(cum_weights) - 1)
        
        return indices
    
    def update_pts(self, kappas: torch.Tensor, iotas: torch.Tensor, legacy_pts: torch.Tensor,
                  new_pts: torch.Tensor, legacy_existences: torch.Tensor, new_existences: torch.Tensor,
                  legacy_labels: torch.Tensor, new_labels: torch.Tensor, v_factors1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update particle tracks."""
        num_targets = v_factors1.shape[1]
        num_measurements = v_factors1.shape[0] - 1
        
        for target in range(num_targets):
            weights = v_factors1[0, target, :]
            for measurement in range(num_measurements):
                weights = weights + kappas[measurement, target] * v_factors1[measurement + 1, target, :]
            
            sum_weights = torch.sum(weights)
            
            # Update existence probability
            is_alive = legacy_existences[target] * sum_weights / self.num_particles
            is_dead = 1 - legacy_existences[target]
            legacy_existences[target] = is_alive / (is_alive + is_dead)
            
            # Resample particles
            if sum_weights > 0:
                normalized_weights = weights / sum_weights
                indices = self.resample_systematic(normalized_weights, self.num_particles)
                legacy_pts[:, :, target] = legacy_pts[:, indices, target]
        
        # Merge new and legacy particle tracks
        if new_pts.shape[2] > 0:
            legacy_pts = torch.cat([legacy_pts, new_pts], dim=2)
            
            new_existences = iotas * new_existences / (iotas * new_existences + 1)
            legacy_existences = torch.cat([legacy_existences, new_existences])
            legacy_labels = torch.cat([legacy_labels, new_labels], dim=1)
        
        return legacy_pts, legacy_existences, legacy_labels
    
    def track_step(self, cluttered_measurements: List[torch.Tensor], unknown_number: float,
                  unknown_particles: torch.Tensor, step: int) -> Dict:
        """Perform one tracking step."""
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
            
            # Pruning
            if self.posterior_existences.shape[0] > 0:
                keep_mask = self.posterior_existences >= self.threshold_pruning
                self.posterior_beliefs = self.posterior_beliefs[:, :, keep_mask]
                self.posterior_existences = self.posterior_existences[keep_mask]
                self.posterior_labels = self.posterior_labels[:, keep_mask]
        
        # Generate estimates
        estimated_cardinality = torch.sum(self.posterior_existences).item()
        estimates = self.generate_estimates()
        
        return {
            'estimates': estimates,
            'estimated_cardinality': estimated_cardinality,
        }
    
    def generate_estimates(self) -> Dict:
        """Generate state estimates from particle tracks."""
        estimates = {'state': [], 'label': []}
        
        if self.posterior_existences.shape[0] == 0:
            return estimates
            
        detected_mask = self.posterior_existences > self.detection_threshold
        
        if torch.any(detected_mask):
            detected_beliefs = self.posterior_beliefs[:, :, detected_mask]
            detected_labels = self.posterior_labels[:, detected_mask]
            
            # Compute mean states
            mean_states = torch.mean(detected_beliefs, dim=1)
            
            estimates['state'] = mean_states
            estimates['label'] = detected_labels
            
        return estimates