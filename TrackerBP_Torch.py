import torch
import numpy as np
from typing import Dict, Any

# A helper function to wrap angles to [-180, 180]
def wrap_to_180(angle_deg: torch.Tensor) -> torch.Tensor:
    """Wraps an angle in degrees to the interval [-180, 180]."""
    return (angle_deg + 180) % 360 - 180

class TrackerBP_PyTorch:
    """
    A PyTorch implementation of the classical Belief Propagation multi-target tracker.
    This version is a direct structural mapping of the original NumPy version,
    with vectorized operations for parallel execution.
    """
    def __init__(self, parameters: Dict[str, Any], sensor_pos: torch.Tensor, device: str = 'cpu'):
        self.params = parameters
        self.device = device
        self.sensor_pos = sensor_pos.to(device=self.device, dtype=torch.float32)

        # --- Model Parameters ---
        self.p_s = self.params['p_s']
        self.p_d = self.params['p_d']
        self.num_particles = self.params['num_particles']
        self.d_t = self.params['d_t']
        
        # --- Birth and Clutter ---
        self.birth_intensity = self.params['mu_n'] / (2 * np.pi * self.params['measurement_range'] ** 2)
        self.clutter_intensity = self.params['mu_c'] * self.params['f_c']
        
        # --- Variances ---
        self.var_range = self.params['range_variance']
        self.var_bearing = self.params['bearing_variance']
        self.prior_velocity_cov = torch.tensor(self.params['velocity_noise'], device=self.device, dtype=torch.float32)

        # --- Thresholds ---
        self.pruning_threshold = self.params['pruning_threshold']
        self.detection_threshold = self.params['detection_threshold']
        
        # --- State Transition Model (Constant Velocity) ---
        self.F = torch.tensor([[1, 0, self.d_t, 0],
                               [0, 1, 0, self.d_t],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], device=self.device, dtype=torch.float32)
        
        self.Q = torch.tensor([[self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2, 0],
                               [0, self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2],
                               [self.d_t ** 3 / 2, 0, self.d_t ** 2, 0],
                               [0, self.d_t ** 3 / 2, 0, self.d_t ** 2]], device=self.device, dtype=torch.float32)
        self.L_Q = torch.linalg.cholesky(self.Q * self.params['sigma_v']**2)

        # --- Tracker State Variables ---
        # All state variables are kept on the specified device to minimize CPU-GPU communication.
        self.gamma_states = torch.empty((4, self.num_particles, 0), device=self.device, dtype=torch.float32)
        self.gamma_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.gamma_labels = torch.empty((0,), device=self.device, dtype=torch.long)
        
        # Alpha: The predicted belief
        self.alpha_states = torch.empty((4, self.num_particles, 0), device=self.device, dtype=torch.float32)
        self.alpha_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.alpha_labels = torch.empty((0,), device=self.device, dtype=torch.long)
        
        # Varsigma: The belief for new tracks born from measurements
        self.varsigma_states = torch.empty((4, self.num_particles, 0), device=self.device, dtype=torch.float32)
        self.varsigma_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.varsigma_labels = torch.empty((0,), device=self.device, dtype=torch.long)

        # Intermediate messages for data association
        self.xi = torch.empty((0,), device=self.device)
        self.beta = torch.empty((0, 0), device=self.device)
        self.kappa = torch.empty((0, 0), device=self.device)
        self.iota = torch.empty((0,), device=self.device)
        self.log_likelihood_table = torch.empty((0,0,0), device=self.device)
        
        self.max_label = 0

    def compute_alpha(self) -> None:
        """
        Performs the prediction step (predicts `alpha` from `gamma` of previous time step).
        """
        num_targets = self.gamma_states.shape[2]
        if num_targets == 0:
            self.alpha_states = torch.empty((4, self.num_particles, 0), device=self.device)
            self.alpha_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
            self.alpha_labels = torch.empty((0,), device=self.device, dtype=torch.long)
            return

        # 1. Propagate states using the motion model (vectorized)
        # Reshape for batched matrix multiplication: [4, P, T] -> [4, P*T]
        gamma_states_flat = self.gamma_states.view(4, -1)
        predicted_states_flat = self.F @ gamma_states_flat
        # Reshape back to [4, P, T]
        predicted_states = predicted_states_flat.view(4, self.num_particles, num_targets)

        # 2. Add process noise (vectorized)
        noise = self.L_Q @ torch.randn(4, num_targets * self.num_particles, device=self.device)
        noise = noise.view(4, num_targets, self.num_particles).permute(0, 2, 1) # Reshape to [4, P, T]
        
        self.alpha_states = predicted_states + noise
        self.alpha_labels = self.gamma_labels.clone()
        self.alpha_existence = self.gamma_existence * self.p_s
        
    def compute_xi_sigma(self, measurements: torch.Tensor) -> None:
        """
        Computes the initial belief (`varsigma`) and message (`xi`) for new potential tracks.
        The initial distribution of the new tracks is based on the measurements and the sensor position.
        which is a heuristic.
        """
        num_meas = measurements.shape[1] if measurements.nelement() > 0 else 0
        if num_meas == 0:
            self.varsigma_states = torch.empty((4, self.num_particles, 0), device=self.device)
            self.varsigma_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
            self.varsigma_labels = torch.empty((0,), device=self.device, dtype=torch.long)
            self.xi = torch.empty((0,), device=self.device)
            return

        # --- Sample new particles for each measurement ---
        # Unsqueeze to allow broadcasting over particles
        ranges = measurements[0].unsqueeze(1) + torch.sqrt(torch.tensor(self.var_range, device=self.device)) * torch.randn(num_meas, self.num_particles, device=self.device)
        bearings_rad = torch.deg2rad(measurements[1].unsqueeze(1) + torch.sqrt(torch.tensor(self.var_bearing, device=self.device)) * torch.randn(num_meas, self.num_particles, device=self.device))
        
        # Transpose to get shape [4, num_particles, num_meas]
        self.varsigma_states = torch.zeros(4, self.num_particles, num_meas, device=self.device)
        self.varsigma_states[0, :, :] = self.sensor_pos[0] + (ranges * torch.cos(bearings_rad)).T
        self.varsigma_states[1, :, :] = self.sensor_pos[1] + (ranges * torch.sin(bearings_rad)).T
        
        L_vel = torch.linalg.cholesky(self.prior_velocity_cov)
        velocities = (L_vel @ torch.randn(2, num_meas * self.num_particles, device=self.device))
        self.varsigma_states[2:4, :, :] = velocities.view(2, num_meas, self.num_particles).permute(0, 2, 1)

        # --- Assign new labels ---
        self.varsigma_labels = torch.arange(self.max_label + 1, self.max_label + 1 + num_meas, device=self.device, dtype=torch.long)

        # --- Compute initial existence and message `xi` ---
        birth_clutter_ratio = (self.birth_intensity * self.p_d) / (self.clutter_intensity + 1e-9)
        self.varsigma_existence = torch.full((num_meas,), birth_clutter_ratio, device=self.device)
        self.xi = 1.0 + self.varsigma_existence

    def compute_beta(self, measurements: torch.Tensor) -> None:
        """
        Computes the measurement likelihood factors and initial messages (`beta`) for existing tracks.
        This version operates in log-space for numerical stability.
        """
        num_targets = self.alpha_states.shape[2]
        num_meas = measurements.shape[1] if measurements.nelement() > 0 else 0

        self.beta = torch.zeros(num_meas + 1, num_targets, device=self.device)
        self.log_likelihood_table = torch.zeros(num_meas + 1, num_targets, self.num_particles, device=self.device)
        
        if num_targets == 0:
            return

        # Log-likelihood for miss-detection (measurement index 0)
        self.log_likelihood_table[0, :, :] = torch.log(torch.tensor(1 - self.p_d, device=self.device) + 1e-9)
        
        if num_meas > 0:
            # --- Vectorized Log-Likelihood Calculation ---
            pos_states = self.alpha_states[:2, :, :] # Shape: [2, P, T]
            diff = pos_states - self.sensor_pos.view(2, 1, 1)
            
            pred_range = torch.norm(diff, p=2, dim=0) # Shape: [P, T]
            pred_bearing_deg = torch.rad2deg(torch.atan2(diff[1, :, :], diff[0, :, :])) # Shape: [P, T]

            range_error = measurements[0].view(1, -1) - pred_range.unsqueeze(-1) # Shape [P, T, M]
            bearing_error = wrap_to_180(measurements[1].view(1, -1) - pred_bearing_deg.unsqueeze(-1)) # Shape [P, T, M]
            
            # Log of the normalization factor for the Gaussian likelihood
            log_norm_factor = -torch.log(2 * torch.tensor(np.pi, device=self.device)) - 0.5 * (torch.log(torch.tensor(self.var_range, device=self.device) + 1e-9) + torch.log(torch.tensor(self.var_bearing, device=self.device) + 1e-9))
            
            # Log-likelihood components
            log_range_exp = -0.5 * (range_error ** 2) / (self.var_range + 1e-9)
            log_bearing_exp = -0.5 * (bearing_error ** 2) / (self.var_bearing + 1e-9)
            
            # Total log-likelihood, transposed to get [M, T, P]
            log_likelihood = log_norm_factor + log_range_exp + log_bearing_exp
            self.log_likelihood_table[1:, :, :] = log_likelihood.permute(2, 1, 0)

        # --- Compute `beta` messages using log-sum-exp for stability ---
        v_factors = torch.zeros_like(self.beta)
        v_factors[0, :] = 1.0 # v-factor is 1 for miss-detection, 0 otherwise
        
        existence = self.alpha_existence.unsqueeze(0)
        
        # Use log-sum-exp to average likelihoods in a stable way
        avg_log_likelihood = torch.logsumexp(self.log_likelihood_table, dim=2) - torch.log(torch.tensor(self.num_particles, device=self.device))
        avg_likelihood = torch.exp(avg_log_likelihood)
        
        self.beta = existence * avg_likelihood + (1 - existence) * v_factors

    def compute_kappa_iota(self, num_iter: int = 5) -> None:
        """
        Performs Loopy Belief Propagation to compute data association messages.
        reference for the PyTorch implementation:
        # mu_ba: message from measurement 'b' to target 'a'
        mu_ba = torch.ones(batch_size, num_max_targets, num_max_meas, device = message_in.device)
        message_in = message_in / torch.sum(message_in, dim = -1, keepdim = True)
        
        # Create a mask for valid (target, measurement) pairs.
        total_mask = torch.logical_and(targets_mask[:, :, None], measurements_mask[:, None, :])
        for i in range(num_bp_iter):
            mu_ba_old = mu_ba.clone()

            # mu_ab: message from target 'a' to measurement 'b'.
            mu_ab = message_in[:, :, 1:] / (torch.sum(message_in[:, :, 1:] * mu_ba, dim = 2, keepdim = True) - message_in[:, :, 1:] * mu_ba + message_in[:, :, [0]] + 1e-7)
            # mu_ba: message from measurement 'b' to target 'a'.
            mu_ba = total_mask.float() / (torch.sum(mu_ab, dim = 1, keepdim = True) - mu_ab + message_new_in[:, None, :] + 1e-7)

            assert torch.all(mu_ab[~total_mask] == 0)
            assert torch.all(mu_ba[~total_mask] == 0), 'mu_ba is {:.4f}'.format(torch.sum(torch.abs(mu_ba[~total_mask])))

            # Check for convergence to break early if messages stabilize.
            improvements = (torch.sum((mu_ba_old - mu_ba) ** 2, dim = [1, 2]) / torch.sum(total_mask.float(), dim = [1, 2])) ** 0.5
            if torch.all(improvements < threshold):
                break
        """
        num_meas = self.beta.shape[0] - 1
        num_targets = self.beta.shape[1]
        
        if num_targets == 0 or num_meas == 0:
            self.kappa = torch.empty((num_meas, num_targets), device=self.device)
            self.iota = torch.ones_like(self.xi)
            return

        # Normalize beta messages for stability
        beta_normalized = self.beta / (torch.sum(self.beta, dim=0, keepdim=True) + 1e-9)

        # kappa: Message from measurement 'j' to target 'i'
        kappa = torch.ones(num_meas, num_targets, device=self.device)
        
        for _ in range(num_iter):
            # Message from target 'i' to measurement 'j' (mu_ab)
            # Denominator: Sum of all incoming messages to target 'i'
            denom_sum = beta_normalized[0, :] + torch.sum(beta_normalized[1:, :] * kappa, dim=0)
            mu_ab_denom = denom_sum.unsqueeze(0) - beta_normalized[1:, :] * kappa + 1e-9
            mu_ab = beta_normalized[1:, :] / mu_ab_denom

            # Message from measurement 'j' to target 'i' (recomputing kappa)
            denom_sum_j = self.xi + torch.sum(mu_ab, dim=1)
            kappa_denom = denom_sum_j.unsqueeze(1) - mu_ab + 1e-9
            kappa = 1.0 / kappa_denom
        
        self.kappa = kappa
        
        # Compute iota: posterior probability of a measurement being unassociated
        final_sum_j = torch.sum(mu_ab, dim=1)
        self.iota = 1.0 / (self.xi + final_sum_j + 1e-9)

    def compute_gamma(self) -> None:
        """
        Updates the posterior state (`gamma`) by resampling particles and merging new tracks.
        """
        num_targets = self.alpha_states.shape[2]
        
        # --- Update existing tracks ---
        if num_targets > 0:
            # Calculate posterior particle log-weights
            log_weights_unnorm = self.log_likelihood_table[1:, :, :].permute(1, 0, 2) + torch.log(self.kappa + 1e-9).T.unsqueeze(-1)
            
            # Combine with miss-detection log-likelihood
            log_weights_unnorm = torch.cat([self.log_likelihood_table[0, :, :].unsqueeze(1), log_weights_unnorm], dim=1)
            
            # Update existence probability using log-sum-exp for stability
            log_sum_weights = torch.logsumexp(log_weights_unnorm, dim=(1, 2)) - torch.log(torch.tensor(self.num_particles, device=self.device))
            sum_weights = torch.exp(log_sum_weights)
            
            is_alive = self.alpha_existence * sum_weights
            is_dead = 1 - self.alpha_existence
            self.gamma_existence = is_alive / (is_alive + is_dead + 1e-12)
            self.gamma_labels = self.alpha_labels.clone()
            
            # Resample particles from posterior distribution
            # The posterior over association variables is used to select a single association hypothesis for each track
            # Then particles are resampled from the likelihood of that hypothesis
            
            # Posterior over associations
            post_assoc = torch.exp(log_weights_unnorm - torch.max(log_weights_unnorm, dim=1, keepdim=True)[0])
            post_assoc_sum = torch.sum(post_assoc, dim=1, keepdim=True)
            post_assoc_norm = post_assoc / (post_assoc_sum + 1e-9)
            
            # Sample one association hypothesis for each target
            # Reshape for multinomial sampling: [T, M+1, P] -> [T, (M+1)*P]
            post_assoc_reshaped = post_assoc_norm.view(num_targets, -1)
            assoc_indices = torch.multinomial(post_assoc_reshaped, 1).squeeze(1)
            
            # Unravel indices to get association and particle index
            assoc_hyp = assoc_indices // self.num_particles
            particle_idx = assoc_indices % self.num_particles

            # Get the weights for the chosen association
            log_weights = torch.gather(log_weights_unnorm, 1, assoc_hyp.view(-1, 1, 1).expand(-1, -1, self.num_particles)).squeeze(1)

            # Convert log-weights to weights by subtracting the max for stability
            weights = torch.exp(log_weights - torch.max(log_weights, dim=1, keepdim=True)[0])
            norm_weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-12)
            indices = torch.multinomial(norm_weights, self.num_particles, replacement=True) # [T, P]
            
            # Gather resampled particles.
            # Create indices for gathering: [T, P] -> [1, P, T]
            indices_reshaped = indices.T.unsqueeze(0)
            # Expand to match state dimensions: [4, P, T]
            indices_expanded = indices_reshaped.expand(self.alpha_states.shape[0], -1, -1)
            self.gamma_states = torch.gather(self.alpha_states, 2, indices_expanded)
        else: # No existing tracks, initialize gamma from empty
            self.gamma_states = torch.empty((4, self.num_particles, 0), device=self.device)
            self.gamma_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
            self.gamma_labels = torch.empty((0,), device=self.device, dtype=torch.long)

        # --- Update and merge new tracks ---
        if self.varsigma_states.shape[2] > 0:
            # The existence probability of new tracks is scaled by iota
            new_existence = (self.iota * self.varsigma_existence) / (self.iota * self.varsigma_existence + 1 + 1e-12)
            
            self.gamma_states = torch.cat([self.gamma_states, self.varsigma_states], dim=2)
            self.gamma_existence = torch.cat([self.gamma_existence, new_existence])
            self.gamma_labels = torch.cat([self.gamma_labels, self.varsigma_labels])
            
            # Update max label
            if self.varsigma_labels.nelement() > 0:
                self.max_label = torch.max(self.varsigma_labels).item()

    def prune(self):
        """
        Prunes tracks with an existence probability below a threshold.
        """
        if self.gamma_existence.nelement() == 0:
            return

        keep_idx = self.gamma_existence >= self.pruning_threshold
        if torch.any(keep_idx):
            self.gamma_states = self.gamma_states[:, :, keep_idx]
            self.gamma_existence = self.gamma_existence[keep_idx]
            self.gamma_labels = self.gamma_labels[keep_idx]
        else: # If all tracks are pruned
            self.gamma_states = torch.empty((4, self.num_particles, 0), device=self.device)
            self.gamma_existence = torch.empty((0,), device=self.device, dtype=torch.float32)
            self.gamma_labels = torch.empty((0,), device=self.device, dtype=torch.long)

    def estimate_state(self) -> Dict[str, Any]:
        """
        Estimates the state for targets with existence probability above the detection threshold.
        """
        estimates = {}
        if self.gamma_existence.nelement() == 0:
            return estimates
        
        detected_idx = self.gamma_existence > self.detection_threshold
        if not torch.any(detected_idx):
            return estimates

        detected_states = self.gamma_states[:, :, detected_idx]
        mean_states = torch.mean(detected_states, dim=1) # Average over particles
        
        estimates['state'] = mean_states.cpu().numpy()
        estimates['label'] = self.gamma_labels[detected_idx].cpu().numpy()
        estimates['existence'] = self.gamma_existence[detected_idx].cpu().numpy()
        
        return estimates