# ------------------------------------------------------------------------
# Neural Enhanced Belief Propagation for Multiobject Tracking
# Copyright (c) 2025 MIngchao Liang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as func

def perform_prediction(particles_kinematic_prior,
                       prob_existence_prior,
                       targets_id,
                       scan_time,
                       parameters):
    """
    Performs the prediction step of the particle filter for each existing target.
    It predicts the future state of particles based on a motion model.

    Args:
        particles_kinematic_prior (Tensor): The state of all particles for all targets from the previous step.
                                            Shape: [batch_size, num_targets, num_particles, dim_state]
        prob_existence_prior (Tensor): The probability of existence for each target from the previous step.
                                       Shape: [batch_size, num_targets]
        targets_id (Tensor): The ID for each target.
        scan_time (float): The time elapsed since the last scan (delta t).
        parameters (dict): A dictionary of model hyperparameters.

    Returns:
        Tuple[Tensor, Tensor]:
        - particles_kinematic_predict: The predicted particle states.
        - prob_existence_predict: The predicted probability of existence.
    """
    # Get the shape dimensions for clarity
    batch_size, num_targets, num_particles, dim_state = particles_kinematic_prior.shape
    
    # Create a mask to identify which target slots are currently active (not -1)
    targets_mask = targets_id != -1

    # 1. Predict the probability of existence.
    # This is modeled as the prior probability of existence multiplied by a constant survival probability.
    # A target "survives" if it still exists in the current frame.
    prob_existence_predict = prob_existence_prior * parameters['prob_survival']

    # 2. Predict the kinematic states of the particles.
    particles_kinematic_predict = particles_kinematic_prior.clone()
    
    # The code currently only implements a linear prediction model (Constant Velocity).
    if parameters['linear_prediction']:
        # F is the state transition matrix for a Constant Velocity (CV) model.
        # It updates position based on velocity: x_k = x_{k-1} + v_{k-1} * dt
        F = torch.tensor([[1, 0, scan_time, 0],
                          [0, 1, 0, scan_time],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], device = particles_kinematic_prior.device)
        
        # This matrix would be for mapping process noise to the state, but it is not explicitly used below.
        W = torch.tensor([[scan_time ** 2 / 2, 0],
                          [0, scan_time ** 2 / 2],
                          [scan_time, 0],
                          [0, scan_time]], device = particles_kinematic_prior.device)

        # Apply the state transition matrix to the first 4 state dimensions [x, y, vx, vy].
        # Unsqueeze/squeeze are used to make dimensions compatible for matrix multiplication.
        particles_kinematic_predict[:, :, :, : 4] = torch.matmul(F, particles_kinematic_prior[:, :, :, : 4].unsqueeze(-1)).squeeze(-1) + \
        # Add process noise (random motion) to account for unmodeled dynamics (e.g., acceleration).
        torch.randn(batch_size, num_targets, num_particles, 4, device = particles_kinematic_prior.device) * (torch.tensor(parameters['driving_var_xy2'], device = particles_kinematic_prior.device) ** 0.5)  
        
        # This part for predicting the 'z' coordinate is not implemented.
        if parameters['use_z']:
            raise ValueError('z prediction not implemented')

        # Zero out the predictions for any targets that are not active.
        particles_kinematic_predict[~targets_mask, :, :] = 0
    else:
        raise ValueError('nonliner prediction not implemented')

    return particles_kinematic_predict, prob_existence_predict

def generate_new_beliefs(measurements,
                         measurements_mask,
                         count_id,
                         parameters):
    """
    Generates new potential targets (beliefs) from the current set of measurements.
    This is the "birth" model of the filter.

    Args:
        measurements (Tensor): The raw measurements from the current scan.
        measurements_mask (Tensor): A mask indicating which measurements are valid.
        count_id (int): The starting ID to assign to new targets.
        parameters (dict): A dictionary of model hyperparameters.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
        - particles_kinematic_new: Initialized particles for each new potential target.
        - targets_id_new: Unique IDs assigned to these new targets.
        - message_new_in: The initial belief/message that a measurement is a new target, for BP.
        - constants: The factor used to compute the message (birth/clutter ratio).
    """
    batch_size, num_max_meas = measurements_mask.shape
    # Initialize tensors to hold the state for new targets, assuming one new target per measurement.
    particles_kinematic_new = torch.zeros(batch_size, num_max_meas, parameters['num_particles'], 6, device = measurements.device) # State: [x, y, vx, vy, turn_rate, z]
    targets_id_new = torch.arange(num_max_meas, dtype = torch.long, device = measurements.device).repeat(batch_size, 1) + count_id
    targets_id_new[~measurements_mask] = -1 # Invalidate IDs for invalid measurements.

    particles_kinematic_new_list = []
    # -----------------------------
    # init x and y
    # -----------------------------
    # Initialize particle positions by sampling from a Gaussian centered at the measurement's position.
    particles_kinematic_new_list.append(measurements[measurements_mask, :][..., None, :2] +
        torch.randn(batch_size, num_max_meas, parameters['num_particles'], 2, device = measurements.device)[measurements_mask, :, :] * (parameters['meas_var_xy'] ** 0.5))

    # -----------------------------
    # init velocity
    # -----------------------------
    # If velocity is part of the measurement, initialize particles around that velocity.
    if parameters['use_velocity']:
        particles_kinematic_new_list.append(measurements[measurements_mask, :][..., None, 8 : 10] +
        torch.randn(batch_size, num_max_meas, parameters['num_particles'], 2, device = measurements.device)[measurements_mask, :, :] * (parameters['meas_var_velocity'] ** 0.5))
    # If not, initialize velocity from a wide, zero-mean Gaussian prior.
    else:
        particles_kinematic_new_list.append(torch.randn(batch_size, num_max_meas, parameters['num_particles'], 2, device = measurements.device)[measurements_mask, :, :] * (parameters['prior_var_velocity'] ** 0.5))

    # -----------------------------
    # init turn rate
    # -----------------------------
    # Sampled uniformly from 0 to 2*pi. Not used in the CV model but part of the state vector.
    particles_kinematic_new_list.append(torch.rand(batch_size, num_max_meas, parameters['num_particles'], 1, device = measurements.device)[measurements_mask, :] * np.pi * 2)

    # -----------------------------
    # init z
    # -----------------------------
    # Initialize z-coordinate similar to x and y.
    particles_kinematic_new_list.append(measurements[measurements_mask, :][..., None, [2]] +
                                        torch.randn(batch_size, num_max_meas, parameters['num_particles'], 1, device = measurements.device)[measurements_mask, :] * (parameters['meas_var_z'] ** 0.5))

    # convert list to tensor
    # Concatenate the initialized state components into a single tensor.
    particles_kinematic_new[measurements_mask, :, :] = torch.cat(particles_kinematic_new_list, dim = -1)

    # -----------------------------
    # calculate input messages for new targets
    # -----------------------------
    # This message represents the prior belief that a measurement corresponds to a new target.
    # It is based on the ratio of the expected rate of new targets (births) to the rate of false alarms (clutter).
    integral_factor = torch.ones(batch_size, num_max_meas, device = measurements.device)
    constants = integral_factor * ((parameters['mean_newborn'] * parameters['birth_distribution']) / (parameters['mean_clutter'] * parameters['false_alarm_distribution']))
    message_new_in = torch.zeros_like(measurements_mask, dtype=torch.float)
    # The message is 1 + C, where C is the birth-to-clutter ratio.
    message_new_in[measurements_mask] = 1 + constants[measurements_mask]

    return particles_kinematic_new, targets_id_new, message_new_in, constants

def loopy_bp_da(message_in,
                message_new_in,
                prob_existence_new_deep,
                targets_mask,
                measurements_mask,
                dist_mask,
                threshold: float,
                num_bp_iter: int,
                num_gnn_iter: int,
                nebp = None,
                h_a = None,
                h_b = None):
    """
    Performs Loopy Belief Propagation (BP) for Data Association (DA).
    This function iteratively calculates the probability of association between targets and measurements.
    It can be a standard BP or a Neural Enhanced BP (NEBP) if a GNN model is provided.

    Args:
        message_in (Tensor): Initial messages from existing targets to measurements.
                             Shape: [batch_size, num_targets, num_meas + 1] (+1 for miss-detection)
        message_new_in (Tensor): Initial messages from the birth model to measurements.
                                 Shape: [batch_size, num_meas]
        prob_existence_new_deep (Tensor): A NN-refined probability that a measurement is a new target.
        targets_mask (Tensor): Mask for active targets.
        measurements_mask (Tensor): Mask for valid measurements.
        dist_mask (Tensor): Gating mask to prune distant target-measurement pairs.
        threshold (float): Convergence threshold for BP iterations.
        num_bp_iter (int): Number of standard BP iterations.
        num_gnn_iter (int): Number of GNN enhancement iterations.
        nebp (nn.Module, optional): The Neural BP GNN model. Defaults to None.
        h_a (Tensor, optional): Hidden features for existing targets (factor nodes 'a').
        h_b (Tensor, optional): Hidden features for measurements (factor nodes 'b').

    Returns:
        A tuple containing final messages, association probabilities, and updated hidden states.
    """

    assert message_in.shape[0] == message_new_in.shape[0]
    assert message_in.shape[2] == message_new_in.shape[1] + 1
    assert num_bp_iter >= 0
    assert num_gnn_iter >= 0

    batch_size, num_max_meas = message_new_in.shape
    _, num_max_targets, _ = message_in.shape
    
    # Initialize hidden states if not provided (for the GNN)
    if h_a is None and h_b is None:
        h_a = torch.randn(1)
        h_b = torch.randn(1)
    
    # Initialize output messages and probabilities
    message_out = torch.ones(batch_size, num_max_targets, num_max_meas, device = message_in.device)
    message_new_out = torch.ones_like(message_new_in)
    asso_prob = torch.ones_like(message_in)
    asso_prob_new = torch.ones_like(message_new_in)
    message_in_deep_scale = torch.ones_like(message_new_in)
    
    # Handle edge case where there's nothing to associate
    if num_max_meas == 0 or num_max_targets == 0 or num_bp_iter == 0:
        if nebp is not None:
            # The GNN can still provide a scale factor even with no associations
            message_in_deep_scale = nebp.node_s(torch.cat([h_b, message_new_in[:, :, None]], dim = -1)).squeeze(-1)
        return message_out, message_new_out, asso_prob, asso_prob_new, message_in, message_in.unsqueeze(1), message_in_deep_scale, h_a, h_b

    message_in_ori = message_in.clone()
    # Normalize input messages to be probabilities for stable calculations
    message_in = message_in / torch.sum(message_in, dim = -1, keepdim = True)
    message_in_deep_iter = message_in.clone().unsqueeze(1)
    message_in_deep_scale_iter = message_in.clone().unsqueeze(1)
    message_in_deep_list = []
    message_in_deep_scale_list = []
    message_out_deep = message_in.clone()
    # mu_ba: message from measurement 'b' to target 'a'
    mu_ba = torch.ones(batch_size, num_max_targets, num_max_meas, device = message_in.device)
    
    # Create a mask for valid (target, measurement) pairs.
    total_mask = torch.logical_and(targets_mask[:, :, None], measurements_mask[:, None, :])

    # ----------------------------------------
    # conventional BP (if nebp is None)
    # ----------------------------------------
    if nebp is None:
        # This is the standard Loopy BP algorithm without any neural network enhancement.
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

    # ----------------------------------------
    # NEBP (Neural Enhanced Belief Propagation)
    # ----------------------------------------
    else:
        # First, run a few iterations of standard BP to get initial message values.
        for i in range(num_bp_iter):
            mu_ba_old = mu_ba.clone()

            mu_ab = message_in[:, :, 1:] / (torch.sum(message_in[:, :, 1:] * mu_ba, dim = 2, keepdim = True) - message_in[:, :, 1:] * mu_ba + message_in[:, :, [0]] + 1e-7)
            mu_ba = total_mask.float() / (torch.sum(mu_ab, dim = 1, keepdim = True) - mu_ab + message_new_in[:, None, :] + 1e-7)

            assert torch.all(mu_ab[~total_mask] == 0)
            assert torch.all(mu_ba[~total_mask] == 0), 'mu_ba is {:.4f}'.format(torch.sum(torch.abs(mu_ba[~total_mask])))

            improvements = (torch.sum((mu_ba_old - mu_ba) ** 2, dim = [1, 2]) / torch.sum(total_mask.float(), dim = [1, 2])) ** 0.5
            if torch.all(improvements < threshold):
                break
        
        # Initialize tensors for GNN outputs
        message_in_deep = torch.zeros(batch_size, num_max_targets, num_max_meas + 1, device = h_a.device)
        message_in_deep_iter = message_in_deep[:, None, :, :]
        message_in_deep_scale = torch.ones(batch_size, num_max_meas, device = h_b.device)

        # Now, begin the GNN enhancement loop.
        for _ in range(num_gnn_iter):
            for _ in range(3): # This inner loop seems to be for message aggregation within the GNN
                mu_ba_ = mu_ba.clone()
                mu_ab_ = mu_ab.clone()
                # The GNN (nebp) takes current messages and hidden features and outputs corrections.
                h_a, h_b, _, _, _, _, message_in_deep, message_in_deep_scale = nebp(h_a, h_b, mu_ab_, mu_ba_,
                                                     mu_a = message_in, mu_b = message_new_in, mask = total_mask, dist_mask = dist_mask)
                with torch.no_grad():
                    # Apply distance mask to GNN output, pruning far-away corrections
                    tmp = message_in_deep[:, :, 1 :].clone()
                    tmp[~dist_mask] = -10
                    message_in_deep[:, :, 1 :] = tmp

                message_in_deep_list.append((message_in_deep).clone())
                message_in_deep_scale_list.append((message_in_deep_scale).clone())
            
            # Expand the GNN scale factors to match message dimensions.
            message_in_deep_scale_expanded = torch.cat([torch.ones(batch_size, num_max_targets, 1, device = h_b.device),
                                                        message_in_deep_scale[:, None, :].expand(-1, num_max_targets, -1) ], dim = -1)
            prob_existence_new_deep_expanded = torch.cat([torch.ones(batch_size, num_max_targets, 1, device = h_b.device),
                                                            prob_existence_new_deep[:, None, :].expand(-1, num_max_targets, -1) ], dim = -1)
            
            # This is the "neural enhancement": update the messages using the GNN's corrections.
            # The GNN learns to fix errors in the local BP messages by reasoning about the global graph.
            message_in = func.relu(prob_existence_new_deep_expanded * message_in + func.relu(message_in_deep) )
            message_new_in = 1 + (message_new_in - 1) * prob_existence_new_deep
            
            # Store the GNN outputs for this iteration
            message_in_deep_iter = torch.stack(message_in_deep_list).transpose(0, 1)
            message_in_deep_scale_iter = torch.stack(message_in_deep_scale_list).transpose(0, 1)

            message_in_ = message_in.clone()

            # After enhancement, run standard BP again with the *new, improved* messages.
            for _ in range(num_bp_iter):
                mu_ba_old = mu_ba.clone()
                mu_ab = message_in[:, :, 1:] / (torch.sum(message_in[:, :, 1:] * mu_ba, dim = 2, keepdim = True) - message_in[:, :, 1:] * mu_ba + message_in[:, :, [0]] + 1e-7)
                mu_ba = total_mask.float() / (torch.sum(mu_ab, dim = 1, keepdim = True) - mu_ab + message_new_in[:, None, :] + 1e-7)

                assert torch.all(mu_ab[~total_mask] == 0)
                assert torch.all(mu_ba[~total_mask] == 0), 'mu_ba is {:.4f}'.format(torch.sum(torch.abs(mu_ba[~total_mask])))

                improvements = (torch.sum((mu_ba_old - mu_ba) ** 2, dim = [1, 2]) / torch.sum(total_mask.float(), dim = [1, 2])) ** 0.5
                if torch.all(improvements < threshold):
                    break
        
        # ----------------------------------------
        # shape feature only (Ablation case with no GNN iterations)
        # ----------------------------------------
        if num_gnn_iter == 0:
            prob_existence_new_deep_expanded = torch.cat([torch.ones(batch_size, num_max_targets, 1, device = h_b.device),
                                                            prob_existence_new_deep[:, None, :].expand(-1, num_max_targets, -1) ], dim = -1)
            message_in = func.relu(prob_existence_new_deep_expanded * message_in)
            message_new_in = 1 + (message_new_in - 1) * prob_existence_new_deep

            for _ in range(num_bp_iter):
                mu_ba_old = mu_ba.clone()
                mu_ab = message_in[:, :, 1:] / (torch.sum(message_in[:, :, 1:] * mu_ba, dim = 2, keepdim = True) - message_in[:, :, 1:] * mu_ba + message_in[:, :, [0]] + 1e-7)
                mu_ba = total_mask.float() / (torch.sum(mu_ab, dim = 1, keepdim = True) - mu_ab + message_new_in[:, None, :] + 1e-7)

                assert torch.all(mu_ab[~total_mask] == 0)
                assert torch.all(mu_ba[~total_mask] == 0), 'mu_ba is {:.4f}'.format(torch.sum(torch.abs(mu_ba[~total_mask])))

                improvements = (torch.sum((mu_ba_old - mu_ba) ** 2, dim = [1, 2]) / torch.sum(total_mask.float(), dim = [1, 2])) ** 0.5
                if torch.all(improvements < threshold):
                    break
        

    # After iterations, calculate the final posterior association probabilities.
    # P(target 'a' is not detected)
    asso_prob[:, :, 0] = message_in[:, :, 0]
    # P(target 'a' is associated with measurement 'b')
    asso_prob[:, :, 1:] = message_in[:, :, 1:] * mu_ba
    # Normalize to get a valid probability distribution for each target.
    asso_prob = asso_prob / (torch.sum(asso_prob, dim = -1, keepdim = True) + 1e-6)

    # Calculate the posterior probability that a measurement is a new target.
    asso_prob_new = message_new_in / (torch.sum(mu_ab, dim = -2) + message_new_in)
    
    # The final messages are returned for use in the update step.
    message_out = mu_ba
    message_new_out = measurements_mask.float() / (torch.sum(mu_ab, dim = -2) + measurements_mask.float() + 1e-7)

    assert torch.all(message_out[~total_mask] == 0)
    assert torch.all(message_new_out[~measurements_mask] == 0)


    return message_out, message_new_out, asso_prob, asso_prob_new, \
           message_in, message_in_deep_iter, message_in_deep_scale, h_a, h_b


def sequential_bp(particles_kinematic_prior,
                  prob_existence_prior,
                  prob_detection,
                  targets_id,
                  measurements,
                  measurements_mask,
                  motion_feat_existing,
                  app_feat_existing,
                  motion_feat_new,
                  app_feat_new,
                  scan_time,
                  count_id: int,
                  parameters: dict,
                  nebp = None,
                  dynamic_pd = False,
                  affinity_gt = None,
                  ground_truths_label_new = None,
                  mode = 'val'):
    """
    The main driver function for one time step of the tracker.
    It orchestrates the prediction, likelihood calculation, data association, and update steps.
    """
    assert isinstance(count_id, int) and count_id >= 0
    assert mode in ['train', 'val']
    
    # Create a mask for active targets.
    targets_mask = targets_id != -1

    # ==================================
    # 1. PREDICTION STEP
    # ==================================
    # Predict the state of existing targets using the motion model.
    particles_kinematic_predict, prob_existence_predict \
        = perform_prediction(particles_kinematic_prior = particles_kinematic_prior,
                             prob_existence_prior = prob_existence_prior,
                             targets_id = targets_id,
                             scan_time = scan_time,
                             parameters = parameters)

    # ==================================
    # 2. BIRTH MODEL
    # ==================================
    # Generate new potential tracks from the current measurements.
    particles_kinematic_new, targets_id_new, message_new_in, constants \
     = generate_new_beliefs(measurements,
                            measurements_mask,
                            count_id,
                            parameters)


    batch_size, num_max_meas, dim_meas = measurements.shape
    _, num_max_targets, _, dim_state = particles_kinematic_prior.shape
    
    # ==================================
    # 3. LIKELIHOOD CALCULATION
    # ==================================
    # Calculate the likelihood of each measurement given each predicted target.
    # This answers: "How likely is it that measurement 'j' was generated by target 'i'?"
    log_likelihood = torch.zeros(batch_size, num_max_targets, num_max_meas, parameters['num_particles'], device = particles_kinematic_prior.device)
    log_likelihood_mask = torch.logical_and(targets_mask.unsqueeze(-1), measurements_mask.unsqueeze(-2))

    # constant_factor combines detection probability and clutter density.
    if not dynamic_pd:
        constant_factor = 1 / (2 * np.pi * parameters['meas_var_xy']) \
            * parameters['prob_detection'] \
            / (parameters['mean_clutter'] * parameters['false_alarm_distribution'])
        parameters['prob_detection'] = parameters['prob_detection'] * torch.ones_like(targets_mask)
    else:
        constant_factor = 1 / (2 * np.pi * parameters['meas_var_xy']) \
            * prob_detection.unsqueeze(-1) \
            / (parameters['mean_clutter'] * parameters['false_alarm_distribution'])
        parameters['prob_detection'] = prob_detection

    # -----------------------------
    # log likelihood of x
    # -----------------------------
    # The likelihood is modeled as a Gaussian. We use log-likelihood for numerical stability.
    log_likelihood_x = (particles_kinematic_predict[:, :, :, 0][:, :, None, :] - measurements[:, :, 0][:, None, :, None]) ** 2 * (- 1 / (2 * parameters['meas_var_xy']))
    log_likelihood_x[~log_likelihood_mask, :] = 0

    # -----------------------------
    # log likelihood of y
    # -----------------------------
    log_likelihood_y = (particles_kinematic_predict[:, :, :, 1][:, :, None, :] - measurements[:, :, 1][:, None, :, None]) ** 2 * (- 1 / (2 * parameters['meas_var_xy']))
    log_likelihood_y[~log_likelihood_mask, :] = 0

    log_likelihood = log_likelihood_x + log_likelihood_y


    # -----------------------------
    # log likelihood of z
    # -----------------------------
    if parameters['use_z']:
        log_likelihood_z = (particles_kinematic_predict[:, :, :, 5][:, :, None, :] - measurements[:, :, 2][:, None, :, None]) ** 2 * (- 1 / (2 * parameters['meas_var_z']))
        log_likelihood_z[~log_likelihood_mask, :] = 0

        log_likelihood = log_likelihood + log_likelihood_z
        constant_factor = constant_factor * (1 / (2 * np.pi * parameters['meas_var_z']) ** 0.5)

    # -----------------------------
    # log_likelihood of velocity
    # -----------------------------
    if parameters['use_velocity']:
        log_likelihood_velocity = torch.sum((particles_kinematic_predict[:, :, :, 2 : 4][:, :, None, :, :] - measurements[:, :, 8 : 10][:, None, :, None, :]) ** 2, dim = -1) * (- 1 / (2 * parameters['meas_var_velocity']))
        log_likelihood_velocity[~log_likelihood_mask, :] = 0

        log_likelihood = log_likelihood + log_likelihood_velocity
        constant_factor = constant_factor * (1 / (2 * np.pi * parameters['meas_var_velocity']))
    
    # Convert log-likelihood back to likelihood.
    likelihood = torch.exp(log_likelihood)
    likelihood[~log_likelihood_mask, :] = 0
    
    # If ground truth is provided (for training), use it to mask the likelihood.
    if ground_truths_label_new is not None:
        likelihood = likelihood * ground_truths_label_new[:, None, :, None]

    # -----------------------------
    # calculate dist mask
    # ----------------------------- 
    estimations_predict = torch.mean(particles_kinematic_predict, dim = -2)
    dist_mask = torch.ones(batch_size, num_max_targets, num_max_meas, device = log_likelihood.device, dtype = torch.bool)

    # -----------------------------
    # calculate input message for existing targets
    # -----------------------------
    # Prepare the initial messages for the BP algorithm.
    message_in = torch.zeros(batch_size, num_max_targets, num_max_meas + 1, device = log_likelihood.device)
    # Message for miss-detection: P(miss) = 1 - P_detection
    message_in[:, :, 0] = 1 - parameters['prob_detection']
    # Message for association: avg(likelihood over particles) * constant_factor
    message_in[:, :, 1 :] = torch.mean(likelihood, dim = -1) * constant_factor

    assert torch.all(log_likelihood[~log_likelihood_mask, :] == 0)
    assert torch.all(message_in[:, :, 1:][~log_likelihood_mask] == 0)
    
    # Weight the message by the predicted existence probability of the target.
    message_in = message_in * prob_existence_predict.unsqueeze(-1)
    message_in[:, :, 0] = message_in[:, :, 0] + (1 - prob_existence_predict)

    asso_prob_single = message_in / torch.sum(message_in, dim = -1, keepdim = True)
    
    # During training, can mix messages with ground truth affinity for guidance.
    if affinity_gt is not None:
        message_in = message_in / torch.sum(message_in, dim = -1, keepdim = True)
        message_in = message_in * 0.5 + affinity_gt * 0.5 + 1e-6
        message_in = message_in / torch.sum(message_in, dim = -1, keepdim = True)

    prob_existence_new_deep = torch.ones_like(measurements_mask, dtype = torch.float)
    h_a, h_b = None, None
    # If using NEBP, prepare features for the GNN.
    if nebp is not None:
        # Use an MLP to get a preliminary existence probability for new targets based on their features.
        prob_existence_new_deep = nebp.new_mlp(torch.cat([app_feat_new, measurements[:, :, [7]]], dim = -1)).squeeze(-1)
        
        threshold = parameters['prob_new_deep_threshold']
        scale = parameters['prob_new_deep_scale']
        if mode == 'val': # In validation mode, can apply hard thresholding or scaling.
            if scale is None:
                prob_existence_new_deep[prob_existence_new_deep < threshold] = 0
                prob_existence_new_deep[prob_existence_new_deep >= threshold] = 1
            else:
                shift = np.log(threshold / (1 - threshold))
                prob_existence_new_deep_logit = torch.log(prob_existence_new_deep / (1 - prob_existence_new_deep))
                prob_existence_new_deep = torch.sigmoid(scale * (prob_existence_new_deep_logit - shift))
        
        # Concatenate motion and appearance features to create hidden states for the GNN.
        h_a = torch.cat([motion_feat_existing, app_feat_existing], dim = -1)
        h_b = torch.cat([motion_feat_new, app_feat_new], dim = -1)

    # ==================================
    # 4. DATA ASSOCIATION (BP)
    # ==================================
    message_out, message_new_out, asso_prob, asso_prob_new, \
    message_in_nebp, message_in_deep, message_in_deep_scale, \
    h_a, h_b = \
        loopy_bp_da(message_in = message_in,
                    message_new_in = message_new_in,
                    prob_existence_new_deep = prob_existence_new_deep,
                    targets_mask = targets_mask,
                    measurements_mask = measurements_mask,
                    dist_mask = dist_mask,
                    threshold = parameters['threshold_bp'],
                    num_bp_iter = parameters['num_bp_iter'],
                    num_gnn_iter = parameters['num_gnn_iter'],
                    nebp = nebp,
                    h_a = h_a,
                    h_b = h_b)

    asso_prob_single_nebp = message_in_nebp / torch.sum(message_in_nebp, dim = -1, keepdim = True)
    asso_prob_single_deep = message_in_deep


    # ==================================
    # 5. UPDATE STEP
    # ==================================
    # -----------------------------
    # update likelihood (Re-weighting based on GNN output)
    # -----------------------------
    # This subtle step incorporates the GNN's global insight into the particle filter update.
    # It back-calculates a scaling factor from the GNN-refined messages to update the original likelihoods.
    message_in_tmp = torch.zeros(batch_size, num_max_targets, num_max_meas + 1, device = log_likelihood.device)
    message_in_tmp[:, :, 0] = 1 - parameters['prob_detection']
    message_in_tmp[:, :, 1 :] = torch.mean(likelihood, dim = -1) * constant_factor

    message_in_nebp_tmp = message_in_nebp * \
                        (((1 - parameters['prob_detection']) * prob_existence_predict).unsqueeze(-1) + (1 - prob_existence_predict.unsqueeze(-1)) ) / \
                        (message_in_nebp[:, :, [0]] + 1e-6)
    message_in_nebp_tmp[:, :, 0] = message_in_nebp_tmp[:, :, 0] - (1 - prob_existence_predict)
    message_in_nebp_tmp = message_in_nebp_tmp / (prob_existence_predict.unsqueeze(-1) + 1e-6)

    likelihood = likelihood * (message_in_nebp_tmp / (message_in_tmp + 1e-6))[:, :, 1:, None]

    # -----------------------------
    # update existing targets
    # -----------------------------
    # Calculate posterior particle weights using the final messages and re-weighted likelihoods.
    weights = torch.sum(message_out.unsqueeze(-1) * likelihood, dim = -2) * constant_factor + (1 - parameters['prob_detection'].unsqueeze(-1)) # [batch_size, num_max_targets, num_particles]
    # Resample particles based on these weights (core of the particle filter update).
    resample_indices = torch.multinomial(weights[targets_mask, :], parameters['num_particles'], replacement = True)
    particles_kinematic_posterior = torch.zeros_like(particles_kinematic_predict)
    particles_kinematic_posterior[targets_mask, :, :] = torch.gather(particles_kinematic_predict[targets_mask, :, :], dim = 1,
                                                                     index = resample_indices.unsqueeze(-1).expand(-1, -1, dim_state))
    # New state estimate is the weighted average of the particles.
    weights_normalized = weights / (torch.sum(weights, dim = -1, keepdim = True))
    estimations_posterior = torch.sum(weights_normalized[..., None] * particles_kinematic_predict, dim = -2)

    # Update the probability of existence for each target based on whether it was associated.
    alive_update = torch.mean(weights[targets_mask, :], dim = -1)
    alive_tmp = prob_existence_predict[targets_mask] * alive_update
    dead_tmp = 1 - prob_existence_predict[targets_mask]
    prob_existence_posterior = torch.zeros_like(prob_existence_predict)
    prob_existence_posterior[targets_mask] = alive_tmp / (alive_tmp + dead_tmp)


    # Use a linear assignment solver to find the single best association for scoring purposes.
    tracking_score = torch.zeros_like(targets_mask).float()
    prob_detection_update = prob_detection.clone()
    tracking_score_new = measurements[:, :, 7].clone() # measurement score
    prob_detection_new = measurements[:, :, 7].clone()
    for batch in range(batch_size):
        dist_matrix = torch.sum((estimations_posterior[batch, :, None, 0 : 2] - measurements[batch, None, :, 0 : 2]) ** 2, dim = -1) ** 0.5
        asso_prob_batch = torch.log(asso_prob[batch, :, 1:].clone() + 1e-6)
        asso_prob_batch[dist_matrix > 2] = float('nan')
        asso_prob_batch[prob_existence_posterior[batch, :] < 0.5, :] = float('nan')
        # Find the optimal assignment that maximizes the total log-probability.
        row_inds, col_inds = solve_dense(-asso_prob_batch.detach().cpu().numpy())
        # Update scores based on the assignment
        tracking_score[batch, row_inds] = measurements[batch, col_inds, 7]
        tracking_score_new[batch, col_inds] = 0
        prob_detection_update[batch, row_inds] = measurements[batch, col_inds, 7]
    tracking_score[~targets_mask] = 0
    prob_detection_update[~targets_mask] = 0

    # -----------------------------
    # update new targets
    # -----------------------------
    # Update the false alarm distribution with the GNN-refined existence probability.
    constants = constants * prob_existence_new_deep 
    # The posterior existence probability of a new target.
    prob_existence_new = message_new_out * constants / (message_new_out * constants + 1)
    # The state estimate for a new target is just the measurement itself.
    estimations_new = measurements


    return particles_kinematic_posterior, estimations_posterior, prob_existence_posterior, tracking_score, prob_detection_update, targets_id, \
        particles_kinematic_new, estimations_new, prob_existence_new, tracking_score_new, prob_detection_new, targets_id_new, \
        asso_prob, asso_prob_new, asso_prob_single_nebp, asso_prob_single_deep, message_in_deep_scale, prob_existence_new_deep
