import numpy as np
import pickle
import copy
import torch
import time
import logging
from TrackerBP import TrackerBP as TrackerBP_NumPy
from TrackerBP_Torch import TrackerBP_PyTorch
from Utils import set_parameters, track_formation
from copy import deepcopy
import os

def process_trial_pytorch(mc, parameters, data, device):
    """
    Process one Monte Carlo trial using the PyTorch tracker.
    """
    results_folder = "/Users/lipingb/Desktop/distributed_experiment_pytorch"
    os.makedirs(results_folder, exist_ok=True)
    
    n_steps = parameters['n_steps']
    trajectories = data['all_true_tracks'][mc]
    measurements = data['all_measurements'][mc]
    
    estimates_sensor1_torch = [None] * n_steps
    estimates_sensor2_torch = [None] * n_steps
    parameters_sensor1_torch = copy.deepcopy(parameters)
    parameters_sensor2_torch = copy.deepcopy(parameters)
    tracker1_torch = TrackerBP_PyTorch(parameters_sensor1_torch, device=device)
    tracker2_torch = TrackerBP_PyTorch(parameters_sensor2_torch, device=device)
    sensor_pos_1_torch = torch.tensor(parameters['sensor_positions'][:, 0], dtype=torch.float32, device=device)
    sensor_pos_2_torch = torch.tensor(parameters['sensor_positions'][:, 1], dtype=torch.float32, device=device)

    torch_time = 0
    for step in range(n_steps):
        torch_start_time = time.time()
        
        current_measurements_s1 = measurements[step][0]
        measurements_tensor_s1 = torch.tensor(current_measurements_s1, dtype=torch.float32, device=device)
        tracker1_torch.compute_alpha()
        tracker1_torch.compute_xi_sigma(measurements_tensor_s1, sensor_pos_1_torch)
        tracker1_torch.compute_beta(measurements_tensor_s1, sensor_pos_1_torch)
        tracker1_torch.compute_kappa_iota()
        tracker1_torch.compute_gamma()
        tracker1_torch.prune()
        estimates_sensor1_torch[step] = deepcopy(tracker1_torch.estimate_state())
        if estimates_sensor1_torch[step] is not None and 'position' in estimates_sensor1_torch[step]:
            print(f"Trial {mc+1}, Step {step+1} (PyTorch, Sensor 1) Estimated Position: {estimates_sensor1_torch[step]['position']}")
        else:
            print(f"Trial {mc+1}, Step {step+1} (PyTorch, Sensor 1) No position estimate available.")
        current_measurements_s2 = measurements[step][1]
        measurements_tensor_s2 = torch.tensor(current_measurements_s2, dtype=torch.float32, device=device)
        tracker2_torch.compute_alpha()
        tracker2_torch.compute_xi_sigma(measurements_tensor_s2, sensor_pos_2_torch)
        tracker2_torch.compute_beta(measurements_tensor_s2, sensor_pos_2_torch)
        tracker2_torch.compute_kappa_iota()
        tracker2_torch.compute_gamma()
        tracker2_torch.prune()
        estimates_sensor2_torch[step] = deepcopy(tracker2_torch.estimate_state())
        if estimates_sensor2_torch[step] is not None and 'position' in estimates_sensor2_torch[step]:
            print(f"Trial {mc+1}, Step {step+1} (PyTorch, Sensor 2) Estimated Position: {estimates_sensor2_torch[step]['position']}")
        else:
            print(f"Trial {mc+1}, Step {step+1} (PyTorch, Sensor 2) No position estimate available.")
        torch_time += time.time() - torch_start_time

    print(f"Trial {mc+1} (PyTorch): took {torch_time:.4f} seconds.")
    
    trial_results = {
        'true_tracks': trajectories,
        'estimates_sensor1': estimates_sensor1_torch,
        'estimates_sensor2': estimates_sensor2_torch,
        'parameters': parameters,
        'torch_time': torch_time,
    }
    
    trial_filename = os.path.join(results_folder, f"trial_{mc:04d}_pytorch.pkl")
    with open(trial_filename, 'wb') as f:
        pickle.dump(trial_results, f)
    logging.info(f"PyTorch trial {mc+1} results saved to {trial_filename}")
    return torch_time

def run_numpy_mc_sequential(parameters, data, num_trials):
    """
    Loads data and processes Monte Carlo trials sequentially using the NumPy tracker.
    """
    results_folder = "/Users/lipingb/Desktop/distributed_experiment_numpy"
    os.makedirs(results_folder, exist_ok=True)
    n_steps = parameters['n_steps']
    total_time = 0
    times = []

    for mc in range(num_trials):
        trajectories = data['all_true_tracks'][mc]
        measurements = data['all_measurements'][mc]

        estimates_sensor1_numpy = [None] * n_steps
        estimates_sensor2_numpy = [None] * n_steps
        parameters_sensor1_numpy = copy.deepcopy(parameters)
        parameters_sensor2_numpy = copy.deepcopy(parameters)
        tracker1_numpy = TrackerBP_NumPy(parameters_sensor1_numpy)
        tracker2_numpy = TrackerBP_NumPy(parameters_sensor2_numpy)

        numpy_time = 0
        for step in range(n_steps):
            numpy_start_time = time.time()
            
            current_measurements_s1_numpy = measurements[step][0]
            tracker1_numpy.compute_alpha()
            tracker1_numpy.compute_xi_sigma(current_measurements_s1_numpy, 0)
            tracker1_numpy.compute_beta(current_measurements_s1_numpy, 0)
            tracker1_numpy.compute_kappa_iota()
            tracker1_numpy.compute_gamma()
            tracker1_numpy.prune()
            estimates_sensor1_numpy[step] = deepcopy(tracker1_numpy.estimate_state())
            if estimates_sensor1_numpy[step] is not None and 'position' in estimates_sensor1_numpy[step]:
                print(f"Trial {mc+1}, Step {step+1} (NumPy, Sensor 1) Estimated Position: {estimates_sensor1_numpy[step]['position']}")     
            else:
                print(f"Trial {mc+1}, Step {step+1} (NumPy, Sensor 1) No position estimate available.")            
            current_measurements_s2_numpy = measurements[step][1]
            tracker2_numpy.compute_alpha()
            tracker2_numpy.compute_xi_sigma(current_measurements_s2_numpy, 1)
            tracker2_numpy.compute_beta(current_measurements_s2_numpy, 1)
            tracker2_numpy.compute_kappa_iota()
            tracker2_numpy.compute_gamma()
            tracker2_numpy.prune()
            estimates_sensor2_numpy[step] = deepcopy(tracker2_numpy.estimate_state())
            if estimates_sensor2_numpy[step] is not None and 'position' in estimates_sensor2_numpy[step]:
               print(f"Trial {mc+1}, Step {step+1} (NumPy, Sensor 2) Estimated Position: {estimates_sensor2_numpy[step]['position']}")
            else:
               print(f"Trial {mc+1}, Step {step+1} (NumPy, Sensor 2) No position estimate available.")

            numpy_time += time.time() - numpy_start_time

        logging.info(f"Trial {mc+1} (NumPy): took {numpy_time:.4f} seconds.")
        times.append(numpy_time)
        
        trial_results = {
            'true_tracks': trajectories,
            'estimates_sensor1': estimates_sensor1_numpy,
            'estimates_sensor2': estimates_sensor2_numpy,
            'parameters': parameters,
            'numpy_time': numpy_time
        }
        
        trial_filename = os.path.join(results_folder, f"trial_{mc:04d}_numpy.pkl")
        with open(trial_filename, 'wb') as f:
            pickle.dump(trial_results, f)
        logging.info(f"NumPy trial {mc+1} results saved to {trial_filename}")
    return times

def run_pytorch_mc_sequential(parameters, data, num_trials):
    """
    Loads data and processes Monte Carlo trials sequentially using the PyTorch tracker.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA device found. Using GPU.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logging.info("MPS device found. Using Apple Silicon GPU.")
    else:
        device = 'cpu'
        logging.info("No GPU found. Using CPU.")

    times = []
    for mc in range(num_trials):
        trial_time = process_trial_pytorch(mc, parameters, data, device)
        times.append(trial_time)
        
    logging.info("PyTorch-based sequential tracking experiment completed.")
    return times

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler("log.txt", mode='w'), logging.StreamHandler()])

    parameters = set_parameters()
    with open('/Users/lipingb/Desktop/data_generation.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_trials = 10
    
    # --- Run PyTorch Benchmark ---
    logging.info("--- Running PyTorch (Sequential) Benchmark ---")
    pytorch_times = run_pytorch_mc_sequential(parameters, data, num_trials)

    # --- Run NumPy Benchmark ---
    logging.info("\n--- Running NumPy (Sequential) Benchmark ---")
    numpy_times = run_numpy_mc_sequential(parameters, data, num_trials)

    # --- Summary ---
    if pytorch_times and numpy_times:
        avg_pytorch_time = np.mean(pytorch_times)
        avg_numpy_time = np.mean(numpy_times)
        
        logging.info("\n--- Benchmark Summary ---")
        logging.info(f"Average PyTorch time over {num_trials} trials: {avg_pytorch_time:.4f} seconds.")
        logging.info(f"Average NumPy time over {num_trials} trials: {avg_numpy_time:.4f} seconds.")