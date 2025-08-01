import pickle
import numpy as np
import os

def compare_results(num_trials=10):
    """
    Compares the state estimates from the NumPy and PyTorch implementations.
    """
    pytorch_results_folder = "/Users/lipingb/Desktop/distributed_experiment_pytorch"
    numpy_results_folder = "/Users/lipingb/Desktop/distributed_experiment_numpy"

    print("--- Comparing NumPy and PyTorch State Estimates ---")

    for mc in range(num_trials):
        pytorch_file = os.path.join(pytorch_results_folder, f"trial_{mc:04d}_pytorch.pkl")
        numpy_file = os.path.join(numpy_results_folder, f"trial_{mc:04d}_numpy.pkl")

        if not os.path.exists(pytorch_file) or not os.path.exists(numpy_file):
            print(f"Skipping trial {mc+1}: result files not found.")
            continue

        with open(pytorch_file, 'rb') as f:
            pytorch_results = pickle.load(f)
        
        with open(numpy_file, 'rb') as f:
            numpy_results = pickle.load(f)

        n_steps = pytorch_results['parameters']['n_steps']
        
        total_diff_s1 = 0
        total_diff_s2 = 0
        
        print(f"\n--- Trial {mc+1} ---")

        for step in range(n_steps):
            # Sensor 1
            pytorch_est_s1 = pytorch_results['estimates_sensor1'][step]
            numpy_est_s1 = numpy_results['estimates_sensor1'][step]
            
            if 'state' in pytorch_est_s1 and 'state' in numpy_est_s1:
                # Align labels to ensure we are comparing the same tracks
                py_labels_s1 = pytorch_est_s1['label']
                np_labels_s1 = numpy_est_s1['label']
                
                common_labels = np.intersect1d(py_labels_s1, np_labels_s1)
                
                for label in common_labels:
                    py_idx = np.where(py_labels_s1 == label)[0][0]
                    np_idx = np.where(np_labels_s1 == label)[0][0]
                    
                    # check if the state is zero or not
                    if np.all(pytorch_est_s1['state'][:, py_idx] == 0) or np.all(numpy_est_s1['state'][:, np_idx] == 0):
                        # print state invalid
                        print(f"  Step {step}, Sensor 1, Label {label}: State is zero, skipping comparison.")
                        continue
                    py_state = pytorch_est_s1['state'][:, py_idx]
                    np_state = numpy_est_s1['state'][:, np_idx]
                    
                    diff = np.linalg.norm(py_state - np_state)
                    total_diff_s1 += diff
                    if diff > 1e-3:
                        print(f"  Step {step}, Sensor 1, Label {label}: Difference norm = {diff:.4f}")

            # Sensor 2
            pytorch_est_s2 = pytorch_results['estimates_sensor2'][step]
            numpy_est_s2 = numpy_results['estimates_sensor2'][step]

            if 'state' in pytorch_est_s2 and 'state' in numpy_est_s2:
                py_labels_s2 = pytorch_est_s2['label']
                np_labels_s2 = numpy_est_s2['label']
                
                common_labels = np.intersect1d(py_labels_s2, np_labels_s2)

                for label in common_labels:
                    py_idx = np.where(py_labels_s2 == label)[0][0]
                    np_idx = np.where(np_labels_s2 == label)[0][0]
                    # check if the state is zero or not
                    if np.all(pytorch_est_s2['state'][:, py_idx] == 0) or np.all(numpy_est_s2['state'][:, np_idx] == 0):
                        # print state invalid
                        print(f"  Step {step}, Sensor 2, Label {label}: State is zero, skipping comparison.")
                        continue
                    py_state = pytorch_est_s2['state'][:, py_idx]
                    np_state = numpy_est_s2['state'][:, np_idx]
                    
                    diff = np.linalg.norm(py_state - np_state)
                    total_diff_s2 += diff
                    if diff > 1e-3:
                        print(f"  Step {step}, Sensor 2, Label {label}: Difference norm = {diff:.4f}")

        print(f"  Total difference for Sensor 1: {total_diff_s1:.4f}")
        print(f"  Total difference for Sensor 2: {total_diff_s2:.4f}")


if __name__ == '__main__':
    compare_results()
