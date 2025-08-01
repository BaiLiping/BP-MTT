Converted Meyer's paper and reference in python for better integratation into other projects.


The Torch version is much better:
2025-08-01 12:06:40,805 - 
--- Running NumPy (Sequential) Benchmark ---
2025-08-01 12:06:58,972 - Trial 1 (NumPy): took 18.1658 seconds.
2025-08-01 12:06:59,968 - NumPy trial 1 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0000_numpy.pkl
2025-08-01 12:07:17,565 - Trial 2 (NumPy): took 17.5933 seconds.
2025-08-01 12:07:18,427 - NumPy trial 2 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0001_numpy.pkl
2025-08-01 12:07:36,744 - Trial 3 (NumPy): took 18.3140 seconds.
2025-08-01 12:07:37,615 - NumPy trial 3 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0002_numpy.pkl
2025-08-01 12:07:56,085 - Trial 4 (NumPy): took 18.4672 seconds.
2025-08-01 12:07:56,967 - NumPy trial 4 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0003_numpy.pkl
2025-08-01 12:08:13,983 - Trial 5 (NumPy): took 17.0122 seconds.
2025-08-01 12:08:14,931 - NumPy trial 5 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0004_numpy.pkl
2025-08-01 12:08:31,695 - Trial 6 (NumPy): took 16.7611 seconds.
2025-08-01 12:08:32,492 - NumPy trial 6 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0005_numpy.pkl
2025-08-01 12:08:48,725 - Trial 7 (NumPy): took 16.2296 seconds.
2025-08-01 12:08:49,460 - NumPy trial 7 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0006_numpy.pkl
2025-08-01 12:09:05,445 - Trial 8 (NumPy): took 15.9827 seconds.
2025-08-01 12:09:06,150 - NumPy trial 8 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0007_numpy.pkl
2025-08-01 12:09:23,109 - Trial 9 (NumPy): took 16.9581 seconds.
2025-08-01 12:09:23,899 - NumPy trial 9 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0008_numpy.pkl
2025-08-01 12:09:41,351 - Trial 10 (NumPy): took 17.4491 seconds.
2025-08-01 12:09:42,157 - NumPy trial 10 results saved to /Users/lipingb/Desktop/distributed_experiment_numpy/trial_0009_numpy.pkl
2025-08-01 12:09:42,243 - --- Running PyTorch (Sequential) Benchmark ---
2025-08-01 12:09:42,356 - MPS device found. Using Apple Silicon GPU.
2025-08-01 12:10:00,700 - Trial 1 (PyTorch): took 18.2275 seconds.
2025-08-01 12:10:00,705 - PyTorch trial 1 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0000_pytorch.pkl
2025-08-01 12:10:12,115 - Trial 2 (PyTorch): took 11.4065 seconds.
2025-08-01 12:10:12,122 - PyTorch trial 2 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0001_pytorch.pkl
2025-08-01 12:10:21,031 - Trial 3 (PyTorch): took 8.9054 seconds.
2025-08-01 12:10:21,040 - PyTorch trial 3 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0002_pytorch.pkl
2025-08-01 12:10:29,638 - Trial 4 (PyTorch): took 8.5950 seconds.
2025-08-01 12:10:29,648 - PyTorch trial 4 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0003_pytorch.pkl
2025-08-01 12:10:38,251 - Trial 5 (PyTorch): took 8.5999 seconds.
2025-08-01 12:10:38,253 - PyTorch trial 5 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0004_pytorch.pkl
2025-08-01 12:10:45,814 - Trial 6 (PyTorch): took 7.5582 seconds.
2025-08-01 12:10:45,818 - PyTorch trial 6 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0005_pytorch.pkl
2025-08-01 12:10:53,599 - Trial 7 (PyTorch): took 7.7780 seconds.
2025-08-01 12:10:53,605 - PyTorch trial 7 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0006_pytorch.pkl
2025-08-01 12:11:02,500 - Trial 8 (PyTorch): took 8.8925 seconds.
2025-08-01 12:11:02,505 - PyTorch trial 8 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0007_pytorch.pkl
2025-08-01 12:11:12,027 - Trial 9 (PyTorch): took 9.5188 seconds.
2025-08-01 12:11:12,031 - PyTorch trial 9 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0008_pytorch.pkl
2025-08-01 12:11:19,594 - Trial 10 (PyTorch): took 7.5577 seconds.
2025-08-01 12:11:19,598 - PyTorch trial 10 results saved to /Users/lipingb/Desktop/distributed_experiment_pytorch/trial_0009_pytorch.pkl
2025-08-01 12:11:19,599 - PyTorch-based sequential tracking experiment completed.
2025-08-01 12:11:19,599 - 
--- Benchmark Summary ---
2025-08-01 12:11:19,599 - Average PyTorch time over 10 trials: 9.7039 seconds.
2025-08-01 12:11:19,599 - Average NumPy time over 10 trials: 17.2933 seconds.
