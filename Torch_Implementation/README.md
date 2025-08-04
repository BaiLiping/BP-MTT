# PyTorch Implementation of BP-based Multi-Target Tracking

This folder contains the PyTorch implementation of the Belief Propagation (BP) based multi-target tracking algorithm, converted from the original MATLAB implementation in the Meyer folder.

## Files

### Core Classes
- **`data_generator.py`** - `Data_Generator` class for simulation data generation
- **`bp_filter.py`** - `BP_Filter` class implementing the tracking algorithm
- **`main_pytorch.py`** - Main test script demonstrating usage

## Usage

```python
python main_pytorch.py
```

## Key Features

- **GPU Acceleration**: Automatic CUDA detection and tensor operations
- **Vectorized Operations**: Efficient PyTorch tensor computations
- **Memory Efficient**: Optimized particle filtering implementation
- **Modular Design**: Separate classes for data generation and filtering
- **Performance Monitoring**: Built-in timing and metrics

## Classes Overview

### Data_Generator
Handles simulation data generation:
- Target trajectory generation with constant velocity model
- Range/bearing measurement simulation
- Clutter generation and detection modeling
- Complete simulation pipeline

### BP_Filter
Implements the BP-based tracking algorithm:
- Particle prediction with motion model
- Measurement likelihood evaluation
- New target initialization
- Belief propagation data association
- Particle weight updates and resampling
- Track management and pruning

## Algorithm Parameters

The implementation uses the same parameters as the original MATLAB version:
- 200 time steps
- 5 targets with staggered appearance times
- 2 sensors with range/bearing measurements
- 10,000 particles per target
- Constant velocity motion model
- Poisson clutter model

## Performance

The PyTorch implementation provides significant speedup over MATLAB through:
- GPU tensor operations
- Vectorized computations
- Optimized memory access patterns
- Efficient resampling algorithms

## Requirements

- PyTorch
- NumPy
- Matplotlib (for visualization)
- CUDA (optional, for GPU acceleration)