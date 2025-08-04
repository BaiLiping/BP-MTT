# NumPy Implementation of BP-based Multi-Target Tracking

This folder contains the NumPy implementation of the Belief Propagation (BP) based multi-target tracking algorithm, converted from the PyTorch implementation for performance comparison.

## Files

### Core Classes
- **`data_generator.py`** - `Data_Generator` class for simulation data generation
- **`bp_filter.py`** - `BP_Filter` class implementing the tracking algorithm
- **`main_numpy.py`** - Main test script demonstrating usage

## Usage

```python
python main_numpy.py
```

## Key Features

- **Pure NumPy**: Uses only NumPy arrays for all computations
- **CPU Optimized**: Efficient vectorized operations for CPU execution
- **Memory Efficient**: Optimized array operations and memory management
- **Identical Algorithm**: Same BP-based tracking algorithm as PyTorch version
- **Reproducible**: Fixed random seeds for consistent results

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

## Performance Characteristics

The NumPy implementation is designed for:
- CPU-only execution
- Stable numerical computations
- Clear, readable code structure
- Baseline performance comparison

## Algorithm Parameters

Uses the same parameters as the PyTorch version:
- 200 time steps (reduced to 100 for comparison)
- 5 targets with staggered appearance times
- 2 sensors with range/bearing measurements
- 5,000 particles per target (reduced for comparison)
- Constant velocity motion model
- Poisson clutter model

## Requirements

- NumPy
- Matplotlib (for visualization)
- Python 3.7+

## Comparison with PyTorch

Use the `performance_comparison.py` script in the parent directory to compare:
- Execution speed
- Memory usage
- Numerical accuracy
- Implementation differences