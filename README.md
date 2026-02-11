# Monte Carlo pi Estimation: CPU vs GPU

## Overview
A comparison of CPU (serial C++) and GPU (CUDA) implementations of Monte Carlo 
simulation to estimate value of pi. Estimates value of pi by randomly sampling points 
in a unit square and counting how many fall inside the circle inscribed. 
This ratio will approx π/4.

Running on Google Colab's Telsa T4 (no gpu :/ ) with CUDA 12.8 using nvcc4jupyter.

## How it works

The Monte Carlo method works by:
1. Generate random (x, y) points in range [-1, 1]
2. Check if point falls inside unit circle: x² + y² ≤ 1
3. Estimate π = 4 × (points_in_circle / total_points)

**CPU Version:** Sequential generation using C++ MT19937
- Result: ~958ms for 10M simulations

**GPU Version:** Parallel CUDA kernel
- Uses 256 threads/block with 8x loop unrolling
- Shared memory reduction, atomic add to global counters
- All random numbers generated upfront with cuRAND MT19937 (mersenne twister)

## Performance

Sample size: 10,000,000 points
- CPU (serial): ~958ms
- GPU (CUDA): ~204ms (Tesla T4, hardware dependent!!) 

Expected speedup on modern GPUs: 50-200x for 10M+ samples

The actual speedup depends on GPU model, memory bandwidth, L2 cache size, 
and CUDA version.

## Known Biases & Limitations

**Biases:**
- Small sample size (10M) may not fully converge to π
- No GPU warm-up run first kernel includes driver initialization overhead (yikes!)
- Different RNGs between CPU/GPU (MT19937 vs cuRAND MT19937)
- Fixed seed with deterministic results, so doesn't explore full random space
- Single trial with no averaging (too long)

**Limitations:**
- No statistical uncertainty quantification
- GPU stores all N random numbers simultaneously (memory overhead)
- Atomic operations create serialization bottleneck
- CPU version not parallelized (could use OpenMP for fair comparison)
- No convergence analysis or error bars

## Pros & Cons

**Pros:**
- Embarrassingly parallel algorithm ideal for GPU
- Clean code separation (generation vs computation)
- Optimized memory patterns (coalescing, shared memory)
- Scalable to billions of samples
- Cache-flush aware benchmarking included
- Deterministic with fixed seed for reproducibility

**Cons:**
- Atomic contention reduces parallelism efficiency
- Memory-bound for large N (stores all random numbers)
- No multi-trial statistical analysis
- CPU baseline not optimized (no SIMD, no threading)
- Cold start effects not properly handled

## Future Improvements

- Add multiple trials with mean/variance analysis
- Implement OpenMP CPU version for fair comparison
- Add proper warm-up iterations for both CPU/GPU
- Plot convergence curve (estimate vs sample count)
- Explore different block sizes and unroll factors
- Replace atomics with warp-level primitives
- Add CPU SIMD version (AVX2/AVX-512)
- Benchmark with different RNG algorithms
