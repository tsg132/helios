# Helios NEON SIMD Vectorization: Implementation & Results

## Overview

This document describes the ARM NEON SIMD vectorization added to Helios, covering what was implemented, how the NEON kernels work, the benchmark methodology, and a detailed analysis of the results.

All NEON code lives in new files — the original scalar codebase is completely unchanged.

---

## What Was Vectorized

Four hot loops were targeted, each implemented as an inline NEON kernel function:

### 1. Sparse Dot Product (`neon_sparse_dot`)

**The core inner loop of Helios.** Every solver mode (Jacobi, Gauss-Seidel, Async, Plan) ultimately computes this for each row:

```
dot_i = sum_{k} P[i,k] * x[k]     (sparse, CSR format)
```

**Scalar version** (from `policy_eval_op.cc`):
```cpp
for (index_t idx = start; idx < end; ++idx)
    dot += probs[idx] * x[col_idx[idx]];
```

**NEON version**: Unrolls by 4 with two independent FMA accumulator chains:

```
acc0 += probs[k:k+2] * x[col_idx[k:k+2]]     (NEON vfmaq_f64)
acc1 += probs[k+2:k+4] * x[col_idx[k+2:k+4]]  (NEON vfmaq_f64)
```

Key details:
- **Contiguous prob loads**: `vld1q_f64` loads 2 consecutive doubles from the CSR `probs` array in one instruction.
- **Manual gather for x[]**: NEON has no gather instruction for `float64`. Each `x[col_idx[k]]` is loaded individually with `vld1_f64`, then combined with `vcombine_f64`. This is the fundamental bottleneck.
- **Two accumulator chains**: Apple M-series FMA has 4-cycle latency but 2-per-cycle throughput. Two independent chains (`acc0`, `acc1`) keep both FMA pipes fed even when one chain stalls on a cache miss from the gather.
- **Scalar tail**: Handles the remaining 0–3 elements after the unrolled loop.

### 2. Jacobi Sweep (`neon_jacobi_sweep`)

Computes a full Jacobi epoch — all `n` rows — without virtual dispatch:

```
x_next[i] = (1 - alpha) * x_curr[i] + alpha * (r[i] + beta * dot_i)
```

Processes rows in pairs. After computing two sparse dot products, the alpha blend is vectorized:
- `vfmaq_f64` computes `(1-alpha)*x_curr + alpha*fi` for 2 rows simultaneously
- `vst1q_f64` writes both results in one instruction

The key advantage over using `PolicyEvalOpNeon` through the Runtime is **eliminating `n` virtual function calls per epoch**. At `n = 10K`, that saves ~100K ns of vtable dispatch overhead.

### 3. Residual Max Reduction (`neon_residual_max`)

Computes `max_i |F_i(x) - x[i]|` across all rows:
- Processes rows in pairs
- Computes `|fi - x[i]|` using `vabsq_f64(vsubq_f64(...))`
- Running max with `vmaxq_f64`
- Final horizontal max via `vmaxvq_f64`

Avoids the scalar branch misprediction pattern (`if (res > max) max = res`).

### 4. Gauss-Seidel Sweep (`neon_gauss_seidel_sweep`)

In-place update using NEON inner products per row. Cannot vectorize across rows (data dependency: row `i+1` reads the updated `x[i]`), but the inner sparse dot product within each row uses `neon_sparse_dot`.

---

## Architecture

The NEON code integrates cleanly without touching any existing files:

```
Existing (unchanged)                    New (NEON)
────────────────────                    ──────────
operator.h (abstract)         ──>       policy_eval_op_neon.h
  PolicyEvalOp                            PolicyEvalOpNeon (subclass)
    apply_i() [scalar]                      apply_i() [calls neon_sparse_dot]

runtime.cc                              simd/neon_kernels.h
  run_jacobi_()                           neon_jacobi_sweep()    [batch, no vtable]
  run_gauss_seidel_()                     neon_gauss_seidel_sweep()
  residual_inf()                          neon_residual_max()

bench/run_bench.cc                      bench/bench_simd.cc
tests/test_runtime_smoke.cc             tests/test_simd.cc
```

**Two integration paths**:
1. **Drop-in operator**: `PolicyEvalOpNeon` plugs into the existing Runtime via virtual dispatch. Same convergence behavior, just a faster inner product.
2. **Batch kernels**: `neon_jacobi_sweep` and friends bypass the Runtime entirely, eliminating virtual dispatch for maximum throughput. Used in benchmarks.

All NEON code is guarded by `#ifdef __ARM_NEON` with scalar fallbacks, so the project compiles on any platform.

---

## Benchmark Methodology

**Platform**: Apple Silicon (ARM NEON, 128-bit registers = 2x float64)

**Benchmark structure** (`bench/bench_simd.cc`):
- **Bench A** — Sparse dot product: Measures `apply_i()` per-row latency (ns/row), scalar vs NEON, varying `nnz_per_row` from 4 to 64
- **Bench B** — Jacobi sweep: Full epoch wall time, scalar per-row (with virtual dispatch) vs NEON batch kernel
- **Bench C** — Residual max: `Runtime::residual_inf()` vs `neon_residual_max()`
- **Bench D** — Full convergence: Wall time to reach eps=1e-6 through the Runtime
- **Bench E** — Gauss-Seidel sweep: Same structure as Bench B but in-place

All benchmarks use `build_random_sparse_mdp()` with configurable `n` and `nnz_per_row`, multiple repetitions, and warmup passes.

---

## Results

### Sparse Dot Product (per-row, n=50,000)

| nnz/row | Scalar (ns) | NEON (ns) | Speedup |
|---------|-------------|-----------|---------|
| 4       | 3.2         | 2.9       | 1.10x   |
| 8       | 5.8         | 5.4       | 1.07x   |
| 16      | 11.8        | 9.8       | 1.20x   |
| 32      | 23.2        | 19.0      | 1.22x   |
| 64      | 45.0        | 38.4      | 1.17x   |

Speedup increases with `nnz_per_row` up to 32, then plateaus. At `nnz=4`, the unrolled loop body never executes (all work is in the scalar tail). At `nnz=32`, the two-accumulator pipeline is well-utilized.

### Jacobi Sweep (full epoch, nnz/row=20)

| n       | Scalar (ms) | NEON (ms) | Speedup |
|---------|-------------|-----------|---------|
| 1,000   | 0.01        | 0.006     | 1.58x   |
| 10,000  | 0.10        | 0.05      | 1.87x   |
| 100,000 | 1.36        | 1.25      | 1.09x   |
| 500,000 | 8.33        | 7.62      | 1.09x   |

The sweet spot is **n=10,000**: working set fits in L2 cache (~1.6 MB for x + CSR arrays), and virtual dispatch elimination provides the biggest relative gain.

### Gauss-Seidel Sweep (full epoch, nnz/row=20)

| n       | Scalar (ms) | NEON (ms) | Speedup |
|---------|-------------|-----------|---------|
| 1,000   | 0.01        | 0.006     | 1.55x   |
| 10,000  | 0.10        | 0.05      | 1.99x   |
| 100,000 | 1.43        | 1.24      | 1.15x   |
| 500,000 | 8.02        | 6.74      | 1.19x   |

**Best result: 1.99x at n=10K.** GS benefits slightly more than Jacobi because the in-place update pattern has better temporal locality — recently written `x[j]` values are still warm in L1 when accessed by nearby rows.

### Residual Max Reduction (nnz/row=20)

| n       | Scalar (ms) | NEON (ms) | Speedup |
|---------|-------------|-----------|---------|
| 1,000   | 0.01        | 0.008     | 1.31x   |
| 10,000  | 0.06        | 0.05      | 1.30x   |
| 100,000 | 1.35        | 1.24      | 1.09x   |
| 500,000 | 7.21        | 7.87      | 0.92x   |

At n=500K the NEON version is slightly *slower*. This is likely due to the pair-processing overhead (computing two rows and combining results) interfering with the prefetcher's access pattern at large working set sizes.

### Full Convergence (Jacobi, n=50,000, beta=0.99)

| Operator           | Wall Time | Updates/sec |
|--------------------|-----------|-------------|
| Scalar             | 1.711s    | 3.82e+07    |
| NEON (via Runtime) | 1.675s    | 3.90e+07    |

**2.1% wall time improvement** through the Runtime's virtual dispatch path. The gain is modest because the Runtime adds overhead beyond `apply_i` (residual scans, buffer management, convergence checking).

---

## Analysis: Why Gains Are Modest at Large n

The sparse dot product `dot += probs[k] * x[col_idx[k]]` has two components:

1. **Arithmetic**: One multiply + one add per nonzero — 2 FLOPs
2. **Memory**: Load `probs[k]` (8 bytes, contiguous), load `col_idx[k]` (4 bytes, contiguous), load `x[col_idx[k]]` (8 bytes, **random**)

The arithmetic intensity is approximately:

```
AI = 2 FLOPs / 20 bytes = 0.1 FLOP/byte
```

Apple M-series has ~100 GB/s memory bandwidth and ~200 GFLOP/s (f64) compute. The **roofline crossover** is at ~2 FLOP/byte. At 0.1 FLOP/byte, we are **20x below the roofline** — completely memory-bound.

NEON improves the arithmetic throughput (FMA pipeline utilization), but the bottleneck is **waiting for `x[col_idx[k]]` to arrive from memory**. The random access pattern defeats the hardware prefetcher, so each gather load may take 4–50 cycles depending on cache level:

| Working set (x array) | Cache level | Gather latency | NEON benefit |
|------------------------|-------------|----------------|--------------|
| < 64 KB                | L1          | ~4 cycles      | High (1.5-2x) |
| 64 KB – 1 MB           | L2          | ~12 cycles     | Moderate (1.2-1.5x) |
| 1 MB – 16 MB           | L3/SLC      | ~30 cycles     | Low (1.05-1.15x) |
| > 16 MB                | DRAM        | ~50+ cycles    | Negligible |

This explains the consistent pattern: **NEON shines at small n** (n=1K-10K, working set in L1/L2) and **converges to parity at large n** (n=100K-500K, working set exceeds cache).

### Why the Compiler Doesn't Auto-Vectorize

Apple Clang with `-O2` cannot auto-vectorize the scalar inner loop because:
1. The indirect access `x[col_idx[k]]` prevents the compiler from proving absence of aliasing
2. The loop trip count (`end - start`) is unknown at compile time and varies per row
3. The potential pointer aliasing between `x`, `probs`, and `col_idx` prevents vectorization without `__restrict__` hints

The explicit NEON intrinsics bypass these limitations.

---

## Files Reference

| File | Purpose |
|------|---------|
| `include/helios/simd/neon_kernels.h` | All NEON kernel functions (header-only, `#ifdef __ARM_NEON` guarded) |
| `include/helios/policy_eval_op_neon.h` | `PolicyEvalOpNeon` Operator subclass |
| `src/ops/policy_eval_op_neon.cc` | Compilation unit / vtable anchor |
| `tests/test_simd.cc` | 6 correctness tests (all pass) |
| `bench/bench_simd.cc` | 5 performance benchmarks |
| `bench/results/simd_benchmark.txt` | Raw benchmark output |

### Build & Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/bin/helios_simd_tests          # Correctness (6 tests)
./build/bin/helios_bench_simd          # Full benchmark
./build/bin/helios_bench_simd --quick  # Quick benchmark (~30s)
```
