# Helios Runtime and Asynchronous Execution

This document describes the Helios runtime system, execution modes, and the asynchronous fixed-point iteration implementation.

## Overview

Helios is a deterministic execution engine for computing fixed points of contractive operators:

```
x = F(x),  F: R^n -> R^n
```

The primary use case is **policy evaluation in MDPs** via the Bellman equation `V = r + βPV`.

## Architecture

```
Operator (n, apply_i, residual_i)     Scheduler (init, next)
              │                              │
              └──────────┬───────────────────┘
                         ▼
                   Runtime::run()
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Jacobi    GaussSeidel   Async
                         │
                         ▼
                    RunResult
```

## Core Interfaces

### Operator (`include/helios/operator.h`)

Abstract interface for the fixed-point operator F:

```cpp
class Operator {
    virtual index_t n() const noexcept = 0;           // Vector dimension
    virtual real_t apply_i(index_t i, const real_t* x) const = 0;  // Compute F_i(x)
    virtual real_t residual_i(index_t i, const real_t* x) const;   // |F_i(x) - x_i|

    // Async variants with atomic reads
    virtual real_t apply_i_async(index_t i, const real_t* x) const;
    virtual real_t residual_i_async(index_t i, const real_t* x) const;
};
```

**Thread Safety**: All methods must be thread-safe (read-only access to internal state). The `_async` variants use `std::atomic_ref` to read x values atomically.

### Scheduler (`include/helios/scheduler.h`)

Controls coordinate selection for async iteration:

```cpp
class Scheduler {
    virtual void init(index_t n, size_t num_threads) = 0;
    virtual index_t next(size_t tid) = 0;  // Return next coordinate for thread tid
    virtual void notify(size_t tid, index_t i, real_t residual) {}  // Optional feedback
};
```

### RuntimeConfig (`include/helios/runtime.h`)

```cpp
struct RuntimeConfig {
    size_t num_threads = 1;          // Worker threads (Async mode)
    real_t alpha = 1.0;              // Relaxation parameter
    real_t eps = 1e-6;               // Convergence tolerance
    double max_seconds = 0.0;        // Time limit (0 = unlimited)
    uint64_t max_updates = 0;        // Update limit (0 = unlimited)
    size_t monitor_interval_ms = 100; // Residual check interval
    int residual_scan_stride = 1;    // Sample every Nth coordinate
    bool record_trace = true;        // Record residual history
    Mode mode = Mode::Jacobi;        // Execution mode
};
```

## Execution Modes

### 1. Jacobi (Synchronous)

Double-buffered synchronous iteration where all coordinates see the same x^k:

```
x_i^{k+1} = (1 - α) x_i^k + α F_i(x^k)   for all i = 0,...,n-1
```

**Implementation** (`src/runtime.cc:run_jacobi_()`):
- Two buffers: `x_curr` (read) and `x_next` (write)
- Full sweep updates all coordinates
- Swap buffers after each sweep
- Check residual periodically based on `monitor_interval_ms`

**Convergence**: `‖F(x) - x‖_∞ ≤ ε`

### 2. Gauss-Seidel (Sequential)

In-place updates where later coordinates see earlier updates from the same sweep:

```
x_i^{k+1} = (1 - α) x_i^k + α F_i(x^{k+1}_0, ..., x^{k+1}_{i-1}, x^k_i, ..., x^k_{n-1})
```

**Implementation** (`src/runtime.cc:run_gauss_seidel_()`):
- Single buffer, in-place updates
- Sequential sweep i = 0, 1, ..., n-1
- Typically converges faster than Jacobi per sweep

### 3. Async (Multi-threaded)

Lock-free asynchronous coordinate updates with multiple worker threads:

```
x_i ← (1 - α) x_i + α F_i(x)   // x may contain mixed old/new values
```

**Implementation** (`src/runtime.cc:run_async_()`):

#### Thread Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Monitor Thread                        │
│  - Periodically computes ‖F(x) - x‖_∞                   │
│  - Checks convergence and time/update limits            │
│  - Sets stop flag when done                             │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Worker 0      │ │   Worker 1      │ │   Worker T-1    │
│                 │ │                 │ │                 │
│ while (!stop):  │ │ while (!stop):  │ │ while (!stop):  │
│   i = sched(0)  │ │   i = sched(1)  │ │   i = sched(T-1)│
│   fi = F_i(x)   │ │   fi = F_i(x)   │ │   fi = F_i(x)   │
│   x[i] = blend  │ │   x[i] = blend  │ │   x[i] = blend  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

#### Atomic Operations

Workers use `std::atomic_ref<real_t>` for lock-free updates:

```cpp
const real_t fi = op.apply_i_async(i, x);        // Reads x[j] atomically
atomic_ref<real_t> xi_ref(x[i]);
const real_t xi = xi_ref.load(memory_order_relaxed);
const real_t xnew = (1 - alpha) * xi + alpha * fi;
xi_ref.store(xnew, memory_order_relaxed);
```

**Memory Ordering**: Uses `memory_order_relaxed` throughout. This is sufficient because:
- Convergence is guaranteed by contractivity of F, not ordering
- Occasional stale reads just slow convergence slightly
- No correctness issues from reordering

#### Termination

Coordinated via `atomic<bool> stop`:
1. Monitor thread sets `stop = true` when converged, timed out, or update limit reached
2. Workers check `stop` each iteration and exit
3. Main thread joins all workers, then monitor

## Schedulers

### StaticBlocksScheduler (`include/helios/schedulers/static_blocks.h`)

Partitions [0, n) into T contiguous blocks, one per thread:

```
Thread 0: [0, n/T)
Thread 1: [n/T, 2n/T)
...
Thread T-1: [(T-1)n/T, n)
```

**Implementation**:

```cpp
void init(index_t n, size_t num_threads) {
    // Partition [0, n) into num_threads contiguous blocks
    // First 'remainder' threads get one extra element
    const index_t base_size = n / num_threads;
    const index_t remainder = n % num_threads;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        const index_t size = base_size + (tid < remainder ? 1 : 0);
        block_begin_[tid] = start;
        block_end_[tid] = start + size;
        cursor_[tid] = start;
        start += size;
    }
}

index_t next(size_t tid) {
    const index_t i = cursor_[tid];
    cursor_[tid] = (i + 1 < block_end_[tid]) ? (i + 1) : block_begin_[tid];
    return i;
}
```

Each thread cycles through its assigned block indefinitely.

**Properties**:
- No contention between threads (disjoint blocks)
- Good cache locality within each block
- Deterministic given fixed thread count

### Future Schedulers (TODO)

- **ShuffledBlocksScheduler**: Reshuffle block order each epoch
- **ResidualBucketsScheduler**: Prioritize high-residual coordinates

## Policy Evaluation Operator

The primary operator for MDP policy evaluation (`include/helios/policy_eval_op.h`):

```
F_i(x) = r_i + β Σ_j P_ij x_j
```

**CSR Storage** (`include/helios/mdp.h`):
```cpp
struct MDP {
    index_t n;                    // Number of states
    real_t beta;                  // Discount factor
    vector<index_t> row_ptr;      // CSR row pointers [n+1]
    vector<index_t> col_idx;      // CSR column indices [nnz]
    vector<real_t> probs;         // Transition probabilities [nnz]
    vector<real_t> rewards;       // Per-state rewards [n]
};
```

**Async Implementation** (`src/ops/policy_eval_op.cc`):
```cpp
real_t PolicyEvalOp::apply_i_async(index_t i, const real_t* x) const {
    real_t dot = 0.0;
    for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        const index_t j = col_idx[k];
        // Atomic read of x[j]
        const real_t xj = atomic_ref<real_t>(const_cast<real_t&>(x[j]))
                              .load(memory_order_relaxed);
        dot += probs[k] * xj;
    }
    return rewards[i] + beta * dot;
}
```

## Convergence Criterion

All modes use the infinity-norm residual:

```
‖F(x) - x‖_∞ = max_i |F_i(x) - x_i| ≤ ε
```

For efficiency, can sample every `residual_scan_stride` coordinates.

## RunResult

```cpp
struct RunResult {
    bool converged;              // Did we reach ε tolerance?
    real_t final_residual_inf;   // Final ‖F(x) - x‖_∞
    double wall_time_sec;        // Total wall-clock time
    uint64_t total_updates;      // Total coordinate updates
    double updates_per_sec;      // Throughput
    vector<ResidualSample> trace; // (time, residual) history
};
```

## Example Usage

```cpp
#include "helios/runtime.h"
#include "helios/mdp.h"
#include "helios/policy_eval_op.h"
#include "helios/schedulers/static_blocks.h"

// Build MDP...
MDP mdp = build_my_mdp();
PolicyEvalOp op(&mdp);

// Initial guess
vector<real_t> x(mdp.n, 0.0);

// Configure async run
RuntimeConfig cfg;
cfg.mode = Mode::Async;
cfg.num_threads = 4;
cfg.alpha = 1.0;
cfg.eps = 1e-6;
cfg.max_seconds = 60.0;

// Run
Runtime rt;
StaticBlocksScheduler sched;
RunResult result = rt.run(op, sched, x.data(), cfg);

if (result.converged) {
    printf("Converged in %.3f sec\n", result.wall_time_sec);
}
```

## Tests

The smoke tests (`tests/test_runtime_smoke.cc`) verify all three modes on a ring MDP:

| Test | Mode | Threads | n | β | ε |
|------|------|---------|---|---|---|
| `test_jacobi_ring_convergence` | Jacobi | 1 | 16 | 0.9 | 1e-6 |
| `test_gauss_seidel_ring_convergence` | GaussSeidel | 1 | 16 | 0.9 | 1e-6 |
| `test_async_ring_convergence` | Async | 2 | 16 | 0.9 | 1e-6 |
| `test_async_multithread_stress` | Async | 4 | 256 | 0.95 | 1e-5 |

All tests verify convergence to the analytical solution V = 1/(1-β) = 10.0.
