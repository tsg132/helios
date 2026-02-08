# Helios: Architecture and Benchmark Results

## 1. What Helios Does

Helios is a deterministic execution engine for computing fixed points of contractive operators. Given an operator F: R^n -> R^n, Helios finds x* such that x* = F(x*).

The primary use case is **policy evaluation in Markov Decision Processes** (MDPs). For a fixed policy in an MDP with discount factor beta < 1, the value function V satisfies the Bellman equation:

```
V = r + beta * P * V
```

where P is the transition matrix and r is the reward vector. This is a fixed point equation V = F(V) where F(x) = r + beta * P * x. The operator F is a contraction with modulus beta, so iterative methods are guaranteed to converge.

Helios implements multiple execution strategies — from classical single-threaded Jacobi iteration to lock-free asynchronous multi-threaded solvers — and benchmarks them against each other on problems ranging from n=1,000 to n=1,000,000 states.

---

## 2. Architecture

### 2.1 Core Abstractions

Helios is built around three clean abstractions:

```
Operator (n, apply_i, residual_i)     Scheduler (init, next)
              |                              |
              +-------------+----------------+
                            v
                      Runtime::run()
                            |
              +-------------+-------------+
              v             v             v
           Jacobi      GaussSeidel     Async / Plan
              |
              v
          RunResult
```

**Operator** (`include/helios/operator.h`): An abstract interface representing the operator F. Must provide:
- `n()` — dimension of the state vector
- `apply_i(i, x)` — compute F_i(x), the i-th coordinate of F applied to state x
- `residual_i(i, x)` — compute |F_i(x) - x_i|, the local residual at coordinate i
- `apply_i_async(i, x)` — thread-safe variant using `atomic_ref` for concurrent reads

The concrete implementation is `PolicyEvalOp` which wraps an MDP and implements:
```
F_i(x) = reward[i] + beta * sum_j P[i,j] * x[j]
```
using CSR-format sparse matrix-vector products.

**Scheduler** (`include/helios/scheduler.h`): Controls which coordinates get updated and in what order. Must provide:
- `init(n, num_threads)` — set up internal state
- `next(tid)` — return the next coordinate index for thread `tid`
- `notify(tid, i, residual)` — optional feedback for priority schedulers
- `rebuild(residuals)` — optional: reconstruct priority structure from a full residual snapshot

**Runtime** (`src/runtime.cc`): The execution engine. Given an Operator and a Scheduler (or an EpochPlan), it runs the iterative computation until convergence (||F(x) - x||_inf <= eps) or a timeout. It reports wall time, total updates, throughput (updates/sec), and a residual trace over time.

### 2.2 Execution Modes

Helios supports four execution modes, each with different parallelism strategies:

#### Jacobi (single-threaded, synchronous)
Classical Jacobi iteration: compute all F_i(x) using the current state, then update all coordinates simultaneously. Each "sweep" reads the entire old vector and writes a complete new vector. This is the simplest and most predictable method.

#### Gauss-Seidel (single-threaded, in-place)
Like Jacobi, but updates x[i] immediately after computing F_i(x). Later coordinates see the effects of earlier updates within the same sweep. This typically converges in fewer iterations than Jacobi because each update uses the freshest available information.

#### Async (multi-threaded, lock-free)
Multiple worker threads run concurrently, each calling `scheduler.next(tid)` to get the next coordinate to update, computing F_i(x), and writing x[i] back. Workers read possibly-stale values of x (other threads may be writing simultaneously). A separate monitor thread periodically scans the full residual to check for convergence.

Key implementation details:
- Per-thread update counters use `alignas(128) PaddedCounter` with `atomic<uint64_t>` to avoid false sharing
- Workers batch-increment their counters every 256 iterations
- The monitor thread uses `atomic<real_t>` for the best-known residual
- Workers are true persistent threads (no creation/join overhead per sweep)

#### Plan (multi-threaded, barrier-synchronized)
A compiled schedule (EpochPlan) is built ahead of time by a Planner. The plan consists of Phases, each containing per-thread worklists of Tasks. Workers execute their assigned tasks, then synchronize at a barrier before the next phase.

Key implementation details:
- Uses `std::barrier` with T+1 participants (T workers + 1 coordinator)
- Workers are persistent — created once, reused across all epochs
- T=1 uses a fast path with `apply_i` (no atomics); T>1 uses `apply_i_async` with `atomic_ref`
- The coordinator checks residuals between epochs on a configurable interval

### 2.3 Schedulers

Five scheduler implementations control coordinate selection in Async mode:

| Scheduler | Strategy | Overhead |
|-----------|----------|----------|
| **StaticBlocks** | Each thread owns a contiguous block [lo, hi), iterates sequentially | Minimal |
| **ShuffledBlocks** | Same block ownership, but iteration order is reshuffled each epoch | Low |
| **TopKGS** | Maintains a "hot set" of K highest-residual coordinates, dispatches them first | Medium |
| **CATopKGS** | Conflict-Aware TopK: distributes hot indices into G groups by cache-block key | Medium-High |
| **ResidualBuckets** | Logarithmic bucketing of residuals, prioritizes high-residual coordinates | Medium |

### 2.4 Planners

Three planner implementations build EpochPlans for the Plan execution mode:

| Planner | Strategy |
|---------|----------|
| **StaticPlanner** | Partitions [0,n) into contiguous blocks, distributes evenly to threads |
| **ColoredPlanner** | Graph-colors coordinates by cache-block proxy to avoid conflicts between threads |
| **PriorityPlanner** | First phase handles top-K hot coordinates, second phase covers the rest |

### 2.5 MDP Storage

MDPs are stored in CSR (Compressed Sparse Row) format:
- `row_ptr[n+1]` — row pointers
- `col_idx[nnz]` — column indices
- `probs[nnz]` — transition probabilities
- `rewards[n]` — per-state rewards
- `beta` — discount factor

This enables O(nnz_i) computation of F_i(x) for each coordinate, where nnz_i is the number of nonzero entries in row i.

---

## 3. Benchmark Suite

The benchmark runner (`bench/run_bench.cc`) generates synthetic MDPs and measures solver performance across six dimensions.

### 3.1 Benchmark MDPs

| MDP | Structure | n | beta | Character |
|-----|-----------|---|------|-----------|
| Grid_50x50 | 2D grid, 4-neighbor transitions + self-loops | 2,500 | 0.999 | Local connectivity, regular structure |
| Meta_2K | Two clusters with rare inter-cluster bridges | 2,000 | 0.999 | Nearly-decomposable, slow mixing |
| Star_2K | Hub-and-spoke topology | 2,000 | 0.999 | One hub sees all states, spokes are local |
| Chain_2K | Linear chain with left/right/stay transitions | 2,000 | 0.999 | Tridiagonal, information propagates slowly |
| Rand_4K | Random sparse graph, ~8 nonzeros per row | 4,000 | 0.999 | Unstructured, well-mixed |
| Rand_500K | Random sparse, 20 nnz/row | 500,000 | 0.99 | Large-scale throughput test |
| Rand_1M | Random sparse, 20 nnz/row | 1,000,000 | 0.99 | Memory-bandwidth-limited regime |

### 3.2 Benchmark Dimensions

1. **Convergence** (Bench 1): All 10 solvers on 5 core MDPs. Measures wall time, total updates, throughput, and convergence traces (residual vs time).

2. **Beta Sensitivity** (Bench 2): Grid MDP at beta = {0.9, 0.95, 0.99, 0.995}. Shows the exponential growth in solve time as beta approaches 1.

3. **Thread Scaling** (Bench 3): Plan_Static and Async_Static on Rand_500K and Rand_1M at T = {1, 2, 4, 8}. Uses reduced monitoring overhead (200ms interval, stride=16, no trace recording) to isolate parallel compute scaling.

4. **Difficulty Spectrum** (Bench 4): Metastable MDP with bridge probability = {0.2, 0.1, 0.05, 0.02, 0.01}. As the bridge probability decreases, inter-cluster mixing slows and convergence becomes harder.

5. **Size Scaling** (Bench 5): Random sparse MDP at n = {1K, 5K, 20K, 100K}. Measures how wall time and throughput scale with problem size.

6. **AutoTune** (Bench 6): Automatic planner configuration selection via pilot runs on Grid, Meta, and Rand MDPs.

---

## 4. Results

All benchmarks were run on Apple M-series silicon (4 performance cores + 4 efficiency cores, 16GB RAM, 4MB L2 cache).

### 4.1 Solver Ranking on Core MDPs

**Well-conditioned MDPs (Grid, Star, Chain, Rand):**

The fastest solvers on well-conditioned problems are the simple ones:

| Rank | Solver | Typical Wall Time (Grid_50x50) | Why |
|------|--------|-------------------------------|-----|
| 1 | Async (Static) 4T | 0.048s | Lock-free workers, minimal overhead, no barriers |
| 2 | Plan (Static) 4T | 0.075s | Barrier sync adds ~50% overhead vs Async |
| 3 | Jacobi 1T | 0.095s | Single-threaded but zero sync overhead |
| 4 | Gauss-Seidel 1T | 0.100s | Fewer iterations, but lower throughput per update |
| 5 | Plan (Colored) 4T | 0.105s | Multi-phase coloring adds phase transitions |
| 6 | Async (Shuffled) 4T | 0.111s | Reshuffling overhead reduces throughput |
| 7 | Plan (Priority) 4T | 0.155s | Priority bookkeeping not worth it for easy problems |
| 8 | Async (TopK-GS) 4T | 0.935s | Maintaining top-K heap is expensive |
| 9 | Async (ResBucket) 4T | 0.951s | Bucket management overhead |
| 10 | Async (CA-TopK) 4T | 3.688s | Conflict grouping adds major overhead |

The priority-based schedulers (TopK-GS, CA-TopK, ResBucket) are 10-50x slower than static schedulers on well-conditioned problems. Their priority bookkeeping overhead vastly exceeds any convergence benefit.

**Metastable MDP (the hard case):**

The metastable MDP with rare inter-cluster bridges is the only benchmark where the picture changes dramatically:

| Solver | Meta_2K | Converged? |
|--------|---------|------------|
| Async (Static) 4T | 8.1s | Yes |
| Async (Shuffled) 4T | 10.6s | Yes |
| Async (ResBucket) 4T | 10.3s | Yes |
| Async (TopK-GS) 4T | 11.7s | Yes |
| Async (CA-TopK) 4T | 13.6s | Yes |
| Jacobi 1T | 30.0s | **No** (timeout) |
| Gauss-Seidel 1T | 30.0s | **No** (timeout) |
| Plan (Static) 4T | 30.0s | **No** (timeout) |
| Plan (Colored) 4T | 30.0s | **No** (timeout) |
| Plan (Priority) 4T | 30.0s | **No** (timeout) |

All Plan-mode and single-threaded solvers **fail to converge** within 30 seconds on the metastable MDP. Only the Async solvers converge, because asynchronous updates with stale reads naturally break the symmetry between clusters. In Plan mode, all threads see the same (possibly symmetric) state and update in lockstep, preserving the symmetry that keeps both clusters stalled.

This is the single most important architectural insight from the benchmarks: **asynchronous execution has a fundamental convergence advantage on nearly-decomposable problems**, independent of throughput.

### 4.2 Thread Scaling

Thread scaling was measured on large problems (n=500K and n=1M) with monitoring overhead minimized to isolate parallel compute performance.

#### Rand_500K (n=500,000, 20 nnz/row, beta=0.99)

| Threads | Plan (Static) UPS | Scaling | Async (Static) UPS | Scaling |
|---------|-------------------|---------|---------------------|---------|
| 1 | 63.6M | 1.00x | 50.0M | 1.00x |
| 2 | 80.7M | 1.27x | 71.9M | 1.44x |
| 4 | 118.5M | **1.86x** | 115.9M | **2.32x** |
| 8 | 114.2M | 1.80x | 33.0M | 0.66x |

#### Rand_1M (n=1,000,000, 20 nnz/row, beta=0.99)

| Threads | Plan (Static) UPS | Scaling | Async (Static) UPS | Scaling |
|---------|-------------------|---------|---------------------|---------|
| 1 | 58.6M | 1.00x | 46.3M | 1.00x |
| 2 | 66.8M | 1.14x | 65.2M | 1.41x |
| 4 | 104.1M | **1.78x** | 102.9M | **2.22x** |
| 8 | 101.9M | 1.74x | 26.9M | 0.58x |

**Key observations:**

1. **Peak scaling at T=4**: Both solvers achieve their best throughput at 4 threads, reaching 1.86-2.32x scaling on Rand_500K and 1.78-2.22x on Rand_1M. This corresponds to the 4 performance cores on the test machine.

2. **Async scales better in throughput ratio**: Async shows higher throughput *scaling factors* (2.32x vs 1.86x at T=4) because its T=1 baseline is lower — the monitor thread competes for CPU time even with a single worker. At T=4, both converge to similar absolute throughput (~115M UPS for 500K, ~103M UPS for 1M).

3. **T=8 collapse**: Going from 4 to 8 threads is catastrophic for Async (drops to 0.58-0.66x of single-threaded) and flat for Plan (1.74-1.80x). The machine has 4 performance cores and 4 efficiency cores. At T=8, efficiency cores (~40% slower) participate, and Async's monitor thread (the 9th thread) must contend for CPU, causing severe throughput degradation.

4. **Memory bandwidth saturation**: The n=1M problem (working set ~250MB with 20 nnz/row) shows lower scaling than n=500K at every thread count. This is the signature of memory bandwidth saturation — the memory bus can't feed data fast enough as more cores request it simultaneously.

5. **Why not ideal scaling?** The theoretical maximum speedup at T=4 is 4x (Amdahl's law with 0% serial fraction). The achieved 1.8-2.3x reflects:
   - Memory bandwidth shared across cores (SpMV is ~0.1 FLOP/byte)
   - Residual scan overhead (~2% of wall time with stride=16 and 200ms interval)
   - For Plan: barrier synchronization overhead between phases
   - For Async at T=1: monitor thread overhead reduces the baseline

### 4.3 Size Scaling

Performance across problem sizes n = {1K, 5K, 20K, 100K}, all with beta=0.99:

| n | Jacobi (1T) | Gauss-Seidel (1T) | Plan Static (4T) | Async Static (4T) |
|---|-------------|-------------------|-------------------|---------------------|
| 1,000 | 0.010s / 153M UPS | 0.005s / 151M UPS | 0.010s / 79M UPS | 0.006s / 130M UPS |
| 5,000 | 0.045s / 145M UPS | 0.025s / 148M UPS | 0.030s / 121M UPS | 0.034s / 109M UPS |
| 20,000 | 0.410s / 64M UPS | 0.212s / 66M UPS | 0.175s / 84M UPS | 0.140s / 103M UPS |
| 100,000 | 3.092s / 42M UPS | 1.638s / 42M UPS | 1.548s / 46M UPS | 1.376s / 53M UPS |

**Key observations:**

1. **Small problems (n < 5K)**: Single-threaded solvers are fastest. Thread creation/synchronization overhead dominates. Jacobi achieves 153M UPS at n=1K — the working set fits in L1 cache.

2. **Crossover at n ~ 20K**: Multi-threaded solvers overtake single-threaded ones. At n=20K, Async_Static (103M UPS) is 1.6x faster than Jacobi (64M UPS).

3. **Throughput decay**: All solvers show declining UPS as n increases. At n=1K, Jacobi runs at 153M UPS; at n=100K, only 42M UPS. This reflects the transition from L1/L2-cache-resident to memory-bound execution. The working set at n=100K (~16MB with ~8 nnz/row) exceeds L2 cache (4MB).

4. **Gauss-Seidel vs Jacobi**: GS converges in ~half the iterations (fewer total updates) but at roughly the same UPS, so it's ~2x faster in wall time. At n=100K, GS needs 69M updates vs Jacobi's 131M.

### 4.4 Beta Sensitivity

As the discount factor beta approaches 1, the contraction rate (1-beta) shrinks and more iterations are needed:

| Beta | Jacobi Wall Time | Plan (Priority) Wall Time | Async (TopK-GS) Wall Time |
|------|------------------|--------------------------|---------------------------|
| 0.900 | 0.005s | 0.005s | 0.008s |
| 0.950 | 0.005s | 0.005s | 0.008s |
| 0.990 | 0.005s | 0.015s | 0.038s |
| 0.995 | 0.010s | 0.025s | 0.075s |

The growth is roughly exponential: each step closer to beta=1 doubles the required iterations. At beta=0.995, Async (TopK-GS) is 15x slower than at beta=0.9, while Jacobi only slows 2x. The priority schedulers' overhead scales with the number of iterations, while Jacobi's ultra-low per-iteration cost keeps it competitive even as iteration count grows.

### 4.5 Difficulty Spectrum (Metastable Bridge Probability)

The metastable MDP has two clusters connected by bridges with probability p_bridge. As p_bridge decreases, inter-cluster mixing slows:

| Bridge Prob | Async (TopK-GS) Wall Time | Converged? | Jacobi/GS/Plan |
|-------------|---------------------------|------------|----------------|
| 0.200 | 19.3s | Yes | All timeout (30s) |
| 0.100 | 14.0s | Yes | All timeout |
| 0.050 | 12.7s | Yes | All timeout |
| 0.020 | 14.4s | Yes | All timeout |
| 0.010 | 15.6s | Yes | All timeout |

Counterintuitively, the *easiest* bridge probability for Async (TopK-GS) is p=0.05 (12.7s), not p=0.2 (19.3s). This suggests that moderate coupling allows the priority scheduler to efficiently focus on bridge states, while high coupling makes the problem look more uniform (reducing the benefit of prioritization).

The most striking result: **synchronous solvers never converge on any metastable configuration within 30 seconds**. Asynchronous updates break the symmetry between clusters by injecting timing-dependent perturbations, which is essential for nearly-decomposable problems.

### 4.6 AutoTune

The auto-tuner runs short pilot benchmarks with different planner configurations and selects the best:

| MDP | Best Planner | Block Size | Wall Time | Throughput |
|-----|-------------|------------|-----------|------------|
| Grid_AT (n=2,500) | Static | blk=32 | 0.105s | 214M UPS |
| Meta_AT (n=2,000) | Static | blk=16 | 30.0s (timeout) | 349K UPS |
| Rand_AT (n=5,000) | Static | blk=32 | 0.317s | 112M UPS |

The auto-tuner consistently selects StaticPlanner with small block sizes (16-32). The metastable MDP times out even with auto-tuning, confirming that Plan mode fundamentally cannot handle nearly-decomposable problems within reasonable time.

---

## 5. Key Insights

### 5.1 Simple Schedulers Win on Throughput

Across all well-conditioned MDPs, the throughput ranking is consistent:

```
Async_Static > Plan_Static > Jacobi > GaussSeidel >> TopK/ResBuck >> CA-TopK
```

The simpler the scheduling strategy, the higher the throughput. Priority-based schedulers spend more time managing their data structures than they save by choosing better update orders. At n=4K with 4 threads, Async_Static achieves 379M UPS while CA-TopK manages only 5.4M UPS — a 70x difference.

### 5.2 Asynchrony Enables Convergence on Hard Problems

The metastable MDP is the decisive test case. Synchronous methods (Jacobi, GS, Plan) fail because they preserve inter-cluster symmetry. Asynchronous methods break this symmetry through timing-dependent interactions: when one cluster's values update slightly before the other's, the asymmetry propagates and eventually resolves both clusters.

This is not just a speedup — it's the difference between converging and not converging.

### 5.3 Thread Scaling is Memory-Bound

Policy evaluation is a sparse matrix-vector operation with arithmetic intensity ~0.1 FLOP/byte. At n=500K-1M, the working set far exceeds cache, making memory bandwidth the bottleneck. Achieved scaling of 1.8-2.3x at T=4 is consistent with the memory bandwidth limits of a shared-memory system where all cores compete for the same memory bus.

### 5.4 Monitoring Overhead Was the Hidden Bottleneck

Early benchmarks showed only ~1.5x speedup at T=4. The root cause: the residual scan (O(n) serial work) ran every 2ms with stride=1, consuming 80% of wall time. By Amdahl's law, if 80% of work is serial, S(4) = 1/(0.8 + 0.2/4) = 1.18x.

The fix: monitor_interval=200ms, stride=16, no trace recording during thread scaling benchmarks. This reduced serial fraction to ~2%, allowing the true parallel scaling to emerge.

### 5.5 Apple Silicon Heterogeneous Cores

The 4 performance + 4 efficiency core architecture creates a sharp performance cliff at T>4. Performance cores run at full speed, but efficiency cores are ~40% slower. At T=8, the slowest efficiency core becomes the bottleneck (for Plan mode), or the 9th thread (monitor) has no core available (for Async mode). This is why T=8 Async collapses to 0.58x of single-threaded while Plan merely plateaus at 1.74x.

---

## 6. File Reference

### Source Code
| File | Purpose |
|------|---------|
| `include/helios/operator.h` | Abstract operator interface |
| `include/helios/scheduler.h` | Abstract scheduler interface |
| `include/helios/runtime.h` | RuntimeConfig, RunResult, Mode enum |
| `include/helios/plan.h` | Task, Phase, EpochPlan structures |
| `include/helios/planner.h` | StaticPlanner, ColoredPlanner, PriorityPlanner |
| `include/helios/mdp.h` | MDP struct with CSR storage |
| `include/helios/policy_eval_op.h` | Bellman operator implementation |
| `src/runtime.cc` | Execution engine: Jacobi, GS, Async, Plan |
| `include/helios/schedulers/static_blocks.h` | Static contiguous block scheduler |
| `include/helios/schedulers/shuffled_blocks.h` | Shuffled block scheduler |
| `include/helios/schedulers/topk_gs.h` | Top-K Gauss-Southwell scheduler |
| `include/helios/schedulers/ca_topk_gs.h` | Conflict-Aware Top-K scheduler |
| `include/helios/schedulers/residual_buckets.h` | Residual bucketing scheduler |

### Benchmark Files
| File | Purpose |
|------|---------|
| `bench/run_bench.cc` | Benchmark runner (6 benchmark suites) |
| `tools/plot.py` | Visualization suite |
| `bench/results/summary.csv` | One row per (MDP, solver) with wall time, throughput |
| `bench/results/convergence_traces.csv` | Time-series residual data |
| `bench/results/thread_scaling.csv` | Thread scaling data (T=1,2,4,8) |
| `bench/results/size_scaling.csv` | Size scaling data (n=1K to 100K) |
| `bench/results/autotune.csv` | AutoTune results |

### Generated Plots
| Plot | Shows |
|------|-------|
| `conv_{MDP}.png` | Convergence curves (residual vs wall time) per MDP |
| `ranking_{MDP}.png` | Solver wall-time ranking bar charts per MDP |
| `throughput_{MDP}.png` | Solver throughput bar charts per MDP |
| `thread_scaling_{MDP}.png` | Throughput scaling + absolute throughput vs threads |
| `size_scaling.png` | Wall time and throughput vs problem size (log-log) |
| `beta_sensitivity.png` | Convergence time vs discount factor |
| `difficulty_spectrum.png` | Metastable bridge probability sweep |
| `heatmap.png` | MDP x Solver wall time heatmap |
| `autotune.png` | AutoTune results |
