# Helios

**Helios** is a production-grade C++20 fixed-point solver for contractive operators, with a focus on **policy evaluation in Markov Decision Processes (MDPs)**. It implements a family of iterative solvers — from classical single-threaded Jacobi to lock-free asynchronous multi-threaded execution — and benchmarks them rigorously on problems ranging from n = 1,000 to n = 1,000,000 states.

The central research question: **when and why do asynchronous solvers converge on problems where all synchronous methods fail?**

---

## What This Study Is About

### The Problem

Policy evaluation in MDPs requires solving the Bellman fixed-point equation:

```
V = r + β·P·V     ⟺     V = F(V)
```

where P is the transition matrix (row-stochastic), r is the reward vector, and β < 1 is the discount factor. The operator F(x) = r + β·P·x is a contraction with modulus β, so iterative methods are guaranteed to converge — in theory.

**In practice, some MDPs are structurally hard.** The *metastable MDP* — two densely-connected clusters joined by rare bridge transitions (probability `p_bridge`) — exposes a fundamental failure mode of synchronous iteration: all threads see a globally symmetric state and update in lockstep, preserving cluster symmetry and making it impossible for either cluster's values to propagate to the other. The solver runs forever without converging, not because it is slow, but because it is structurally trapped.

### The Key Finding

Asynchronous execution breaks this symmetry through timing-dependent perturbations: when one cluster's coordinates update slightly before the other's, the resulting asymmetry propagates and both clusters eventually converge. This is not a throughput advantage — it is a **convergence guarantee** that synchronous methods lack on nearly-decomposable problems.

**Empirically:** on the canonical metastable hard case (n=5,000, β=0.999, p_bridge=0.01), all four external baselines — Naive Jacobi, raw-parallel `std::thread`, OpenMP, and Eigen SpMV — timeout at 60 seconds without converging. Helios `Async_Static` converges in 8.1 seconds.

### Scope

This repository implements:

- **10 solver variants**: Jacobi, Gauss-Seidel, and 8 async/plan configurations covering 5 schedulers and 3 planners
- **4 external baselines**: Naive, RawParallel (`std::thread`), OpenMP, and Eigen Jacobi
- **6 benchmark dimensions**: convergence, thread scaling, size scaling, beta sensitivity, difficulty spectrum, and auto-tuning
- **A formal analysis** in `docs/helios_paper.tex` covering the metastable MDP spectral theory, the Cheeger inequality, and the symmetry-breaking theorem

---

## Repository Layout

```
helios/
├── include/helios/         # Public headers
│   ├── operator.h          # Abstract operator interface
│   ├── scheduler.h         # Abstract scheduler interface
│   ├── runtime.h           # RuntimeConfig, RunResult, Mode enum
│   ├── plan.h              # Task, Phase, EpochPlan structures
│   ├── planner.h           # StaticPlanner, ColoredPlanner, PriorityPlanner
│   ├── mdp.h               # MDP struct (CSR storage)
│   ├── policy_eval_op.h    # Bellman operator F_i(x) = r_i + β·Σ P_ij·x_j
│   ├── profiling.h         # Per-thread timing instrumentation (Plan mode)
│   ├── autotune.h          # Automatic planner configuration selection
│   └── schedulers/         # Five scheduler implementations
├── src/                    # Implementation files
│   ├── runtime.cc          # Jacobi, GS, Async, Plan execution loops
│   ├── planners.cc         # EpochPlan builders
│   ├── schedulers/         # Scheduler implementations
│   └── ops/                # PolicyEvalOp, NEON SIMD variant
├── bench/
│   ├── run_bench.cc        # Main benchmark suite (6 benchmark suites)
│   ├── baselines.cc        # External baseline comparison binary
│   ├── gen/                # MDP generators (grid, metastable, random)
│   └── results/            # Generated CSV output files
├── tests/                  # Unit and integration tests
├── tools/
│   └── plot.py             # Publication-quality plot generation
├── docs/
│   ├── helios_paper.tex    # Full LaTeX paper with proofs
│   └── architecture_and_results.md  # Detailed architecture and results doc
└── cmake/                  # CMake helper modules
```

---

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (Clang ≥ 14 or GCC ≥ 12) | **Yes** | Core library |
| CMake ≥ 3.20 | **Yes** | Build system |
| libomp (Homebrew on macOS) | Optional | `OpenMP_Jacobi` baseline |
| Eigen3 (Homebrew on macOS) | Optional | `Eigen_Jacobi` baseline |
| Python 3 + matplotlib + pandas | Optional | Plot generation |

### macOS (Apple Silicon / Homebrew)

```bash
brew install libomp eigen
pip3 install matplotlib pandas numpy
```

### Linux (apt)

```bash
apt-get install libomp-dev libeigen3-dev python3-matplotlib python3-pandas
```

---

## Build

```bash
# Configure (Release, with OpenMP + Eigen baselines)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything (library + benchmarks + tests)
cmake --build build -j$(nproc)

# Build targets individually
cmake --build build --target helios           # Library only
cmake --build build --target helios_bench     # Main benchmark suite
cmake --build build --target helios_baselines # External baseline comparison
cmake --build build --target helios_tests     # Test suite
```

**Optional CMake flags:**

| Flag | Default | Description |
|---|---|---|
| `HELIOS_WITH_OPENMP` | `ON` | Include OpenMP Jacobi baseline |
| `HELIOS_WITH_EIGEN` | `ON` | Include Eigen Jacobi baseline |
| `HELIOS_ENABLE_LTO` | `ON` | Link-time optimization |
| `HELIOS_ENABLE_ASAN` | `OFF` | AddressSanitizer |
| `HELIOS_ENABLE_UBSAN` | `OFF` | UndefinedBehaviorSanitizer |
| `HELIOS_BUILD_TESTS` | `ON` | Build test suite |
| `HELIOS_BUILD_BENCH` | `ON` | Build benchmark executables |

---

## Reproducing Results

All benchmark output is written to `bench/results/`. After building, run the steps below in order.

### Step 1 — Verify the test suite passes

```bash
./build/bin/helios_tests
```

Expected: all tests pass (smoke test, operator contract, schedulers, complex MDPs, Phase 3, stress, SIMD).

### Step 2 — Run the main benchmark suite

```bash
mkdir -p bench/results
./build/bin/helios_bench
```

Runtime: ~5–15 minutes depending on hardware. Writes:

| File | Contents |
|---|---|
| `bench/results/summary.csv` | All 10 Helios solvers × 7 MDPs: wall time, throughput, bandwidth |
| `bench/results/convergence_traces.csv` | Residual-vs-time traces for all (solver, MDP) pairs |
| `bench/results/thread_scaling.csv` | T={1,2,4,8} on Rand_500K and Rand_1M |
| `bench/results/size_scaling.csv` | n={1K,5K,20K,100K} across Jacobi/GS/Plan/Async |
| `bench/results/difficulty_spectrum.csv` | p_bridge={0.2,0.1,0.05,0.02,0.01} for Async_TopKGS |
| `bench/results/autotune.csv` | Auto-tuning results for Grid/Meta/Rand |

### Step 3 — Run the baseline comparison

```bash
./build/bin/helios_baselines
```

Runtime: ~3–10 minutes. Writes:

| File | Contents |
|---|---|
| `bench/results/baselines.csv` | Naive / RawParallel / OpenMP / Eigen vs Helios on easy and metastable MDPs |

Requires OpenMP and Eigen3 to be found at configure time; if missing, those rows are skipped with a warning.

### Step 4 — Generate plots

```bash
python3 tools/plot.py bench/results
```

Writes all figures to `bench/results/` (PNG format):

| Plot | Shows |
|---|---|
| `conv_{MDP}.png` | Convergence curves (residual vs wall time) |
| `ranking_{MDP}.png` | Solver wall-time bar charts |
| `throughput_{MDP}.png` | Throughput bar charts |
| `thread_scaling_{Rand_500K,Rand_1M}.png` | UPS and speedup vs thread count |
| `size_scaling.png` | Wall time and throughput vs problem size (log-log) |
| `beta_sensitivity.png` | Convergence time vs discount factor |
| `difficulty_spectrum.png` | Solve time vs spectral gap (2·p_bridge) |
| `heatmap.png` | MDP × Solver wall-time heatmap |
| `autotune.png` | Auto-tuning results |

### Step 5 — Compile the paper (optional)

```bash
cd docs
pdflatex helios_paper.tex
pdflatex helios_paper.tex   # twice for cross-references
```

---

## Solvers

### Helios Execution Modes

| Solver | Mode | Threads | Strategy |
|---|---|---|---|
| `Jacobi` | Synchronous | 1 | Double-buffer: read old, write new, swap |
| `GaussSeidel` | Synchronous | 1 | In-place: each update uses freshest values |
| `Async_Static` | Asynchronous | T | Lock-free workers, static block ownership |
| `Async_Shuffled` | Asynchronous | T | Static blocks, reshuffle per epoch |
| `Async_TopKGS` | Asynchronous | T | Dispatch top-K highest-residual coordinates first |
| `Async_CATopK` | Asynchronous | T | Conflict-aware top-K: group hot indices by cache block |
| `Async_ResBucket` | Asynchronous | T | Log-scale residual bucket priority |
| `Plan_Static` | Plan (barrier) | T | Compiled epoch plan, static contiguous partitions |
| `Plan_Colored` | Plan (barrier) | T | Graph-coloring to avoid cross-thread cache conflicts |
| `Plan_Priority` | Plan (barrier) | T | Two-phase: hot coordinates first, remainder second |

### External Baselines (helios_baselines)

| Solver | Implementation |
|---|---|
| `Naive_Jacobi` | Plain C++ double-buffer, no parallelism |
| `RawParallel_Jacobi` | `std::thread` + `std::barrier`, 4 threads |
| `OpenMP_Jacobi` | `#pragma omp parallel for`, 4 threads |
| `Eigen_Jacobi` | `Eigen::SparseMatrix` SpMV, 1 thread |

---

## Benchmark MDPs

| MDP | n | β | Structure | Character |
|---|---|---|---|---|
| `Grid_50x50` | 2,500 | 0.999 | 2D grid, 4-neighbor + self-loop | Regular, local |
| `Meta_2K` | 2,000 | 0.999 | Two clusters, p_bridge = 0.05 | **Metastable hard case** |
| `Star_2K` | 2,000 | 0.999 | Hub-and-spoke | Hub aggregates all state information |
| `Chain_2K` | 2,000 | 0.999 | Linear chain | Slow propagation, tridiagonal |
| `Rand_4K` | 4,000 | 0.999 | Random sparse, ~8 nnz/row | Unstructured, well-mixed |
| `Rand_500K` | 500,000 | 0.99 | Random sparse, 20 nnz/row | Large-scale throughput |
| `Rand_1M` | 1,000,000 | 0.99 | Random sparse, 20 nnz/row | Memory-bandwidth-limited |

---

## Key Results

### Metastable MDP: Structural Convergence Failure

All synchronous solvers (Jacobi, GS, all Plan variants) **timeout** on `Meta_2K` (n=2,000, β=0.999, p_bridge=0.05). All async solvers converge:

| Solver | Wall Time | Converged |
|---|---|---|
| Async_Static 4T | 8.1s | **Yes** |
| Async_Shuffled 4T | 10.6s | **Yes** |
| Async_ResBucket 4T | 10.3s | **Yes** |
| Jacobi 1T | 30.0s | No (timeout) |
| Plan_Static 4T | 30.0s | No (timeout) |

External baselines on a harder metastable instance (n=5,000, β=0.999, p_bridge=0.01, 60s cap):

| Implementation | Converged | Final Residual |
|---|---|---|
| Naive_Jacobi | **No** | 2.06 × 10⁻¹ |
| RawParallel_Jacobi | **No** | 4.76 × 10⁻³ |
| OpenMP_Jacobi | **No** | 7.66 × 10⁻⁴ |
| Eigen_Jacobi | **No** | 7.19 × 10⁻⁶ |
| **Helios Async_Static** | **Yes** | 8.8 × 10⁻⁷ (8.1s) |

The failure is not hardware — all baselines run at 10⁵–10⁶ updates/second. It is not speed — doubled throughput cannot escape the symmetry trap. It is a **structural convergence failure** caused by the cluster-constant invariant proven in the paper.

### Baseline Comparison: Easy MDP (n=500K, β=0.99, ε=10⁻⁶)

| Implementation | Threads | Wall (s) | Speedup vs Naive |
|---|---|---|---|
| Naive_Jacobi | 1 | 11.125 | 1.00× |
| RawParallel_Jacobi | 4 | 4.684 | 2.38× |
| OpenMP_Jacobi | 4 | 3.769 | 2.95× |
| Eigen_Jacobi | 1 | 4.173 | 2.67× |
| Helios Async_Static | 4 | 4.668 | 2.38× |
| **Helios Plan_Static** | **4** | **3.028** | **3.67×** |

Plan_Static beats OpenMP by 20% at n=500K because it eliminates redundant full-array sweeps (353.5M total updates vs 653.5M for Jacobi variants).

### Thread Scaling (Rand_500K, n=500,000)

| Threads | Plan_Static UPS | Async_Static UPS |
|---|---|---|
| 1 | 63.6M | 50.0M |
| 2 | 80.7M | 71.9M |
| 4 | **118.5M (1.86×)** | **115.9M (2.32×)** |
| 8 | 114.2M (1.80×) | 33.0M (0.66×) |

Peak at T=4 (4 performance cores on Apple Silicon). At T=8, efficiency cores participate and Async's monitor thread contends for CPU, causing severe degradation.

### Memory Bandwidth (n=500K, β=0.99)

At 20 nnz/row, each coordinate update accesses ~416 bytes. Plan_Static at T=4 reaches **48.6 GB/s** — 49% of the Apple M-series unified memory peak — confirming the system is memory-bandwidth limited at this scale.

---

## Architecture

Helios is built around three clean abstractions:

```
Operator(n, apply_i, residual_i)    Scheduler(init, next, notify)
               │                              │
               └──────────────┬───────────────┘
                              ▼
                        Runtime::run()
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           Jacobi        GaussSeidel     Async / Plan
                                              │
                                          RunResult
```

- **Operator**: pure interface, thread-safe `apply_i_async` variant uses `atomic_ref` for concurrent reads
- **Scheduler**: controls which coordinate gets updated next; stateless (Static) to stateful priority (TopKGS)
- **Runtime**: persistent worker threads, `alignas(128)` padded counters to eliminate false sharing, monitor thread running an O(n) residual scan on a configurable interval

Plan mode uses a pre-compiled `EpochPlan` (built by a Planner) with `std::barrier` synchronization between phases.

---

## Paper

`docs/helios_paper.tex` is a self-contained technical paper covering:

- The Bellman operator as a contraction and its condition number `1/(1-β)`
- Formal definition of the metastable MDP (Definition 5.1)
- Spectral analysis: second eigenvalue λ₂ = 1 − 2p_bridge, spectral gap δ = 2p_bridge (Proposition 5.2)
- The Cheeger inequality connecting bottleneck structure to mixing time (Theorem 5.4)
- Synchronous failure theorem: Jacobi/GS preserve within-cluster symmetry indefinitely (Theorem 5.5)
- Corollary on cross-cluster symmetric MDPs and wrong-limit convergence (Corollary 5.6)
- Experimental results with interpretive analysis for all 6 benchmark dimensions

---

## Citation

If you use this code or findings, please cite as:

```
@misc{helios2026,
  title  = {Helios: Asynchronous Fixed-Point Iteration for Nearly-Decomposable MDPs},
  year   = {2026},
  note   = {https://github.com/tsg/helios}
}
```

---

## License

See [LICENSE](LICENSE).

