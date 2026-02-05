# Helios Complete Guide: Understanding the Codebase

This document provides a comprehensive explanation of everything implemented in Helios so far. It's designed to help you understand every interface, algorithm, and design decision.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Core Types and Memory Management](#3-core-types-and-memory-management)
4. [The Operator Interface](#4-the-operator-interface)
5. [The MDP Structure](#5-the-mdp-structure)
6. [Policy Evaluation Operator](#6-policy-evaluation-operator)
7. [The Scheduler Interface](#7-the-scheduler-interface)
8. [All Implemented Schedulers](#8-all-implemented-schedulers)
9. [The Runtime](#9-the-runtime)
10. [Execution Modes](#10-execution-modes)
11. [MDP Generators](#11-mdp-generators)
12. [How Everything Fits Together](#12-how-everything-fits-together)
13. [Project Status](#13-project-status)

---

## 1. Project Overview

### What is Helios?

Helios is a **production-grade deterministic execution engine** for computing fixed points of contractive operators. In simpler terms, it solves equations of the form:

```
x = F(x),   where F: R^n -> R^n
```

Where `x` is a vector and `F` is a function that maps vectors to vectors.

### Primary Use Case

The main application is **Policy Evaluation in Markov Decision Processes (MDPs)**. Given a policy π in an MDP, we need to compute the value function V that satisfies the **Bellman equation**:

```
V = r + β·P·V
```

Where:
- `V ∈ R^n` is the value function (one value per state)
- `r ∈ R^n` is the reward vector
- `P` is the n×n transition probability matrix
- `β ∈ [0,1)` is the discount factor

### Why is This Hard?

1. **Large scale**: n can be millions or billions of states
2. **Sparse structure**: P is typically sparse (most transitions have zero probability)
3. **Convergence speed**: Simple methods can be slow; smarter scheduling can help
4. **Parallelism**: We want to use multiple CPU cores efficiently

---

## 2. Mathematical Foundation

### Fixed-Point Iteration

The basic idea is simple: start with any initial guess x⁰, then repeatedly apply F:

```
x^(k+1) = F(x^k)
```

If F is a **contraction** (i.e., it brings points closer together), this converges to the unique fixed point x*.

### Contraction Property

A function F is a contraction with factor β < 1 if:

```
‖F(x) - F(y)‖ ≤ β · ‖x - y‖   for all x, y
```

For the Bellman operator `F(x) = r + β·P·x`, this holds because:
- P is row-stochastic (rows sum to 1), so ‖P·z‖_∞ ≤ ‖z‖_∞
- Multiplying by β < 1 makes it a contraction

### Convergence Criterion

We stop when the **residual** is small:

```
‖F(x) - x‖_∞ = max_i |F_i(x) - x_i| ≤ ε
```

This measures how close x is to being a fixed point.

### Different Iteration Schemes

1. **Jacobi (Synchronous)**: All coordinates updated together, using the same snapshot
   ```
   x^(k+1)_i = (1-α)·x^k_i + α·F_i(x^k)   for all i
   ```

2. **Gauss-Seidel (Sequential)**: Updates use the most recent values
   ```
   x_i = (1-α)·x_i + α·F_i(x)   // x may include earlier updates from this sweep
   ```

3. **Asynchronous**: Multiple threads update concurrently, seeing mixed old/new values

---

## 3. Core Types and Memory Management

**File**: `include/helios/types.h`

### Basic Types

```cpp
namespace helios {
    using index_t = uint32_t;  // Index type (supports up to 4 billion states)
    using real_t = double;     // Floating-point type for values

    constexpr size_t kCacheLine = 64;  // CPU cache line size
}
```

### Why These Choices?

- `uint32_t` for indices: 4 bytes vs 8 for uint64_t, saves memory in sparse matrices
- `double` for values: Full 64-bit precision for numerical stability

### Aligned Memory Allocation

For cache-friendly access:

```cpp
// Allocate n elements with cache-line alignment
void* aligned_malloc(size_t alignment, size_t size);
void aligned_free(void* ptr);

// Allocate array of trivially-destructible type
template<class T>
T* aligned_new_array(size_t n, size_t alignment = kCacheLine);
```

### Branch Prediction Hints

```cpp
#define HELIOS_LIKELY(x)   (__builtin_expect(!!(x), 1))
#define HELIOS_UNLIKELY(x) (__builtin_expect(!!(x), 0))
```

These help the compiler generate better code for predictable branches.

---

## 4. The Operator Interface

**File**: `include/helios/operator.h`

The `Operator` is the core abstraction representing the function F we're finding a fixed point of.

### Interface Definition

```cpp
class Operator {
public:
    virtual ~Operator() = default;

    // Dimension n of the state vector
    virtual index_t n() const noexcept = 0;

    // Compute F_i(x) - the i-th component of F(x)
    // Must be thread-safe (read-only access to internal state)
    virtual real_t apply_i(index_t i, const real_t* x) const = 0;

    // Compute |F_i(x) - x_i| - local residual at coordinate i
    virtual real_t residual_i(index_t i, const real_t* x) const {
        return std::abs(apply_i(i, x) - x[i]);
    }

    // Async versions using atomic reads (for concurrent access)
    virtual real_t apply_i_async(index_t i, const real_t* x) const;
    virtual real_t residual_i_async(index_t i, const real_t* x) const;

    // Name for logging
    virtual std::string_view name() const noexcept { return "operator"; }

    // Debug validation
    virtual void check_invariants() const {}
};
```

### Key Design Decisions

1. **Coordinate-wise interface**: We compute F one coordinate at a time, not the full vector. This enables:
   - Fine-grained scheduling
   - Better cache utilization
   - Parallel updates

2. **Thread-safety requirement**: `apply_i` must be safe to call from multiple threads simultaneously. This means operators should only read (never write) their internal state.

3. **Separate async versions**: In async mode, we need atomic reads to avoid torn reads when other threads are writing.

---

## 5. The MDP Structure

**File**: `include/helios/mdp.h`

An MDP is stored in **Compressed Sparse Row (CSR)** format.

### What is CSR Format?

CSR stores a sparse matrix efficiently by only storing non-zero entries:

```cpp
struct MDP {
    index_t n = 0;              // Number of states
    real_t beta = 0.0;          // Discount factor

    // CSR storage for transition matrix P
    vector<index_t> row_ptr;    // Size n+1: row_ptr[i] = start of row i
    vector<index_t> col_idx;    // Size nnz: column indices of non-zeros
    vector<real_t> probs;       // Size nnz: probability values

    // Rewards
    vector<real_t> rewards;     // Size n: reward for each state
};
```

### CSR Example

Consider a 3-state MDP with transition matrix:
```
P = [0.5  0.5  0.0]
    [0.0  0.3  0.7]
    [0.1  0.0  0.9]
```

CSR representation:
```
row_ptr = [0, 2, 4, 6]      // Row 0 starts at 0, row 1 at 2, row 2 at 4, end at 6
col_idx = [0, 1, 1, 2, 0, 2] // Non-zero column indices
probs   = [0.5, 0.5, 0.3, 0.7, 0.1, 0.9]  // Non-zero values
```

To iterate over row i:
```cpp
for (index_t idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx) {
    index_t j = col_idx[idx];   // Column index
    real_t p = probs[idx];      // P(i,j)
}
```

### MDP Validation

```cpp
void MDP::validate(bool strict_row_stochastic = true, real_t tol = 1e-9) const {
    // Checks:
    // - n > 0
    // - beta in [0, 1)
    // - row_ptr, col_idx, probs have correct sizes
    // - row_ptr is non-decreasing
    // - col_idx values in [0, n)
    // - probs values in [0, 1]
    // - If strict: each row sums to 1.0
}
```

---

## 6. Policy Evaluation Operator

**File**: `include/helios/policy_eval_op.h`, `src/ops/policy_eval_op.cc`

This implements the Bellman operator for policy evaluation.

### The Bellman Operator

```
F_i(x) = r_i + β · Σ_j P_{ij} · x_j
```

In words: the value at state i equals the immediate reward plus the discounted expected future value.

### Implementation

```cpp
class PolicyEvalOp final : public Operator {
public:
    explicit PolicyEvalOp(const MDP* mdp) : mdp_(mdp) {}

    index_t n() const noexcept override {
        return mdp_->n;
    }

    real_t apply_i(index_t i, const real_t* x) const override {
        const auto& mdp = *mdp_;
        const index_t start = mdp.row_ptr[i];
        const index_t end = mdp.row_ptr[i + 1];

        real_t dot = 0.0;
        for (index_t idx = start; idx < end; ++idx) {
            dot += mdp.probs[idx] * x[mdp.col_idx[idx]];
        }

        return mdp.rewards[i] + mdp.beta * dot;
    }
};
```

### Async Version

For concurrent access, we use `std::atomic_ref` to ensure atomic reads:

```cpp
real_t PolicyEvalOp::apply_i_async(index_t i, const real_t* x) const {
    // Same as apply_i, but reads x[j] atomically:
    const real_t xj = atomic_ref<real_t>(const_cast<real_t&>(x[j]))
                      .load(memory_order_relaxed);
    // ...
}
```

This prevents **torn reads** where we see a partially-written value.

---

## 7. The Scheduler Interface

**File**: `include/helios/scheduler.h`

The Scheduler determines which coordinate each thread should update next.

### Interface Definition

```cpp
class Scheduler {
public:
    virtual ~Scheduler() = default;

    // Initialize for n coordinates with num_threads workers
    virtual void init(index_t n, size_t num_threads) = 0;

    // Return the next coordinate index for thread tid
    virtual index_t next(size_t tid) = 0;

    // Optional: receive notification of residual update
    virtual void notify(size_t tid, index_t i, real_t residual) {}

    // Rebuild priority structure from residuals (for priority schedulers)
    virtual void rebuild(const std::vector<real_t>& residuals) {}

    // Does this scheduler benefit from rebuild()?
    virtual bool supports_rebuild() const noexcept { return false; }

    // Name for logging
    virtual std::string_view name() const noexcept { return "scheduler"; }
};
```

### Why Do We Need Schedulers?

Different scheduling strategies have different tradeoffs:

| Strategy | Pros | Cons |
|----------|------|------|
| Sequential | Simple, cache-friendly | Slow convergence |
| Random | Breaks correlations | Less cache-friendly |
| Priority | Fast convergence | Overhead to maintain |

---

## 8. All Implemented Schedulers

We have **5 schedulers**, each with different characteristics.

### 8.1 StaticBlocksScheduler

**File**: `src/schedulers/static_blocks.cc`

The simplest scheduler: divides [0, n) into contiguous blocks, one per thread.

```
n=10, 3 threads:
  Thread 0: [0, 4)  → 0, 1, 2, 3, 0, 1, 2, 3, ...
  Thread 1: [4, 7)  → 4, 5, 6, 4, 5, 6, ...
  Thread 2: [7, 10) → 7, 8, 9, 7, 8, 9, ...
```

**Characteristics**:
- Lock-free (no shared state between threads)
- Excellent cache locality (sequential access)
- No priority (fixed order)

### 8.2 ShuffledBlocksScheduler

**File**: `src/schedulers/shuffled_blocks.cc`

Each thread owns a contiguous block but iterates in **shuffled order**. Reshuffles after each complete pass (epoch).

```
n=10, 2 threads:
  Thread 0 block: [0, 5)
    Epoch 1: [3, 0, 4, 1, 2]  (shuffled)
    Epoch 2: [1, 4, 0, 2, 3]  (reshuffled)
```

**Characteristics**:
- Lock-free
- Good cache locality (indices from contiguous block)
- Breaks access pattern correlations
- Deterministic (seeded RNG)

### 8.3 TopKGSScheduler

**File**: `src/schedulers/topk_gs.cc`

Approximates **Gauss-Southwell** (greedy) coordinate selection by maintaining a "hot set" of K coordinates with largest residuals.

**Algorithm**:
```
REBUILD (called periodically):
  1. Compute residuals for all coordinates
  2. Select Top-K using nth_element (O(n) average)
  3. Optionally sort hot set by residual

NEXT(tid):
  1. k = atomic_fetch_add(hot_cursor, 1)
  2. If k < K: return hot[k]          // Priority phase
  3. Else: return fallback.next(tid)  // Coverage phase (shuffled blocks)
```

**Parameters**:
```cpp
struct TopKGSParams {
    index_t K = 0;        // Hot set size (0 = auto)
    bool sort_hot = false; // Sort by descending residual
    uint64_t seed = 0;     // RNG seed
};
```

**Characteristics**:
- Lock-free (atomic cursor)
- Prioritizes high-residual coordinates
- Falls back to shuffled blocks for coverage
- Rebuild cost: O(n)

### 8.4 CATopKGSScheduler (Conflict-Aware)

**File**: `src/schedulers/ca_topk_gs.cc`

Extends TopKGS by distributing hot indices into **G conflict groups** based on memory locality. Reduces contention when many threads are active.

**Key Idea**: Indices that are close in memory (same cache block) go to the same group. Different threads pick from different groups, reducing false sharing.

**Conflict Key Function**:
```cpp
index_t key(index_t i) const {
    return (i / block_size) % G;  // Cache block proxy
}
```

**Algorithm**:
```
REBUILD:
  1. Select Top-K indices
  2. Assign each to group g = key(i)
  3. Optionally sort within groups
  4. Shuffle group visiting order

NEXT(tid):
  1. Round-robin across groups:
       t = atomic_fetch_add(group_rr_cursor, 1)
       g = group_order[t % G]
       k = atomic_fetch_add(group_cursor[g], 1)
       if k < size(groups[g]): return groups[g][k]
  2. If all exhausted: fallback to shuffled blocks
```

**Parameters**:
```cpp
struct CATopKGSParams {
    index_t K = 0;           // Hot set size
    size_t G = 0;            // Number of groups (0 = 4*threads)
    index_t block_size = 256; // Cache block proxy
    bool sort_within_group = true;
    uint64_t seed = 0;
};
```

**Characteristics**:
- Lock-free (G independent cursors)
- Reduced contention vs TopKGS
- Cache-friendly grouping
- Better scaling with many threads

### 8.5 ResidualBucketsScheduler

**File**: `src/schedulers/residual_buckets.cc`

Groups coordinates into **logarithmic buckets** by residual magnitude. Dispatches from highest bucket first.

**Bucket Assignment**:
```
bucket(i) = clamp(floor(log2(residual_i / base)), 0, B-1)
```

Each bucket represents a 2× range of residual values.

**Algorithm**:
```
REBUILD:
  Pass 1: Count indices per bucket
  Pass 2: Place indices into buckets (sorted by bucket, stable by index within)

NEXT(tid):
  1. Search buckets from highest to lowest
  2. Atomically claim next index from first non-empty bucket
  3. Fall back to round-robin when all exhausted
```

**Parameters**:
```cpp
struct Params {
    uint32_t num_buckets = 32;
    real_t base = 1e-12;
    bool fallback_round_robin = true;
};
```

**Characteristics**:
- Lock-free (per-bucket atomic cursors)
- Continuous prioritization
- O(n) rebuild
- Natural priority ordering

---

## 9. The Runtime

**File**: `include/helios/runtime.h`, `src/runtime.cc`

The Runtime orchestrates execution, handling iteration, monitoring, and convergence checking.

### Configuration

```cpp
struct RuntimeConfig {
    size_t num_threads = 1;        // Worker threads for async
    real_t alpha = 1.0;            // Relaxation parameter
    real_t eps = 1e-6;             // Convergence tolerance
    double max_seconds = 0.0;      // Timeout (0 = no limit)
    uint64_t max_updates = 0;      // Update limit (0 = no limit)
    size_t monitor_interval_ms = 100;  // Residual check interval
    size_t rebuild_interval_ms = 500;  // Priority scheduler rebuild interval
    int residual_scan_stride = 1;  // Check every N-th coordinate
    bool verify_invariants = true; // Debug validation
    bool record_trace = true;      // Record residual history
    Mode mode = Mode::Jacobi;      // Execution mode
};
```

### Results

```cpp
struct RunResult {
    bool converged = false;           // Did we reach eps?
    real_t final_residual_inf = 0.0;  // Final ||F(x) - x||_∞
    double wall_time_sec = 0.0;       // Total wall time
    uint64_t total_updates = 0;       // Number of coordinate updates
    double updates_per_sec = 0.0;     // Throughput
    vector<ResidualSample> trace;     // Residual history [(time, residual), ...]
};
```

### Main Entry Point

```cpp
class Runtime {
public:
    RunResult run(const Operator& op,
                  Scheduler& scheduler,
                  real_t* x,
                  const RuntimeConfig& config);

    static real_t residual_inf(const Operator& op,
                               const real_t* x,
                               int stride = 1);
};
```

---

## 10. Execution Modes

### 10.1 Jacobi Mode (Synchronous)

All coordinates see the same snapshot when computing updates.

```
x_next[i] = (1 - α) · x_curr[i] + α · F_i(x_curr)   for all i
SWAP(x_curr, x_next)
```

**Implementation**: Double-buffering
```cpp
vector<real_t> bufA(n), bufB(n);
real_t* x_curr = bufA.data();
real_t* x_next = bufB.data();

while (!converged) {
    for (i = 0; i < n; ++i) {
        x_next[i] = (1 - alpha) * x_curr[i] + alpha * op.apply_i(i, x_curr);
    }
    swap(x_curr, x_next);
    // Check convergence periodically
}
```

**Characteristics**:
- Deterministic, reproducible
- Simple to implement correctly
- May converge slower than Gauss-Seidel

### 10.2 Gauss-Seidel Mode (Sequential In-Place)

Updates use the most recent values, including earlier updates from the same sweep.

```cpp
for (i = 0; i < n; ++i) {
    x[i] = (1 - alpha) * x[i] + alpha * op.apply_i(i, x);
    // x already contains updates for j < i
}
```

**Characteristics**:
- In-place (no extra memory)
- Often converges faster than Jacobi
- Not parallelizable in standard form

### 10.3 Async Mode (Parallel)

Multiple worker threads update concurrently.

```
┌──────────────┐   ┌──────────────┐       ┌──────────────┐
│  Worker 0    │   │  Worker 1    │  ...  │  Worker T-1  │
│  while !stop │   │  while !stop │       │  while !stop │
│    i = next()│   │    i = next()│       │    i = next()│
│    update(i) │   │    update(i) │       │    update(i) │
└──────┬───────┘   └──────┬───────┘       └──────┬───────┘
       │                  │                      │
       └──────────────────┼──────────────────────┘
                          ▼
                 ┌────────────────┐
                 │   Shared x[]   │  ← concurrent reads/writes
                 └────────────────┘
                          ▲
                 ┌────────────────┐
                 │ Monitor Thread │  ← checks convergence, rebuilds scheduler
                 └────────────────┘
```

**Worker Thread Loop**:
```cpp
while (!stop.load(memory_order_relaxed)) {
    index_t i = sched.next(tid);
    if (i >= n) { yield(); continue; }

    real_t fi = op.apply_i_async(i, x);  // May read stale values

    atomic_ref<real_t> xi_ref(x[i]);
    real_t xi = xi_ref.load(memory_order_relaxed);
    real_t xnew = (1 - alpha) * xi + alpha * fi;
    xi_ref.store(xnew, memory_order_relaxed);

    total_updates.fetch_add(1, memory_order_relaxed);
}
```

**Monitor Thread**:
```cpp
while (!stop.load(memory_order_relaxed)) {
    sleep_for(monitor_interval_ms);

    // Scan residuals
    real_t mx = 0.0;
    for (i = 0; i < n; ++i) {
        mx = max(mx, op.residual_i_async(i, x));
    }

    // Check convergence
    if (mx <= eps) {
        stop.store(true, memory_order_relaxed);
        break;
    }

    // Rebuild priority scheduler if due
    if (sched.supports_rebuild() && time_since_rebuild >= rebuild_interval_ms) {
        sched.rebuild(residuals);
    }
}
```

**Why Relaxed Memory Ordering is Safe**:
- The algorithm tolerates stale reads (contraction property guarantees convergence anyway)
- Atomicity prevents torn writes
- Updates are eventually visible

---

## 11. MDP Generators

**File**: `include/helios/mdp_generators.h`, `src/mdp_generators.cc`

We provide 7 MDP generators for testing and benchmarking.

### 11.1 Ring MDP

Simple circular structure:
```
State i → self (p_self) or (i+1)%n (1-p_self)
```

**Analytical solution**: V* = reward / (1 - β) for all states (by symmetry)

```cpp
MDP build_ring_mdp(index_t n, real_t beta,
                   real_t p_self = 0.5,
                   real_t reward_val = 1.0);
```

### 11.2 Grid MDP

2D grid with local transitions:
```
States arranged in rows×cols grid
Each state can move to 4 neighbors or stay
```

Tests spatial locality and cache-friendly schedulers.

```cpp
MDP build_grid_mdp(index_t rows, index_t cols, real_t beta,
                   real_t p_stay = 0.2, real_t base_reward = 1.0,
                   real_t reward_gradient = 0.0);
```

### 11.3 Metastable MDP

Two clusters with rare inter-cluster transitions:
```
Cluster A: states 0..n/2-1
Cluster B: states n/2..n-1
High intra-cluster probability (0.95)
Low inter-cluster probability (0.05)
```

This is a **hard case**: value propagates quickly within clusters but slowly between them.

```cpp
MDP build_metastable_mdp(index_t n, real_t beta,
                         real_t p_intra = 0.95, real_t p_bridge = 0.05,
                         real_t reward_A = 1.0, real_t reward_B = 2.0,
                         uint64_t seed = 42);
```

### 11.4 Star MDP

Hub-and-spoke structure:
```
State 0: hub (transitions uniformly to all leaves)
States 1..n-1: leaves (high probability to hub, low to other leaves)
```

Creates skewed residual patterns - the hub has many dependencies.

```cpp
MDP build_star_mdp(index_t n, real_t beta,
                   real_t p_to_hub = 0.8,
                   real_t hub_reward = 1.0, real_t leaf_reward = 0.5);
```

### 11.5 Chain MDP

Linear chain with biased drift:
```
State i can go to: i-1 (p_left), i (p_stay), i+1 (p_right)
Boundary: reflecting or absorbing
```

```cpp
MDP build_chain_mdp(index_t n, real_t beta,
                    real_t p_left = 0.25, real_t p_stay = 0.5,
                    real_t p_right = 0.25,
                    int reward_fn = 0, bool reflecting = true);
```

### 11.6 Random Sparse MDP

Random graph structure:
```
Each state has exactly nnz_per_row random outgoing transitions
Probabilities and rewards sampled randomly
```

```cpp
MDP build_random_sparse_mdp(index_t n, index_t nnz_per_row, real_t beta,
                            real_t max_reward = 1.0, uint64_t seed = 42);
```

### 11.7 Multi-Cluster MDP

Generalization of metastable with k clusters:
```
n states divided into k clusters
Dense intra-cluster, sparse inter-cluster
```

```cpp
MDP build_multi_cluster_mdp(index_t n, index_t k, real_t beta,
                            real_t p_intra = 0.9,
                            const std::vector<real_t>& rewards = {},
                            uint64_t seed = 42);
```

---

## 12. How Everything Fits Together

### Architecture Diagram

```
                    ┌─────────────────┐
                    │  RuntimeConfig  │
                    │  - mode         │
                    │  - eps, alpha   │
                    │  - num_threads  │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│   Operator   │      │   Runtime   │      │  Scheduler  │
│  - n()       │◄────►│  - run()    │◄────►│  - next()   │
│  - apply_i() │      │             │      │  - init()   │
│  - residual_i│      └──────┬──────┘      │  - rebuild()│
└──────────────┘             │             └─────────────┘
       ▲                     │
       │              ┌──────┴──────┐
       │              │  RunResult  │
┌──────┴───────┐      │  - converged│
│ PolicyEvalOp │      │  - trace    │
│  (uses MDP)  │      │  - metrics  │
└──────────────┘      └─────────────┘
```

### Typical Usage Flow

```cpp
// 1. Create an MDP
MDP mdp = build_ring_mdp(1000, 0.9);
mdp.validate();

// 2. Create the operator
PolicyEvalOp op(&mdp);

// 3. Initialize solution vector
std::vector<real_t> x(mdp.n, 0.0);

// 4. Configure runtime
RuntimeConfig cfg;
cfg.mode = Mode::Async;
cfg.num_threads = 4;
cfg.eps = 1e-6;
cfg.max_seconds = 30.0;

// 5. Create scheduler
CATopKGSScheduler sched;

// 6. Run!
Runtime rt;
RunResult result = rt.run(op, sched, x.data(), cfg);

// 7. Check results
if (result.converged) {
    std::cout << "Converged in " << result.wall_time_sec << " sec\n";
    std::cout << "Final residual: " << result.final_residual_inf << "\n";
}
```

### Data Flow in Async Mode

```
1. Runtime::run() called
   ↓
2. Scheduler initialized: sched.init(n, num_threads)
   ↓
3. Monitor thread spawned
   ↓
4. Worker threads spawned (T threads)
   ↓
5. Workers loop:
   ├─→ sched.next(tid) returns coordinate i
   ├─→ op.apply_i_async(i, x) computes F_i(x)
   ├─→ x[i] updated atomically
   └─→ total_updates incremented
   ↓
6. Monitor periodically:
   ├─→ Scans residuals: ||F(x) - x||_∞
   ├─→ Records trace point
   ├─→ Calls sched.rebuild(residuals) if due
   └─→ Sets stop=true if converged
   ↓
7. Threads join, results collected
   ↓
8. RunResult returned
```

---

## 13. Project Status

### Completed (Phase 1)

| Component | Status | Description |
|-----------|--------|-------------|
| `types.h` | ✅ | Core types, aligned memory |
| `operator.h` | ✅ | Abstract operator interface |
| `scheduler.h` | ✅ | Abstract scheduler interface |
| `runtime.h` | ✅ | Config, result, mode enum |
| `mdp.h` | ✅ | MDP struct with CSR storage |
| `policy_eval_op` | ✅ | Bellman operator |
| Jacobi mode | ✅ | Synchronous iteration |
| Gauss-Seidel mode | ✅ | Sequential in-place |
| Async mode | ✅ | Multi-threaded parallel |
| StaticBlocksScheduler | ✅ | Contiguous blocks |
| ShuffledBlocksScheduler | ✅ | Randomized blocks |
| TopKGSScheduler | ✅ | Top-K Gauss-Southwell |
| CATopKGSScheduler | ✅ | Conflict-aware Top-K |
| ResidualBucketsScheduler | ✅ | Logarithmic bucketing |
| MDP generators | ✅ | 7 different structures |
| Convergence tests | ✅ | 21 complex MDP tests |

### File Structure

```
helios/
├── include/helios/
│   ├── types.h              # Core types
│   ├── operator.h           # Operator interface
│   ├── scheduler.h          # Scheduler interface
│   ├── runtime.h            # Runtime config/result
│   ├── mdp.h                # MDP structure
│   ├── policy_eval_op.h     # Bellman operator
│   ├── mdp_generators.h     # MDP generators
│   └── schedulers/
│       ├── static_blocks.h
│       ├── shuffled_blocks.h
│       ├── topk_gs.h
│       ├── ca_topk_gs.h
│       └── residual_buckets.h
├── src/
│   ├── runtime.cc           # All three modes
│   ├── mdp_generators.cc
│   ├── ops/
│   │   └── policy_eval_op.cc
│   └── schedulers/
│       ├── static_blocks.cc
│       ├── shuffled_blocks.cc
│       ├── topk_gs.cc
│       ├── ca_topk_gs.cc
│       └── residual_buckets.cc
└── tests/
    ├── test_runtime_smoke.cc    # Basic convergence tests
    ├── test_schedulers.cc       # Scheduler unit tests
    └── test_complex_mdps.cc     # 21 convergence tests
```

### Build Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/helios_tests
```

---

## Summary

Helios implements a complete fixed-point iteration engine with:

1. **Clean abstractions**: Operator, Scheduler, Runtime separation
2. **Multiple execution modes**: Jacobi, Gauss-Seidel, Async
3. **5 scheduler strategies**: From simple to sophisticated priority-based
4. **Lock-free async execution**: Using C++20 `atomic_ref`
5. **Comprehensive testing**: Multiple MDP structures and convergence verification

The key insight is that **mathematical convergence guarantees** (contraction property) allow us to use **minimal synchronization** (relaxed atomics), maximizing throughput while maintaining correctness.
