# Phase 3: Schedule IR + Planner + Executor

## Overview

Phase 3 transforms Helios from a scheduler-driven engine into a **compiler-planned execution engine**. The key insight: instead of making per-coordinate scheduling decisions at runtime (which requires atomic operations and contention), we **compile** an execution plan ahead of time and execute it with minimal synchronization.

**"Compiler" = schedule compiler** (not C++ compiler). The input is operator sparsity (CSR) + optional residual snapshot + hardware knobs. The output is a compiled execution plan (Schedule IR) = phases + per-thread worklists.

```
                    ┌─────────────────────────────────────┐
                    │            Planner                  │
                    │  (Static / Colored / Priority)      │
                    │                                     │
                    │  Input:                             │
                    │    - Operator (sparsity / n)        │
                    │    - x_snapshot (optional residuals) │
                    │    - PlannerConfig (blk, C, K, T)   │
                    │                                     │
                    │  Output:                            │
                    │    - EpochPlan (compiled schedule)   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │          Schedule IR                 │
                    │  EpochPlan = vector<Phase>           │
                    │  Phase = per-thread worklists        │
                    │  Task = {kind, begin, end, weight}   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      Runtime::run_plan()            │
                    │  Plan Executor                      │
                    │                                     │
                    │  - Iterate epochs of the plan       │
                    │  - Launch T threads per phase       │
                    │  - Periodic residual checks         │
                    │  - Profiling counters               │
                    └─────────────────────────────────────┘
```

---

## 1. Dependency Graph

For any operator F where F_i(x) reads coordinates x_j, the **dependency graph** is:

```
j -> i   if F_i reads x_j
```

For the Bellman operator with CSR-stored transition matrix P:

```
F_i(x) = r_i + β · Σ_j P_ij · x_j
```

The dependency is: **j -> i if P_ij > 0**, i.e., CSR row i defines the neighborhood N(i).

```
┌─────────────────────────────────────────────────┐
│  Dependency Graph (CSR Row i)                   │
│                                                 │
│  State i depends on:                            │
│    { j : P_ij > 0 }  =  { col_idx[k] :         │
│                            k ∈ [row_ptr[i],      │
│                                 row_ptr[i+1]) }  │
│                                                 │
│  If two threads update i and j simultaneously   │
│  where j ∈ N(i), they may have a read-write     │
│  conflict (false sharing or stale reads).        │
│                                                 │
│  Coloring ensures threads in the same phase     │
│  touch disjoint cache blocks.                   │
└─────────────────────────────────────────────────┘
```

---

## 2. Schedule IR: Task / Phase / EpochPlan

**File:** `include/helios/plan.h`

### Task

A `Task` is the atomic unit of work in a compiled schedule:

```cpp
enum class TaskKind : uint8_t {
    ONE,    // Single coordinate update (begin only)
    BLOCK   // Block of contiguous coordinates [begin, end)
};

struct Task {
    TaskKind kind;
    index_t  begin;           // First coordinate
    index_t  end;             // Past-the-end (end = begin+1 for ONE)
    double   weight;          // Estimated cost (e.g., sum of nnz in rows)
    uint32_t conflict_key;    // Cache-block proxy for conflict grouping
};
```

- **BLOCK tasks** are the common case: contiguous ranges like `[0, 256)`, `[256, 512)`, etc.
- **weight** is populated by the cost model from MDP sparsity (sum of non-zeros per row).
- **conflict_key** is the block ID, used by ColoredPlanner for cache-block coloring.

### Phase

A `Phase` is one parallel execution step:

```cpp
struct Phase {
    vector<vector<Task>> worklist;   // worklist[tid] = tasks for thread tid
    bool barrier_after;              // sync all threads after this phase
};
```

Each thread processes its worklist independently. If `barrier_after` is true, all threads synchronize before the next phase begins.

### EpochPlan

An `EpochPlan` is a complete plan for one iteration cycle:

```cpp
struct EpochPlan {
    vector<Phase> phases;    // Ordered sequence of phases

    // Metadata
    index_t n;               // Problem dimension
    size_t  threads;         // Number of threads
    index_t blk;             // Block size
    index_t colors;          // Number of colors (0 = no coloring)
    index_t K;               // Top-K size (0 = no priority)
    uint64_t seed;           // Random seed
    string built_from;       // Planner name
};
```

**Key methods:**
- `total_updates()` - Sum of all coordinate updates across all phases
- `max_thread_updates()` per phase - Identifies the bottleneck thread
- `summary()` - Human-readable plan description

### ScheduleProgram

Wraps an `EpochPlan` for the runtime. Currently supports repeating one plan:

```cpp
struct ScheduleProgram {
    EpochPlan epoch_plan;
    const EpochPlan& current_plan() const;
};
```

---

## 3. Planner Interface

**File:** `include/helios/planner.h`

### PlannerConfig

```cpp
struct PlannerConfig {
    size_t   threads;              // T: worker threads
    index_t  blk;                  // Block size for grouping coordinates
    index_t  colors;               // C: number of colors (0 = auto = T)
    index_t  K;                    // Top-K hot set size (0 = auto)
    bool     hot_phase_enabled;    // Enable priority/hot phase
    bool     barrier_between_colors; // Barrier after each color phase
    uint64_t seed;                 // Random seed
};
```

### Abstract Planner

```cpp
class Planner {
    virtual EpochPlan build(const Operator& op,
                            const real_t* x_snapshot,
                            const PlannerConfig& cfg) = 0;
    virtual string_view name() const noexcept = 0;
};
```

A Planner reads operator metadata (dimension, sparsity) and an optional state snapshot (for residual-based priority), then produces an `EpochPlan`. Planners never touch runtime internals.

---

## 4. StaticPlanner

**The simplest planner:** partition coordinates into blocks, assign to threads round-robin.

### Algorithm

```
Input:  n coordinates, block size blk, T threads
Output: Single Phase with T worklists

1. Compute num_blocks = ceil(n / blk)
2. For each block b = 0, ..., num_blocks-1:
     begin = b * blk
     end   = min(begin + blk, n)
     tid   = b % T           (round-robin assignment)
     Add Task(BLOCK, begin, end) to worklist[tid]
3. Return EpochPlan with one Phase, no barriers
```

### Example (n=32, blk=8, T=2)

```
Phase 0:
  Thread 0: [0,8)  [16,24)
  Thread 1: [8,16) [24,32)
```

### Properties
- **No barriers:** All threads run independently
- **Cache-friendly:** Contiguous blocks map to contiguous memory
- **Balanced:** Round-robin ensures roughly equal work per thread
- **Equivalent to:** StaticBlocksScheduler (Phase 2)

---

## 5. ColoredPlanner (Cache-Block Proxy Coloring)

**Key idea:** Assign blocks to colors based on their block ID, then create separate phases per color. Within each color phase, blocks are guaranteed to access disjoint cache regions, eliminating false sharing.

### Conflict Proxy

```
block_id(i)    = floor(i / blk)
color(block_id) = block_id % C
```

Where C is the number of colors (default: T threads).

### Algorithm

```
Input:  n, blk, C colors, T threads
Output: C Phases, one per color

1. Compute num_blocks = ceil(n / blk)
2. For each color c = 0, ..., C-1:
     Create Phase c with T empty worklists
3. For each block b = 0, ..., num_blocks-1:
     color = b % C
     begin = b * blk
     end   = min(begin + blk, n)
     tid   = (count of blocks assigned to this color so far) % T
     Add Task(BLOCK, begin, end) to Phase[color].worklist[tid]
4. Set barrier_after = barrier_between_colors on each Phase
```

### Example (n=64, blk=8, C=2, T=2)

```
Phase 0 (color 0):            Phase 1 (color 1):
  T0: [0,8) [32,40)             T0: [8,16) [40,48)
  T1: [16,24) [48,56)           T1: [24,32) [56,64)
```

Blocks in Phase 0 have even block IDs (0, 2, 4, 6), blocks in Phase 1 have odd (1, 3, 5, 7). Since blocks of the same color are spaced apart by `C * blk` bytes in memory, threads in the same phase never touch adjacent cache lines.

### Why This Helps

In async mode without coloring, two threads might simultaneously update coordinates 255 and 256 — which sit on the same or adjacent cache lines. The CPU's cache coherence protocol bounces the cache line between cores (false sharing), wasting bandwidth.

With C=T coloring, blocks active in the same phase are guaranteed to be at least `T * blk` coordinates apart in memory, virtually eliminating false sharing.

---

## 6. PriorityPlanner (Top-K Compiled Hot Phase)

**Key idea:** Use a residual snapshot to identify the highest-error coordinates, compile them into a "hot phase" that runs first, then follow with a full coverage phase.

### Algorithm

```
Input:  n, blk, C, K, T, x_snapshot
Output: Hot phases + Coverage phases

BUILD-TIME RESIDUAL SNAPSHOT:
1. For each block b:
     block_score[b] = Σ |F_i(x_snapshot) - x_snapshot[i]|  for i ∈ block b

SELECT HOT BLOCKS:
2. Sort blocks by descending block_score
3. Mark top blocks as "hot" until they cover ≥ K indices

EMIT HOT PHASES (optionally colored):
4. For each color c = 0..C-1:
     Create hot Phase containing only hot blocks whose (block_id % C == c)
     Assign to threads round-robin within each color

EMIT COVERAGE PHASES (all blocks, colored):
5. Same as ColoredPlanner over all blocks
```

### Example (n=64, blk=8, K=16, C=2, T=2)

Suppose blocks 3 and 5 have the highest residual scores:

```
Phase 0 (HOT, color 1): block 3 → [24,32)
Phase 1 (HOT, color 1): block 5 → [40,48)
Phase 2 (COVERAGE, color 0): blocks 0,2,4,6
Phase 3 (COVERAGE, color 1): blocks 1,3,5,7
```

The hot blocks get updated first (when their residual contribution matters most), then the full coverage sweep ensures every coordinate is visited.

### Why This Helps

On metastable MDPs (clusters with rare bridges), residuals concentrate at the bridge states. PriorityPlanner focuses early computation on these bottleneck coordinates, then sweeps the rest for convergence. This is the compiled equivalent of Top-K Gauss-Southwell scheduling.

---

## 7. Plan Executor: `Runtime::run_plan()`

**File:** `src/runtime.cc`

### Architecture

```
run_plan(op, plan, x, cfg):
  ┌─────────────────────────────────────────┐
  │  Loop until converged / timeout:        │
  │                                         │
  │    For each Phase in plan.phases:       │
  │      ┌──────────────────────────────┐   │
  │      │  Launch T threads:           │   │
  │      │    Thread tid processes      │   │
  │      │    phase.worklist[tid]       │   │
  │      │    sequentially              │   │
  │      └──────────────────────────────┘   │
  │      If phase.barrier_after: join       │
  │                                         │
  │    Periodically:                        │
  │      Compute ||F(x) - x||_∞            │
  │      Record residual sample             │
  │      Stop if residual ≤ eps             │
  └─────────────────────────────────────────┘
```

### Single-Thread Fast Path

When T=1, the executor skips thread creation entirely and runs in-place:

```cpp
for (auto& task : phase.worklist[0]) {
    for (index_t i = task.begin; i < task.end; ++i) {
        x[i] = (1-α)x[i] + α·F_i(x);    // Gauss-Seidel style
    }
}
```

### Multi-Thread Mode

With T > 1, each thread updates its assigned coordinates using `atomic_ref` for weak-atomic writes (same as Phase 2 async mode):

```cpp
atomic_ref<real_t> xi_ref(x[i]);
real_t fi = op.apply_i_async(i, x);
real_t xi = xi_ref.load(memory_order_relaxed);
xi_ref.store((1-α)*xi + α*fi, memory_order_relaxed);
```

### Profiling Integration

`run_plan` populates `ProfilingResult` with:
- Per-thread `updates_completed` and `time_in_update_ns`
- Global `time_in_residual_scan_ns` and `num_residual_scans`
- Derived `avg_update_cost_ns`

---

## 8. Profiling Counters

**File:** `include/helios/profiling.h`

```cpp
struct ThreadCounters {
    uint64_t updates_completed;     // How many coordinates this thread updated
    uint64_t time_in_update_ns;     // Aggregate wall time in update loops
};

struct ProfilingResult {
    vector<ThreadCounters> per_thread;
    uint64_t time_in_residual_scan_ns;
    uint64_t num_residual_scans;
    uint64_t total_updates;

    double avg_update_cost_ns();     // Derived: time / updates
    double avg_residual_scan_ns();   // Derived: scan_time / scans
};
```

These counters are embedded in `RunResult` and populated by `run_plan`. They feed into the cost model for auto-tuning.

---

## 9. Cost Model

**File:** `include/helios/cost_model.h`, `src/cost_model.cc`

### Task Cost

For a BLOCK task over coordinates [begin, end):

```
c(task) = Σ_{i=begin}^{end-1} nnz(row i)
```

where `nnz(row i) = row_ptr[i+1] - row_ptr[i]` counts the number of non-zero transitions from state i. This proxies the actual computation cost (one multiply-add per non-zero).

If measured `avg_update_cost_ns` is available (from profiling), the cost becomes:

```
c(task) = avg_update_cost_ns × (end - begin)
```

### Plan Cost Estimate

```
est(plan) = max_tid Σ_{task ∈ worklist[tid]} c(task)    [bottleneck]
          + λ × num_phases                               [phase overhead]
          + penalty × num_barriers                       [barrier cost]
```

- **Bottleneck cost:** The slowest thread determines wall time
- **Phase penalty (λ):** Each phase has thread launch/join overhead
- **Barrier penalty:** Synchronization points add latency

### CostEstimate

```cpp
struct CostEstimate {
    double bottleneck_cost;    // Max thread cost
    double total_cost;         // Sum of all thread costs
    double phase_penalty;      // λ × num_phases
    double barrier_penalty;    // Penalty for barriers
    double estimated_cost;     // bottleneck + penalties
};
```

### Weight Population

`populate_task_weights(plan, mdp)` fills in `task.weight` for every task in a plan using the MDP's CSR structure.

---

## 10. Auto-Tuning

**File:** `include/helios/autotune.h`, `src/autotune.cc`

### Strategy

```
┌───────────────────────────────────────────────────────┐
│                    Autotune Pipeline                  │
│                                                       │
│  1. Generate candidate configurations                │
│     blk ∈ {64, 128, 256, 512, 1024}                 │
│     C   ∈ {T, 2T, 4T}                               │
│     K   ∈ {0.5%, 1%, 2%, 5%} of n                   │
│                                                       │
│  2. For each candidate:                              │
│     - Build plan (Static / Colored / Priority)        │
│     - Populate task weights from MDP                  │
│     - Compute est(plan) via cost model               │
│                                                       │
│  3. Sort by est(plan), take top M candidates          │
│                                                       │
│  4. Pilot run each top-M candidate:                  │
│     - Restore x to snapshot                          │
│     - Run for pilot_seconds (e.g., 0.5s)             │
│     - Measure updates/sec and residual drop           │
│                                                       │
│  5. Select best by residual drop rate                 │
│                                                       │
│  6. Return best plan + configuration                  │
└───────────────────────────────────────────────────────┘
```

### Candidate Generation

The autotuner explores three planner types with a grid of parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `blk` | 64, 128, 256, 512, 1024 | Block size |
| `colors` | T, 2T, 4T | Color count (× thread count) |
| `K` | 0.5%, 1%, 2%, 5% of n | Hot set fraction |
| `rebuild_ms` | 50, 100, 200 | Rebuild interval |

### Selection Criteria

1. **Phase 1 (cost model):** Rank all candidates by `est(plan)`. Take top M (default 3).
2. **Phase 2 (pilot runs):** Run each top-M candidate for a short time. Measure residual drop.
3. **Final selection:** Choose the candidate with the highest residual drop ratio (= most convergence progress per unit time).

### API

```cpp
AutotuneResult autotune(const Operator& op,
                         const MDP* mdp,
                         real_t* x,
                         const AutotuneConfig& atcfg);
```

Returns `AutotuneResult` containing:
- `best_plan` - The selected EpochPlan ready for `run_plan`
- `best` - The winning candidate's configuration
- `all_candidates` - Full list with costs and pilot results
- `autotune_time_sec` - Time spent auto-tuning

---

## 11. New Files Summary

### Headers (include/helios/)

| File | Purpose |
|------|---------|
| `plan.h` | Schedule IR: Task, Phase, EpochPlan, ScheduleProgram |
| `planner.h` | PlannerConfig, Planner interface, StaticPlanner, ColoredPlanner, PriorityPlanner |
| `profiling.h` | ThreadCounters, ProfilingResult |
| `cost_model.h` | CostModelConfig, CostEstimate, estimate_plan_cost(), populate_task_weights() |
| `autotune.h` | AutotuneConfig, AutotuneCandidate, AutotuneResult, autotune() |

### Sources (src/)

| File | Purpose |
|------|---------|
| `planners.cc` | StaticPlanner, ColoredPlanner, PriorityPlanner implementations |
| `cost_model.cc` | Cost estimation and weight population |
| `autotune.cc` | Autotune pipeline: candidate generation, cost model screening, pilot runs |

### Modified Files

| File | Changes |
|------|---------|
| `runtime.h` | Added `run_plan()` method, `ProfilingResult` in `RunResult` |
| `runtime.cc` | Implemented `run_plan()` with single-thread fast path and multi-thread mode |
| `CMakeLists.txt` | Added new source files |

### Tests (tests/)

| File | Tests |
|------|-------|
| `test_phase3.cc` | 11 tests covering all Phase 3 components |

---

## 12. Test Results

All 11 Phase 3 tests pass:

```
=== Phase 3 Tests ===

PASS: Plan IR basic
PASS: StaticPlanner ring convergence
PASS: ColoredPlanner ring convergence
PASS: ColoredPlanner multithread
PASS: Plan matches scheduler mode
PASS: Plan Grid MDP convergence
PASS: PriorityPlanner ring convergence
PASS: PriorityPlanner metastable convergence
PASS: Profiling counters
PASS: Cost model
PASS: Autotune

All tests passed.
```

### Key Validation Points

1. **Correctness:** `run_plan` reaches the same fixed point as scheduler-based modes (verified by `test_plan_matches_scheduler`)
2. **Coverage:** Plans cover all n coordinates each epoch (verified by `total_updates()`)
3. **Convergence:** All three planners converge on ring, grid, and metastable MDPs
4. **Profiling:** Counters are populated with non-zero values
5. **Cost model:** Produces stable non-zero estimates
6. **Autotune:** Selects a plan that converges correctly

---

## 13. Usage Example

### Direct Plan Execution

```cpp
#include "helios/planner.h"
#include "helios/runtime.h"

// Build MDP and operator
MDP mdp = build_ring_mdp(1024, 0.9);
PolicyEvalOp op(&mdp);
vector<real_t> x(1024, 0.0);

// Configure planner
PlannerConfig pcfg;
pcfg.threads = 4;
pcfg.blk = 128;
pcfg.colors = 4;

// Compile plan
ColoredPlanner planner;
EpochPlan plan = planner.build(op, x.data(), pcfg);

// Print plan summary
printf("%s", plan.summary().c_str());

// Execute
RuntimeConfig cfg;
cfg.num_threads = 4;
cfg.eps = 1e-6;
cfg.max_seconds = 60.0;

Runtime rt;
RunResult result = rt.run_plan(op, plan, x.data(), cfg);

printf("Converged: %d, time: %.3f sec\n", result.converged, result.wall_time_sec);
printf("%s", result.profiling.summary().c_str());
```

### With Auto-Tuning

```cpp
#include "helios/autotune.h"

AutotuneConfig atcfg;
atcfg.runtime_cfg.num_threads = 4;
atcfg.runtime_cfg.eps = 1e-6;
atcfg.pilot_seconds = 0.5;

AutotuneResult at = autotune(op, &mdp, x.data(), atcfg);
printf("%s", at.summary().c_str());

// Run with best plan
RunResult result = rt.run_plan(op, at.best_plan, x.data(), atcfg.runtime_cfg);
```

---

## 14. Architecture: Phase 2 vs Phase 3

### Phase 2: Online Scheduling

```
Worker Thread → sched.next(tid) → get index i → update x[i] → repeat
                     ↑
              Monitor rebuilds scheduler periodically
```

- **Decisions at runtime:** Scheduler decides which index to update next
- **Atomics everywhere:** `next()` uses atomic cursors, `sched.rebuild()` swaps epoch data
- **Flexible:** Can adapt instantly to residual changes

### Phase 3: Compiled Planning

```
Planner.build() → EpochPlan → run_plan() → repeat plan until converged
       ↑                            ↓
  x_snapshot                 Periodic residual check
```

- **Decisions at compile time:** Planner decides all assignments before execution
- **Minimal atomics:** Only `atomic_ref` for x[i] updates (same as Phase 2 async)
- **Predictable:** Deterministic work assignment, no contention on scheduler state
- **Cost model:** Can estimate and compare plans before running them

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Small problems (n < 1000) | Phase 2 GaussSeidel or Jacobi |
| Uniform structure | Phase 3 StaticPlanner or ColoredPlanner |
| Metastable / clustered | Phase 3 PriorityPlanner |
| Unknown structure | Phase 3 Autotune |
| Rapidly changing residuals | Phase 2 Online TopK/RBS |
| High thread count (8+) | Phase 3 ColoredPlanner (reduces false sharing) |
