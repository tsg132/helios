# Helios Async Runtime: Multithreading Deep Dive

This document explains how Helios implements asynchronous parallel fixed-point iteration, covering the thread architecture, synchronization primitives, memory ordering, and the theoretical foundations that make lock-free concurrent updates correct.

---

## Table of Contents

1. [Overview](#overview)
2. [Execution Modes Comparison](#execution-modes-comparison)
3. [Thread Architecture](#thread-architecture)
4. [The Worker Thread Loop](#the-worker-thread-loop)
5. [The Monitor Thread](#the-monitor-thread)
6. [Atomic Operations and Memory Ordering](#atomic-operations-and-memory-ordering)
7. [Why Relaxed Memory Ordering is Safe](#why-relaxed-memory-ordering-is-safe)
8. [Convergence Under Asynchrony](#convergence-under-asynchrony)
9. [Termination Protocol](#termination-protocol)
10. [Performance Characteristics](#performance-characteristics)

---

## Overview

Helios supports three execution modes for fixed-point iteration `x = F(x)`:

| Mode | Parallelism | Update Visibility | Use Case |
|------|-------------|-------------------|----------|
| **Jacobi** | Sequential | All updates visible after full sweep | Baseline, reproducible |
| **Gauss-Seidel** | Sequential | Updates visible immediately | Faster convergence |
| **Async** | Parallel | Updates visible eventually | Maximum throughput |

The async mode spawns `T` worker threads that concurrently update coordinates of the solution vector `x`, plus one monitor thread that checks for convergence.

---

## Execution Modes Comparison

### Jacobi (Synchronous)

```
Iteration k:
  ┌─────────────────────────────────────────────────┐
  │ FOR i = 0 to n-1:                               │
  │   x_next[i] = (1-α)·x_curr[i] + α·F_i(x_curr)   │
  │ SWAP(x_curr, x_next)                            │
  └─────────────────────────────────────────────────┘
```

All coordinates read from `x_curr` (iteration k), write to `x_next` (iteration k+1).

### Gauss-Seidel (Sequential In-Place)

```
Iteration k:
  ┌─────────────────────────────────────────────────┐
  │ FOR i = 0 to n-1:                               │
  │   x[i] = (1-α)·x[i] + α·F_i(x)  ← sees updates  │
  │                                   from j < i    │
  └─────────────────────────────────────────────────┘
```

Each coordinate immediately sees updates from earlier coordinates in the same sweep.

### Async (Parallel In-Place)

```
  ┌──────────────┐   ┌──────────────┐       ┌──────────────┐
  │  Worker 0    │   │  Worker 1    │  ...  │  Worker T-1  │
  │              │   │              │       │              │
  │  while !stop │   │  while !stop │       │  while !stop │
  │    i = next()│   │    i = next()│       │    i = next()│
  │    update(i) │   │    update(i) │       │    update(i) │
  └──────────────┘   └──────────────┘       └──────────────┘
          │                  │                      │
          └──────────────────┼──────────────────────┘
                             ▼
                    ┌────────────────┐
                    │   Shared x[]   │  ← concurrent reads/writes
                    └────────────────┘
                             ▲
                    ┌────────────────┐
                    │ Monitor Thread │  ← checks convergence
                    └────────────────┘
```

Workers concurrently read/write to shared `x[]`. Each update may see a mix of old and new values from other coordinates.

---

## Thread Architecture

The async runtime creates `T + 1` threads:

```cpp
// From runtime.cc:run_async_()

const int T = max(1, (int) cfg.num_threads);

// Shared state
atomic<bool> stop{false};
atomic<uint64_t> total_updates{0};
atomic<real_t> residual_inf_atomic{infinity()};

// 1. Monitor thread (1 thread)
thread monitor([&]() {
    // Periodically compute residual and check convergence
    ...
});

// 2. Worker threads (T threads)
vector<thread> workers;
for (int tid = 0; tid < T; ++tid) {
    workers.emplace_back([&, tid]() {
        // Main update loop
        ...
    });
}

// 3. Join all threads
for (auto& th : workers) th.join();
stop.store(true);
monitor.join();
```

### Thread Roles

| Thread | Count | Responsibility |
|--------|-------|----------------|
| **Workers** | T | Update coordinates via `x[i] = (1-α)·x[i] + α·F_i(x)` |
| **Monitor** | 1 | Compute `‖F(x) - x‖_∞`, check convergence, record trace |

---

## The Worker Thread Loop

Each worker thread runs the following loop:

```cpp
// From runtime.cc lines 496-539

workers.emplace_back([&, tid]() {

    while (!stop.load(memory_order_relaxed)) {

        // ═══════════════════════════════════════════════════════════════
        // Step 1: Check termination conditions
        // ═══════════════════════════════════════════════════════════════
        if (timed_out()) {
            stop.store(true, memory_order_relaxed);
            break;
        }

        if (cfg.max_updates != 0 &&
            total_updates.load(memory_order_relaxed) >= cfg.max_updates) {
            stop.store(true, memory_order_relaxed);
            break;
        }

        // ═══════════════════════════════════════════════════════════════
        // Step 2: Get next coordinate from scheduler
        // ═══════════════════════════════════════════════════════════════
        const index_t i = sched.next(tid);

        if (i >= n) {
            this_thread::yield();  // No work available, yield CPU
            continue;
        }

        // ═══════════════════════════════════════════════════════════════
        // Step 3: Compute F_i(x) using current (possibly stale) x
        // ═══════════════════════════════════════════════════════════════
        const real_t fi = op.apply_i_async(i, x);

        // ═══════════════════════════════════════════════════════════════
        // Step 4: Atomically update x[i]
        // ═══════════════════════════════════════════════════════════════
        atomic_ref<real_t> xi_ref(x[i]);

        const real_t xi = xi_ref.load(memory_order_relaxed);

        const real_t xnew = (real_t(1.0) - alpha) * xi + alpha * fi;

        xi_ref.store(xnew, memory_order_relaxed);

        // ═══════════════════════════════════════════════════════════════
        // Step 5: Increment global update counter
        // ═══════════════════════════════════════════════════════════════
        total_updates.fetch_add(1, memory_order_relaxed);
    }

});
```

### Key Points

1. **Scheduler-driven**: The scheduler determines which coordinate each thread updates
2. **Stale reads allowed**: `apply_i_async(i, x)` may read partially-updated values from other threads
3. **Atomic updates**: `atomic_ref<real_t>` ensures each `x[i]` is updated atomically (no torn writes)
4. **Relaxed ordering**: No synchronization barriers between updates

---

## The Monitor Thread

The monitor thread runs concurrently with workers:

```cpp
// From runtime.cc lines 410-491

const bool do_rebuild = sched.supports_rebuild() && cfg.rebuild_interval_ms > 0;

thread monitor([&]() {

    // Buffer for residuals (only allocated if scheduler needs rebuild)
    vector<real_t> residuals_buf;
    if (do_rebuild) residuals_buf.resize(n);

    auto t_last_rebuild = Clock::now();

    // Helper to scan residuals and optionally collect them for rebuild
    auto scan_residuals = [&](bool collect) -> real_t {
        real_t mx = 0.0;
        for (index_t i = 0; i < n; ++i) {
            const real_t r = op.residual_i_async(i, x);
            if (collect) residuals_buf[i] = r;
            if (i % stride == 0 && r > mx) mx = r;
        }
        return mx;
    };

    // ═══════════════════════════════════════════════════════════════════
    // Initial residual computation at t=0
    // ═══════════════════════════════════════════════════════════════════
    {
        const bool need_rebuild = do_rebuild;
        real_t mx = scan_residuals(need_rebuild);

        residual_inf_atomic.store(mx, memory_order_relaxed);

        if (cfg.record_trace) out.trace.push_back({0.0, mx});

        // Initial rebuild for priority schedulers
        if (need_rebuild) {
            sched.rebuild(residuals_buf);
            t_last_rebuild = Clock::now();
        }

        if (mx <= cfg.eps) stop.store(true, memory_order_relaxed);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Periodic monitoring loop
    // ═══════════════════════════════════════════════════════════════════
    while (!stop.load(memory_order_relaxed)) {

        // Check timeout
        if (timed_out()) {
            stop.store(true, memory_order_relaxed);
            break;
        }

        // Sleep for monitor_interval_ms
        if (interval > 0) {
            this_thread::sleep_for(chrono::milliseconds(interval));
        } else {
            this_thread::yield();
        }

        // Check if rebuild is due
        const bool need_rebuild = do_rebuild &&
            chrono::duration_cast<chrono::milliseconds>(Clock::now() - t_last_rebuild).count()
                >= static_cast<long long>(cfg.rebuild_interval_ms);

        // Compute current residual ‖F(x) - x‖_∞ (collect residuals if rebuild needed)
        real_t mx = scan_residuals(need_rebuild);

        residual_inf_atomic.store(mx, memory_order_relaxed);

        // Record trace point
        if (cfg.record_trace) out.trace.push_back({now_sec(), mx});

        // Rebuild priority scheduler if due
        if (need_rebuild) {
            sched.rebuild(residuals_buf);
            t_last_rebuild = Clock::now();
        }

        // Check convergence
        if (mx <= cfg.eps) {
            stop.store(true, memory_order_relaxed);
            break;
        }

        // Check update limit
        if (cfg.max_updates != 0 &&
            total_updates.load(memory_order_relaxed) >= cfg.max_updates) {
            stop.store(true, memory_order_relaxed);
            break;
        }
    }
});
```

### Monitor Responsibilities

| Task | Frequency | Action |
|------|-----------|--------|
| Residual scan | Every `monitor_interval_ms` | Compute `max_i |F_i(x) - x_i|` |
| Convergence check | After each scan | If `‖F(x)-x‖_∞ ≤ ε`, set `stop=true` |
| Timeout check | Each iteration | If wall time exceeded, set `stop=true` |
| Trace recording | After each scan | Append `(time, residual)` to trace |
| Scheduler rebuild | Every `rebuild_interval_ms` | Call `sched.rebuild(residuals)` if supported |

### Scheduler Rebuild

For priority schedulers like `ResidualBucketsScheduler`, the monitor periodically rebuilds the priority structure:

```cpp
const bool do_rebuild = sched.supports_rebuild() && cfg.rebuild_interval_ms > 0;

// In monitor loop:
const bool need_rebuild = do_rebuild &&
    chrono::duration_cast<chrono::milliseconds>(Clock::now() - t_last_rebuild).count()
        >= static_cast<long long>(cfg.rebuild_interval_ms);

if (need_rebuild) {
    sched.rebuild(residuals_buf);  // Re-prioritize based on current residuals
    t_last_rebuild = Clock::now();
}
```

This ensures high-residual coordinates are processed first as the solution evolves, improving convergence speed.

**Rebuild behavior:**
- Collects residuals for all n coordinates during the scan
- Calls `sched.rebuild(residuals)` to re-bucket indices by priority
- The scheduler resets thread hints so workers start from highest-priority buckets
- Controlled by `cfg.rebuild_interval_ms` (default: 500ms, 0 = never)

---

## Atomic Operations and Memory Ordering

### Primitives Used

```cpp
// 1. Stop flag (shared termination signal)
atomic<bool> stop{false};

// 2. Update counter
atomic<uint64_t> total_updates{0};

// 3. Current residual estimate
atomic<real_t> residual_inf_atomic{infinity()};

// 4. Per-element atomic access (C++20)
atomic_ref<real_t> xi_ref(x[i]);
```

### Memory Ordering Choices

All atomic operations use `memory_order_relaxed`:

```cpp
stop.load(memory_order_relaxed)
stop.store(true, memory_order_relaxed)
total_updates.fetch_add(1, memory_order_relaxed)
xi_ref.load(memory_order_relaxed)
xi_ref.store(xnew, memory_order_relaxed)
```

**Why relaxed?** See next section.

---

## Why Relaxed Memory Ordering is Safe

### The Key Insight

In asynchronous fixed-point iteration, **correctness does not require immediate visibility** of updates. The algorithm tolerates:

1. **Stale reads**: Thread A may not see Thread B's recent update
2. **Reordering**: Updates may become visible in different orders to different threads
3. **Delays**: Updates may take arbitrary time to propagate

### Theoretical Foundation

For a **contractive operator** F with contraction factor β < 1:

```
‖F(x) - F(y)‖ ≤ β · ‖x - y‖
```

The Asynchronous Convergence Theorem (Bertsekas & Tsitsiklis) states:

> If F is contractive and each coordinate is updated infinitely often with bounded staleness, the iteration converges to the fixed point x*.

**Bounded staleness** means: when computing F_i(x), the values x_j used are from at most D iterations ago, for some finite D.

### What Relaxed Ordering Guarantees

Even with `memory_order_relaxed`:

1. **Atomicity**: Each 8-byte `real_t` write is atomic (no torn writes)
2. **Eventual visibility**: Updates eventually become visible to other threads
3. **Per-location ordering**: Writes to the same `x[i]` are seen in a consistent order

### What We DON'T Need

| Property | Not Required | Why |
|----------|--------------|-----|
| Sequential consistency | ✗ | Algorithm tolerates any interleaving |
| Acquire-release | ✗ | No data dependencies between coordinates |
| Total order | ✗ | Different threads can see different orders |

### The `stop` Flag

The only cross-thread synchronization is the `stop` flag:

```cpp
// Workers check:
while (!stop.load(memory_order_relaxed)) { ... }

// Monitor sets:
stop.store(true, memory_order_relaxed);
```

With relaxed ordering, workers may continue for a few more iterations after `stop` is set. This is acceptable because:
- Extra iterations don't affect correctness
- Termination is guaranteed (flag eventually propagates)

---

## Convergence Under Asynchrony

### Update Equation

Each worker applies the relaxed update:

```cpp
x[i] = (1 - α) · x[i] + α · F_i(x)
```

Where:
- `α ∈ (0, 1]` is the step size (typically α = 1)
- `F_i(x)` is computed using the current (possibly stale) snapshot of x

### Contraction Property

For the Bellman operator in policy evaluation:

```
F_i(x) = r_i + β · Σ_j P_ij · x_j
```

With discount factor β < 1, F is a contraction in the ∞-norm:

```
‖F(x) - F(y)‖_∞ ≤ β · ‖x - y‖_∞
```

### Convergence Guarantee

Under async updates:

1. **Monotonic progress** (on average): Each update moves x closer to x*
2. **Bounded interference**: Stale reads introduce noise but don't prevent convergence
3. **Eventually consistent**: As updates slow down near convergence, staleness decreases

---

## Termination Protocol

### Shutdown Sequence

```
1. Monitor detects convergence (or timeout/update limit)
   └─→ stop.store(true)

2. Workers see stop=true
   └─→ Exit their loops

3. Main thread joins workers
   └─→ for (auto& th : workers) th.join();

4. Main thread ensures stop=true
   └─→ stop.store(true)  // Defensive

5. Main thread joins monitor
   └─→ monitor.join();

6. Collect results
   └─→ Return RunResult with final metrics
```

### Race Condition Handling

Multiple threads may set `stop=true` simultaneously:

```cpp
// Worker timeout check
if (timed_out()) {
    stop.store(true, memory_order_relaxed);
    break;
}

// Monitor convergence check
if (mx <= cfg.eps) {
    stop.store(true, memory_order_relaxed);
    break;
}
```

This is safe because:
- All paths set `stop=true` (same value)
- `store` to `atomic<bool>` is atomic

---

## Performance Characteristics

### Scalability

| Threads | Ideal Speedup | Actual Speedup | Bottleneck |
|---------|---------------|----------------|------------|
| 1 | 1× | 1× | - |
| 2 | 2× | ~1.8× | Cache coherence |
| 4 | 4× | ~3.2× | Memory bandwidth |
| 8+ | 8×+ | Varies | Scheduler contention |

### Memory Access Patterns

```
Worker 0: x[0], x[4], x[8], ...   ← Strided access (cache-friendly)
Worker 1: x[1], x[5], x[9], ...
Worker 2: x[2], x[6], x[10], ...
Worker 3: x[3], x[7], x[11], ...
```

With StaticBlocksScheduler:
```
Worker 0: x[0], x[1], x[2], ...   ← Contiguous access (excellent cache)
Worker 1: x[n/4], x[n/4+1], ...
```

### False Sharing Mitigation

Adjacent `x[i]` and `x[i+1]` may share a cache line (64 bytes = 8 doubles).

| Scheduler | False Sharing Risk |
|-----------|-------------------|
| StaticBlocks | Low (each thread works on contiguous region) |
| ResidualBuckets | Higher (scattered access patterns) |

---

## Summary

The Helios async runtime achieves high-performance parallel fixed-point iteration through:

1. **Lock-free updates**: Using `atomic_ref<real_t>` for per-element atomicity
2. **Relaxed memory ordering**: Exploiting the algorithm's tolerance for staleness
3. **Decoupled monitoring**: Separate thread for convergence checking
4. **Scheduler abstraction**: Pluggable work distribution strategies

The key insight is that **mathematical convergence guarantees** (contraction + bounded staleness) allow us to use **minimal synchronization** (relaxed atomics only), maximizing throughput while maintaining correctness.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Helios Async Runtime                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌───────────────┐   │
│   │Worker 0 │  │Worker 1 │  │Worker 2 │  ...  │   Monitor     │   │
│   │         │  │         │  │         │       │               │   │
│   │ next()  │  │ next()  │  │ next()  │       │ sleep(100ms)  │   │
│   │ F_i(x)  │  │ F_i(x)  │  │ F_i(x)  │       │ scan ‖F-x‖    │   │
│   │ x[i]=.. │  │ x[i]=.. │  │ x[i]=.. │       │ check ε       │   │
│   └────┬────┘  └────┬────┘  └────┬────┘       └───────┬───────┘   │
│        │            │            │                    │           │
│        │    atomic_ref<real_t>   │                    │           │
│        ▼            ▼            ▼                    ▼           │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │                    Shared x[0..n-1]                          │ │
│   │   (memory_order_relaxed for all accesses)                   │ │
│   └─────────────────────────────────────────────────────────────┘ │
│        │            │            │                    │           │
│        │            ▼            │                    │           │
│        │    ┌──────────────┐     │                    │           │
│        └───►│ atomic<bool> │◄────┘                    │           │
│             │    stop      │◄─────────────────────────┘           │
│             └──────────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
