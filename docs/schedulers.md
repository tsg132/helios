# Helios Schedulers: Detailed Implementation Guide

This document provides an in-depth explanation of the scheduling subsystem in Helios, covering the abstract interface, concrete implementations, and the algorithms used for coordinate selection in asynchronous fixed-point iteration.

---

## Table of Contents

1. [Overview](#overview)
2. [Scheduler Interface](#scheduler-interface)
3. [StaticBlocksScheduler](#staticblocksscheduler)
4. [ShuffledBlocksScheduler](#shuffledblocksscheduler)
5. [TopKGSScheduler](#topkgsscheduler)
   - [Algorithm Overview](#algorithm-overview)
   - [Rebuild Algorithm](#rebuild-algorithm)
   - [Index Dispatch](#topk-index-dispatch)
6. [CATopKGSScheduler](#catopkgsscheduler)
   - [CA-TopK-GS Algorithm Overview](#ca-topk-gs-algorithm-overview)
   - [Conflict Key Function](#conflict-key-function)
   - [Rebuild Algorithm](#ca-topk-gs-rebuild-algorithm)
   - [Index Dispatch](#ca-topk-gs-index-dispatch)
7. [ResidualBucketsScheduler](#residualbucketsscheduler)
   - [Bucket Assignment Algorithm](#bucket-assignment-algorithm)
   - [Two-Pass Rebuild Algorithm](#two-pass-rebuild-algorithm)
   - [Lock-Free Index Dispatch](#lock-free-index-dispatch)
   - [Data Structure Layout](#data-structure-layout)
8. [Thread Safety Guarantees](#thread-safety-guarantees)
9. [Performance Considerations](#performance-considerations)

---

## Overview

In asynchronous fixed-point iteration, multiple worker threads concurrently update coordinates of the solution vector `x`. The **scheduler** determines which coordinate index each thread should process next.

The key design goals are:
- **Lock-free operation**: No mutexes in the hot path
- **Priority scheduling**: Process high-residual coordinates first (for faster convergence)
- **Continuous operation**: Schedulers cycle indefinitely until convergence
- **Snapshot isolation**: Rebuilding the schedule doesn't interfere with ongoing dispatches

---

## Scheduler Interface

All schedulers implement the abstract `Scheduler` interface defined in `include/helios/scheduler.h`:

```cpp
class Scheduler {
public:
    virtual ~Scheduler() = default;

    // Initialize scheduler for n coordinates with num_threads workers
    virtual void init(index_t n, size_t num_threads) = 0;

    // Return the next coordinate index for thread tid to process
    virtual index_t next(size_t tid) = 0;

    // Optional: notify scheduler of a residual update (for adaptive scheduling)
    virtual void notify(size_t tid, index_t i, real_t residual) {}

    // Rebuild internal priority structure from current residuals.
    // Default is no-op; priority schedulers (e.g., ResidualBucketsScheduler) override.
    virtual void rebuild(const std::vector<real_t>& residuals) { (void)residuals; }

    // Returns true if this scheduler benefits from periodic rebuild() calls.
    virtual bool supports_rebuild() const noexcept { return false; }

    // Return scheduler name for logging/debugging
    virtual string_view name() const noexcept { return "scheduler"; }
};
```

### Contract

| Method | Precondition | Postcondition |
|--------|--------------|---------------|
| `init(n, T)` | Called once before `next()` | Scheduler ready for T threads over [0, n) |
| `next(tid)` | `tid < num_threads` | Returns valid index in [0, n) or cycles |
| `notify(...)` | Optional | May influence future `next()` calls |
| `rebuild(residuals)` | `residuals.size() == n` | Priority structure rebuilt from residuals |
| `supports_rebuild()` | None | Returns true if scheduler uses `rebuild()` |

---

## StaticBlocksScheduler

**File**: `src/schedulers/static_blocks.cc`

The simplest scheduler: partitions `[0, n)` into contiguous blocks, one per thread. Each thread cycles through its own block indefinitely.

### Initialization

```cpp
void StaticBlocksScheduler::init(index_t n, size_t num_threads) {
    n_ = n;
    num_threads_ = (num_threads > 0) ? num_threads : 1;

    block_begin_.resize(num_threads_);
    block_end_.resize(num_threads_);
    cursor_.resize(num_threads_);

    const index_t base_size = n_ / static_cast<index_t>(num_threads_);
    const index_t remainder = n_ % static_cast<index_t>(num_threads_);

    index_t start = 0;
    for (size_t tid = 0; tid < num_threads_; ++tid) {
        // First 'remainder' threads get one extra element
        const index_t size = base_size + (tid < remainder ? 1 : 0);
        block_begin_[tid] = start;
        block_end_[tid] = start + size;
        cursor_[tid] = start;
        start += size;
    }
}
```

**Example**: `n=10, num_threads=3`
```
Thread 0: [0, 4)  → indices 0, 1, 2, 3
Thread 1: [4, 7)  → indices 4, 5, 6
Thread 2: [7, 10) → indices 7, 8, 9
```

### Index Dispatch

```cpp
index_t StaticBlocksScheduler::next(size_t tid) {
    if (tid >= num_threads_ || n_ == 0) return n_;  // Invalid: return sentinel

    const index_t begin = block_begin_[tid];
    const index_t end = block_end_[tid];

    if (begin >= end) return n_;  // Empty block

    const index_t i = cursor_[tid];

    // Advance cursor, wrapping around within the block
    cursor_[tid] = (i + 1 < end) ? (i + 1) : begin;

    return i;
}
```

**Behavior**: Thread `tid` cycles through its block: `begin → begin+1 → ... → end-1 → begin → ...`

### Characteristics

| Property | Value |
|----------|-------|
| Lock-free | Yes (no shared state between threads) |
| Priority | None (round-robin within block) |
| Cache locality | Excellent (contiguous access) |
| Load balancing | Static (blocks differ by at most 1) |

---

## ShuffledBlocksScheduler

**Files**: `include/helios/schedulers/shuffled_blocks.h`, `src/schedulers/shuffled_blocks.cc`

An enhancement over StaticBlocksScheduler that adds randomization. Each thread owns a contiguous block of indices but iterates through them in **shuffled order**. After completing a full epoch (one pass through the block), the order is reshuffled.

### Motivation

In some problems, the sequential access pattern of StaticBlocks can lead to systematic biases or slower convergence. Shuffling the access order within each block:
- Breaks correlation patterns between neighboring indices
- Can improve convergence for certain problem structures
- Maintains cache locality benefits (indices still come from a contiguous block)

### Initialization

```cpp
void ShuffledBlocksScheduler::init(index_t n, size_t num_threads) {
    // Partition [0, n) into contiguous blocks (same as StaticBlocks)
    // ...

    for (size_t tid = 0; tid < num_threads_; ++tid) {
        // Initialize per-thread RNG with deterministic seed
        rngs_[tid].seed(static_cast<uint64_t>(tid) * 0x9E3779B97F4A7C15ULL + 42);

        // Fill shuffled_indices_[tid] with [block_begin, block_end)
        std::iota(shuffled_indices_[tid].begin(), shuffled_indices_[tid].end(), start);

        // Initial shuffle
        std::shuffle(shuffled_indices_[tid].begin(), shuffled_indices_[tid].end(), rngs_[tid]);
    }
}
```

**Example**: `n=10, num_threads=2`
```
Thread 0 block: [0, 5) → indices 0, 1, 2, 3, 4
  Epoch 1 order: [3, 0, 4, 1, 2]  (shuffled)
  Epoch 2 order: [1, 4, 0, 2, 3]  (reshuffled)

Thread 1 block: [5, 10) → indices 5, 6, 7, 8, 9
  Epoch 1 order: [7, 5, 9, 6, 8]  (shuffled)
  Epoch 2 order: [6, 9, 5, 8, 7]  (reshuffled)
```

### Index Dispatch

```cpp
index_t ShuffledBlocksScheduler::next(size_t tid) {
    auto& indices = shuffled_indices_[tid];
    const size_t pos = cursor_[tid];
    const index_t i = indices[pos];

    // Advance cursor
    cursor_[tid] = pos + 1;

    // If epoch complete, reshuffle for next epoch
    if (cursor_[tid] >= indices.size()) {
        reshuffle(tid);  // Shuffles indices and resets cursor to 0
    }

    return i;
}
```

### Characteristics

| Property | Value |
|----------|-------|
| Lock-free | Yes (no shared state between threads) |
| Priority | None (random within block) |
| Cache locality | Good (indices from contiguous block, but accessed randomly) |
| Load balancing | Static (blocks differ by at most 1) |
| Epoch behavior | Reshuffles after each complete pass |

### When to Use

- When StaticBlocks shows slow convergence due to access pattern correlations
- When you want randomization without the overhead of priority scheduling
- As a middle ground between StaticBlocks and ResidualBuckets

---

## TopKGSScheduler

**Files**: `include/helios/schedulers/topk_gs.h`, `src/schedulers/topk_gs.cc`

An approximation to Gauss-Southwell coordinate selection using a **Top-K hot set**. Instead of always selecting the single coordinate with maximum residual (expensive), we maintain a set of K coordinates with the largest residuals and dispatch from that set first.

### Motivation

Exact Gauss-Southwell chooses `i = argmax_i |F_i(x) - x_i|` at each step, which is O(n) per update and too expensive. Top-K GS approximates this by:

1. **Rebuild phase**: Periodically compute all residuals and select the K largest (O(n) using `nth_element`)
2. **Priority phase**: Dispatch indices from the hot set (O(1) per update)
3. **Coverage phase**: Fall back to shuffled blocks to ensure all coordinates are updated

This provides a good trade-off between greedy convergence and computational overhead.

### Parameters

```cpp
struct TopKGSParams {
    index_t K = 0;        // Hot set size (0 = auto: max(n*0.01, threads*256))
    bool sort_hot = false; // Sort hot set by descending residual
    uint64_t seed = 0;     // RNG seed for fallback (0 = default)
};
```

### Algorithm Overview

```
REBUILD (called periodically by runtime monitor):
  1. Compute residuals: rho[i] = |F_i(x) - x_i| for all i
  2. Select Top-K: Use nth_element to find K largest
  3. Optionally sort hot set by descending residual
  4. Reset hot_cursor = 0, reshuffle fallback blocks

NEXT(tid):
  1. k = atomic_fetch_add(hot_cursor, 1)
  2. If k < K: return hot[k]        // Priority phase
  3. Else: return fallback.next(tid) // Coverage phase
```

### Rebuild Algorithm

The rebuild uses `std::nth_element` for O(n) average-case Top-K selection:

```cpp
void TopKGSScheduler::rebuild(const std::vector<real_t>& residuals) {
    // Build (residual, index) pairs
    std::vector<std::pair<real_t, index_t>> pairs(n_);
    for (index_t i = 0; i < n_; ++i) {
        pairs[i] = {residuals[i], i};
    }

    // Partition: largest K elements at the end
    const size_t pivot_pos = n_ - K_;
    std::nth_element(pairs.begin(), pairs.begin() + pivot_pos, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    // Extract top-K indices
    for (index_t i = 0; i < K_; ++i) {
        d->hot[i] = pairs[pivot_pos + i].second;
    }

    // Optional: sort for closer-to-greedy order
    if (params_.sort_hot) {
        std::sort(d->hot.begin(), d->hot.end(),
                  [&residuals](index_t a, index_t b) { return residuals[a] > residuals[b]; });
    }

    // Reset cursors and reshuffle fallback
    d->hot_cursor.store(0, std::memory_order_relaxed);
    // ... reshuffle fallback blocks ...
}
```

**Key insight**: For any index `i` in the hot set:
```
rho[i] >= rho_sorted[K]
```
So all dispatched indices are "high residual" and approximate greedy selection.

### TopK Index Dispatch

```cpp
index_t TopKGSScheduler::next(size_t tid) {
    auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);

    // Priority phase: try to get from hot set
    const index_t k = d->hot_cursor.fetch_add(1, std::memory_order_relaxed);
    if (k < K_) {
        return d->hot[k];  // High-residual index
    }

    // Coverage phase: fall back to shuffled blocks
    // (ensures all coordinates are updated for convergence)
    return fallback_next(d, tid);
}
```

**Two-phase behavior**:
1. **Priority phase** (first K calls after rebuild): Returns high-residual indices from hot set
2. **Coverage phase** (remaining calls): Returns indices from shuffled blocks for full coverage

### Relationship to Gauss-Southwell

| Method | Selection Rule | Cost per Update |
|--------|---------------|-----------------|
| Exact GS | `argmax_i rho[i]` | O(n) |
| Top-K GS | Any `i` from hot set | O(1) |

Top-K GS trades optimality for efficiency: instead of always picking THE maximum, we pick from a set of K large values.

### Recommended Parameters

```cpp
// Default K formula:
K = max(n * 0.01, num_threads * 256)  // clamped to [1, n]

// Rebuild cadence (set in RuntimeConfig):
cfg.rebuild_interval_ms = 100;  // Every 100ms
```

### Characteristics

| Property | Value |
|----------|-------|
| Lock-free | Yes (atomic cursor for hot set) |
| Priority | Top-K approximation to Gauss-Southwell |
| Cache locality | Moderate (hot set indices not contiguous) |
| Load balancing | Dynamic (hot set) + Static (fallback blocks) |
| Rebuild cost | O(n) using nth_element |

### When to Use

- When residuals are highly skewed (some coordinates much larger than others)
- For problems where greedy coordinate selection significantly improves convergence
- When you want priority scheduling with guaranteed coverage
- As a cheaper alternative to exact Gauss-Southwell

---

## CATopKGSScheduler

**Files**: `include/helios/schedulers/ca_topk_gs.h`, `src/schedulers/ca_topk_gs.cc`

A **Conflict-Aware** extension of Top-K Gauss-Southwell that reduces parallel contention by distributing hot indices into conflict groups. Instead of all threads competing for a single shared hot set cursor, indices are partitioned by their memory locality (cache block) so that threads accessing different groups touch different memory regions.

### Motivation

Standard Top-K GS has a contention bottleneck: all threads race to increment `hot_cursor` during the priority phase. When indices in the hot set are close together in memory, multiple threads may also experience false sharing when updating neighboring x[i] values.

CA-TopK-GS addresses this by:
1. **Conflict grouping**: Assigning hot indices to G groups based on a conflict key (cache block proxy)
2. **Round-robin group access**: Threads pick from groups in round-robin order, spreading contention
3. **Cache-friendly grouping**: Indices in the same cache block go to the same group, so different groups access different memory regions

### Parameters

```cpp
struct CATopKGSParams {
    index_t K = 0;           // Hot set size (0 = auto: max(n*0.01, threads*256))
    size_t G = 0;            // Number of conflict groups (0 = auto: 4*threads)
    index_t block_size = 256; // Cache block proxy: key(i) = (i / block_size) % G
    bool sort_within_group = true;  // Sort indices within each group by residual
    uint64_t seed = 0;       // RNG seed for group order and fallback
};
```

### CA-TopK-GS Algorithm Overview

```
REBUILD (called periodically by runtime monitor):
  1. Compute residuals: rho[i] = |F_i(x) - x_i| for all i
  2. Select Top-K: Use nth_element to find K largest
  3. Initialize G conflict groups (empty)
  4. Assign each top-K index to a group: g = key(i) = (i / block_size) % G
  5. Optionally sort indices within each group by descending residual
  6. Shuffle group visiting order for load balancing
  7. Reset group_rr_cursor = 0, reshuffle fallback blocks
  8. Publish snapshot atomically

NEXT(tid):
  1. Repeat up to G times:
       t = atomic_fetch_add(group_rr_cursor, 1)
       g = group_order[t % G]
       k = atomic_fetch_add(group_cursor[g], 1)
       if k < size(groups[g]): return groups[g][k]  // Priority phase
  2. If all groups exhausted: return fallback.next(tid)  // Coverage phase
```

### Conflict Key Function

The conflict key determines which group an index belongs to. The default uses a **cache block proxy**:

```cpp
index_t key(index_t i) const {
    return (i / params_.block_size) % G_;
}
```

**Example** with `block_size = 256` and `G = 8`:
```
Indices 0-255    → key = (0-255)/256 % 8 = 0
Indices 256-511  → key = 1
Indices 512-767  → key = 2
...
Indices 2048-2303 → key = 0 (wraps around)
```

**Why cache block proxy works**:
- Consecutive indices in memory often share cache lines (64 bytes = ~16 floats)
- Grouping by block_size ensures indices in the same group are memory-local
- Different groups access different memory regions, reducing false sharing
- When thread A updates x[i] from group 0 and thread B updates x[j] from group 1, they're unlikely to contend on cache lines

### CA-TopK-GS Rebuild Algorithm

```cpp
void CATopKGSScheduler::rebuild(const std::vector<real_t>& residuals) {
    // Step 1: Build (residual, index) pairs and select Top-K
    std::vector<std::pair<real_t, index_t>> pairs(n_);
    for (index_t i = 0; i < n_; ++i) {
        pairs[i] = {residuals[i], i};
    }

    const size_t pivot_pos = n_ - K_;
    std::nth_element(pairs.begin(), pairs.begin() + pivot_pos, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    // Step 2: Initialize conflict groups
    auto d = std::make_shared<EpochData>();
    d->groups.resize(G_);
    d->group_cursor = std::make_unique<std::atomic<index_t>[]>(G_);

    for (size_t g = 0; g < G_; ++g) {
        d->groups[g].clear();
        d->group_cursor[g].store(0, std::memory_order_relaxed);
    }

    // Step 3: Assign each top-K index to its conflict group
    for (index_t i = 0; i < K_; ++i) {
        const index_t idx = pairs[pivot_pos + i].second;
        const index_t g = key(idx);  // (idx / block_size) % G
        d->groups[g].push_back(idx);
    }

    // Step 4: Optionally sort within groups
    if (params_.sort_within_group) {
        for (size_t g = 0; g < G_; ++g) {
            std::sort(d->groups[g].begin(), d->groups[g].end(),
                      [&residuals](index_t a, index_t b) {
                          return residuals[a] > residuals[b];
                      });
        }
    }

    // Step 5: Shuffle group visiting order
    d->group_order.resize(G_);
    std::iota(d->group_order.begin(), d->group_order.end(), size_t{0});
    std::shuffle(d->group_order.begin(), d->group_order.end(), rng);

    // Step 6: Reset cursors and fallback
    d->group_rr_cursor.store(0, std::memory_order_relaxed);
    // ... initialize fallback shuffled blocks ...

    std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}
```

**Group distribution example** (`K=20`, `G=4`, `block_size=256`):

If top-K indices are: `[10, 50, 300, 320, 600, 650, 900, 950, ...]`
```
Group 0 (key=0): indices with (i/256)%4 = 0 → [10, 50, 1024, ...]
Group 1 (key=1): indices with (i/256)%4 = 1 → [300, 320, ...]
Group 2 (key=2): indices with (i/256)%4 = 2 → [600, 650, ...]
Group 3 (key=3): indices with (i/256)%4 = 3 → [900, 950, ...]
```

### CA-TopK-GS Index Dispatch

```cpp
index_t CATopKGSScheduler::next(size_t tid) {
    auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);
    if (!d || n_ == 0) return n_;

    // Priority phase: round-robin across conflict groups
    for (size_t attempt = 0; attempt < G_; ++attempt) {
        // Get next group in round-robin order
        const size_t t = d->group_rr_cursor.fetch_add(1, std::memory_order_relaxed);
        const size_t g = d->group_order[t % G_];

        // Try to pop from this group
        const index_t k = d->group_cursor[g].fetch_add(1, std::memory_order_relaxed);
        if (k < static_cast<index_t>(d->groups[g].size())) {
            return d->groups[g][k];  // Success!
        }
        // Group exhausted, try next
    }

    // Coverage phase: all groups exhausted
    return fallback_next(d, tid);
}
```

**Two-level dispatch visualization**:

```
                     group_rr_cursor
                           ↓
group_order:     [2, 0, 3, 1]  (shuffled)
                  ↑
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
groups[2]:  [i₅, i₆, i₇]    groups[0]:  [i₀, i₁]
            ↑ cursor=0                   ↑ cursor=0

Thread A: t=0 → group 2 → k=0 → return i₅
Thread B: t=1 → group 0 → k=0 → return i₀
Thread C: t=2 → group 3 → k=0 → return i₈
Thread D: t=3 → group 1 → k=0 → return i₃
Thread E: t=4 → group 2 → k=1 → return i₆
...
```

### Comparison: TopKGS vs CA-TopKGS

| Aspect | TopKGS | CA-TopKGS |
|--------|--------|-----------|
| Hot set structure | Single array with one cursor | G arrays, each with own cursor |
| Contention pattern | All threads compete on `hot_cursor` | Threads spread across G group cursors |
| Memory access | Hot indices may be scattered | Hot indices grouped by cache locality |
| Dispatch cost | O(1) | O(G) worst case, O(1) typical |
| Rebuild cost | O(n) | O(n) (same, plus G-way distribution) |

### Recommended Parameters

```cpp
// Default formulas:
K = max(n * 0.01, num_threads * 256)  // Same as TopKGS
G = 4 * num_threads                    // More groups = less contention
block_size = 256                       // ~16 cache lines per block

// Tuning:
// - Increase G if you see contention on group cursors
// - Decrease block_size for finer-grained locality
// - Set sort_within_group=true for greedy-like behavior
```

### Characteristics

| Property | Value |
|----------|-------|
| Lock-free | Yes (atomic cursors per group) |
| Priority | Top-K with conflict-aware grouping |
| Cache locality | Good (grouped by memory locality) |
| Load balancing | Dynamic (round-robin across groups) + Static (fallback) |
| Rebuild cost | O(n) + O(K log K) if sorting within groups |
| Contention | Reduced vs TopKGS (spread across G cursors) |

### When to Use

- When TopKGS shows contention bottlenecks with many threads
- For large problems where cache locality matters
- When hot indices tend to cluster in memory regions
- As an upgrade from TopKGS when scaling to more threads

---

## ResidualBucketsScheduler

**Files**: `include/helios/schedulers/residual_buckets.h`, `src/schedulers/residual_buckets.cc`

A priority scheduler that processes high-residual coordinates first. Uses logarithmic bucketing to group coordinates by residual magnitude.

### Parameters

```cpp
struct Params {
    uint32_t num_buckets = 32;       // Number of priority buckets
    real_t base = 1e-12;             // Residual scale factor
    bool fallback_round_robin = true; // Cycle after buckets exhausted
};
```

---

### Bucket Assignment Algorithm

The core bucketing formula assigns each coordinate to a bucket based on its residual:

```
b(i) = min(B-1, max(0, floor(log₂(rᵢ / base))))
```

Where:
- `rᵢ` = residual of coordinate i
- `base` = scale parameter (e.g., 1e-12)
- `B` = number of buckets

**Implementation**:

```cpp
uint32_t ResidualBucketsScheduler::bucket_of_(real_t r, real_t base, uint32_t B) {
    // r <= 0 goes to bucket 0
    if (!(r > 0)) return 0;

    // x = r / base
    const real_t x = r / base;

    // If x <= 1, log2(x) <= 0, so bucket 0
    if (!(x > 1)) return 0;

    // lg = log2(r / base)
    const real_t lg = std::log2(x);

    // b = floor(lg), clamped to [0, B-1]
    int b = static_cast<int>(std::floor(lg));

    if (b >= static_cast<int>(B)) return B - 1;

    return static_cast<uint32_t>(b);
}
```

**Bucket ranges** (with `base = 1e-12`, `B = 32`):

| Bucket | Residual Range |
|--------|----------------|
| 0 | r ≤ 1e-12 (or r ≤ 0) |
| 1 | 1e-12 < r ≤ 2e-12 |
| 2 | 2e-12 < r ≤ 4e-12 |
| 3 | 4e-12 < r ≤ 8e-12 |
| ... | ... |
| 31 | r > 2³⁰ × 1e-12 ≈ 1e-3 |

**Key insight**: Each bucket represents a factor of 2× in residual magnitude. Higher bucket = higher priority.

---

### Two-Pass Rebuild Algorithm

When `rebuild(residuals)` is called, the scheduler builds a new index ordering:

```cpp
void ResidualBucketsScheduler::rebuild(const std::vector<real_t>& residuals) {
    const uint32_t B = std::max<uint32_t>(1, params_.num_buckets);
    const real_t base = std::max<real_t>(params_.base, 1e-300);

    // ═══════════════════════════════════════════════════════════════════
    // PASS 1: Count how many indices fall into each bucket
    // ═══════════════════════════════════════════════════════════════════
    std::vector<uint32_t> counts(B, 0);
    bool any_non_tiny = false;

    for (index_t i = 0; i < n_; ++i) {
        const real_t r = residuals[i];
        if (r > base) any_non_tiny = true;
        const uint32_t b = bucket_of_(r, base, B);
        ++counts[b];
    }

    // Build new Data structure
    auto d = std::make_shared<Data>();
    d->n = n_;
    d->B = B;
    d->all_tiny = !any_non_tiny;

    // ═══════════════════════════════════════════════════════════════════
    // Build offsets array (exclusive prefix sum)
    // ═══════════════════════════════════════════════════════════════════
    d->offsets.resize(B + 1);
    d->offsets[0] = 0;
    for (uint32_t b = 0; b < B; ++b) {
        d->offsets[b + 1] = d->offsets[b] + counts[b];
    }

    // Allocate indices array
    d->indices.resize(d->offsets[B]);

    // Construct cursor vector (each atomic initialized to 0)
    d->cursor = std::vector<std::atomic<uint32_t>>(B);
    d->rr_cursor.store(0, std::memory_order_relaxed);

    // ═══════════════════════════════════════════════════════════════════
    // PASS 2: Place each index into its bucket
    // ═══════════════════════════════════════════════════════════════════
    std::vector<uint32_t> write_ptr = d->offsets;  // Copy as write pointers

    for (index_t i = 0; i < n_; ++i) {
        const real_t r = residuals[i];
        const uint32_t b = bucket_of_(r, base, B);
        const uint32_t pos = write_ptr[b]++;
        d->indices[pos] = i;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Reset thread hints so all threads start from highest-priority bucket
    // ═══════════════════════════════════════════════════════════════════
    std::fill(thread_bucket_hint_.begin(), thread_bucket_hint_.end(), 0);

    // Atomically publish the new data structure
    std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}
```

**Important**: After placing indices, we reset all thread hints to 0 so threads start scanning from the highest-priority bucket (B-1) on the next `next()` call. This ensures threads don't miss newly populated high-priority buckets after rebuild.

**Example**: `n=10`, residuals yield bucket assignments `[0,0,1,1,1,2,2,3,3,3]`

```
Pass 1: counts = [2, 3, 2, 3]

Offsets (prefix sum):
  offsets[0] = 0
  offsets[1] = 0 + 2 = 2
  offsets[2] = 2 + 3 = 5
  offsets[3] = 5 + 2 = 7
  offsets[4] = 7 + 3 = 10

Pass 2: Fill indices array
  Bucket 0: indices[0..1]  = [0, 1]
  Bucket 1: indices[2..4]  = [2, 3, 4]
  Bucket 2: indices[5..6]  = [5, 6]
  Bucket 3: indices[7..9]  = [7, 8, 9]

Final layout:
  indices:  [0, 1 | 2, 3, 4 | 5, 6 | 7, 8, 9]
             ↑     ↑         ↑      ↑
           off[0] off[1]   off[2]  off[3]    off[4]=10
```

---

### Lock-Free Index Dispatch

The `next(tid)` function dispatches indices to threads without locks:

```cpp
index_t ResidualBucketsScheduler::next(size_t tid) {
    // ═══════════════════════════════════════════════════════════════════
    // Step 1: Atomically load current data snapshot
    // ═══════════════════════════════════════════════════════════════════
    auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);
    if (!d) return kInvalidIndex;

    // ═══════════════════════════════════════════════════════════════════
    // Step 2: Fast path - round-robin if all residuals are tiny
    // ═══════════════════════════════════════════════════════════════════
    if (params_.fallback_round_robin && d->all_tiny) {
        index_t i = d->rr_cursor.fetch_add(1, std::memory_order_relaxed);
        return i % d->n;  // Wrap around for continuous operation
    }

    const uint32_t B = d->B;

    // Use thread's hint to skip exhausted high buckets
    const size_t tt = std::min(tid, num_threads_ - 1);
    uint32_t start_from = thread_bucket_hint_[tt];

    // ═══════════════════════════════════════════════════════════════════
    // Step 3: Search buckets from highest priority to lowest
    // ═══════════════════════════════════════════════════════════════════
    for (uint32_t off = start_from; off < B; ++off) {
        // Map offset to bucket index: highest bucket first
        // off=0 → bucket B-1 (highest priority)
        // off=1 → bucket B-2
        // ...
        const uint32_t b = (B - 1) - off;

        const uint32_t begin = d->offsets[b];
        const uint32_t end = d->offsets[b + 1];

        if (begin == end) continue;  // Empty bucket

        // ═══════════════════════════════════════════════════════════════
        // Step 4: Atomically claim next index in this bucket
        // ═══════════════════════════════════════════════════════════════
        const uint32_t k = d->cursor[b].fetch_add(1, std::memory_order_relaxed);
        const uint32_t idx = begin + k;

        if (idx < end) {
            // Success! Update hint for next call
            thread_bucket_hint_[tt] = off;
            return d->indices[idx];
        }
        // Bucket exhausted (k >= bucket_size), try next bucket
    }

    // ═══════════════════════════════════════════════════════════════════
    // Step 5: All buckets exhausted - fall back to round-robin
    // ═══════════════════════════════════════════════════════════════════
    if (params_.fallback_round_robin) {
        index_t i = d->rr_cursor.fetch_add(1, std::memory_order_relaxed);
        return i % d->n;
    }

    return kInvalidIndex;
}
```

**Visualization of bucket claiming**:

```
Initial state (bucket 2 with 5 indices):
  indices: [..., i₅, i₆, i₇, i₈, i₉, ...]
                 ↑                   ↑
               begin=5             end=10
  cursor[2] = 0

Thread A calls next():
  k = cursor[2].fetch_add(1) → k=0, cursor[2]=1
  idx = 5 + 0 = 5 < 10 ✓
  return indices[5] = i₅

Thread B calls next():
  k = cursor[2].fetch_add(1) → k=1, cursor[2]=2
  idx = 5 + 1 = 6 < 10 ✓
  return indices[6] = i₆

... (after 5 successful claims) ...

Thread F calls next():
  k = cursor[2].fetch_add(1) → k=5, cursor[2]=6
  idx = 5 + 5 = 10 >= 10 ✗
  → bucket exhausted, try next bucket
```

---

### Data Structure Layout

```cpp
struct Data {
    index_t n = 0;                      // Problem size
    uint32_t B = 0;                     // Number of buckets

    vector<index_t> indices;            // Flattened array of all indices
    vector<uint32_t> offsets;           // CSR-style: offsets[b] = start of bucket b
    vector<atomic<uint32_t>> cursor;    // Per-bucket atomic cursor

    atomic<index_t> rr_cursor{0};       // Round-robin fallback cursor
    bool all_tiny = false;              // True if all residuals <= base
};
```

**Memory layout**:

```
offsets:  [0        |3       |7       |9       |12      ]
           ↓         ↓        ↓        ↓        ↓
indices:  [i₀,i₁,i₂ |i₃,i₄,i₅,i₆|i₇,i₈ |i₉,i₁₀,i₁₁]
          ←bucket 0→ ←─bucket 1─→ ←b 2→ ←──bucket 3──→

cursor:   [3,        4,       2,       0      ]
           ↑         ↑        ↑        ↑
         exhausted  over    active   empty
```

---

## Thread Safety Guarantees

### StaticBlocksScheduler

- **Thread isolation**: Each thread only accesses its own `cursor_[tid]`
- **No synchronization needed**: Completely lock-free

### ShuffledBlocksScheduler

- **Thread isolation**: Each thread only accesses its own `shuffled_indices_[tid]`, `cursor_[tid]`, and `rngs_[tid]`
- **No synchronization needed**: Completely lock-free
- **Deterministic seeding**: Each thread's RNG is seeded based on `tid` for reproducibility

### TopKGSScheduler

| Operation | Synchronization | Guarantee |
|-----------|-----------------|-----------|
| `data_` access | `atomic_load/store` | Snapshot isolation |
| `hot_cursor` | `fetch_add` | Atomic increment, no duplicates in priority phase |
| Fallback cursors | Per-thread | No contention |

- **Two-phase dispatch**: Hot set uses shared atomic cursor; fallback is thread-isolated
- **Rebuild safety**: New epoch data published atomically, workers see consistent snapshot

### CATopKGSScheduler

| Operation | Synchronization | Guarantee |
|-----------|-----------------|-----------|
| `data_` access | `atomic_load/store` | Snapshot isolation |
| `group_rr_cursor` | `fetch_add` | Atomic increment for round-robin |
| `group_cursor[g]` | `fetch_add` | Per-group atomic increment |
| Fallback cursors | Per-thread | No contention |

- **Multi-cursor dispatch**: Contention spread across G group cursors instead of one
- **Group round-robin**: Threads naturally distribute across groups via atomic increment
- **Rebuild safety**: New epoch data published atomically, workers see consistent snapshot

### ResidualBucketsScheduler

| Operation | Synchronization | Guarantee |
|-----------|-----------------|-----------|
| `data_` access | `atomic_load/store` | Snapshot isolation |
| `cursor[b]` | `fetch_add` | Atomic increment |
| `rr_cursor` | `fetch_add` | Atomic increment |
| `thread_bucket_hint_` | Per-thread | No contention |

**Memory ordering**:
- `memory_order_release` on publish (rebuild)
- `memory_order_acquire` on load (next)
- `memory_order_relaxed` for cursors (ordering not required)

---

## Performance Considerations

### Cache Behavior

| Scheduler | Access Pattern | Cache Efficiency |
|-----------|---------------|------------------|
| StaticBlocks | Sequential within block | Excellent |
| ShuffledBlocks | Random within block | Good (block is contiguous in memory) |
| TopKGS | Priority then random | Moderate (hot set scattered, fallback good) |
| CA-TopKGS | Priority, grouped by locality | Good (conflict groups are cache-aligned) |
| ResidualBuckets | Priority-ordered | Moderate (indices not contiguous) |

### Contention Points

**TopKGSScheduler**:
- `hot_cursor.fetch_add()` - contention during priority phase when all threads compete for hot set
- Mitigated by: hot set exhausts quickly (K calls), then threads switch to isolated fallback

**CATopKGSScheduler**:
- `group_rr_cursor.fetch_add()` - light contention to determine which group to try
- `group_cursor[g].fetch_add()` - per-group contention, spread across G groups
- Mitigated by: G cursors instead of 1, threads naturally spread across groups

**ResidualBucketsScheduler**:
- `cursor[b].fetch_add()` - contention when multiple threads target same bucket
- Mitigated by: bucket hints skip exhausted buckets

### When to Use Each Scheduler

| Scenario | Recommended Scheduler |
|----------|----------------------|
| Uniform residuals | StaticBlocks (simpler, cache-friendly) |
| Uniform residuals + slow convergence | ShuffledBlocks (breaks access correlations) |
| Highly skewed residuals | TopKGS (approximates Gauss-Southwell) |
| Highly skewed + many threads | CA-TopKGS (reduces contention vs TopKGS) |
| Moderately skewed residuals | ResidualBuckets (continuous prioritization) |
| Many threads, small n | StaticBlocks (less contention) |
| Few threads, large n | ResidualBuckets or TopKGS (better convergence) |
| Want randomization, minimal overhead | ShuffledBlocks |
| Want greedy-like + guaranteed coverage | TopKGS |
| Want greedy-like + cache locality | CA-TopKGS |

---

## Summary

The Helios scheduling subsystem provides:

1. **StaticBlocksScheduler**: Simple, zero-contention partitioning for uniform workloads
2. **ShuffledBlocksScheduler**: Randomized access within blocks, reshuffles each epoch
3. **TopKGSScheduler**: Top-K Gauss-Southwell approximation with fallback coverage
4. **CATopKGSScheduler**: Conflict-aware Top-K GS with reduced contention
5. **ResidualBucketsScheduler**: Priority-based scheduling using logarithmic bucketing

All schedulers are:
- Fully lock-free in the dispatch hot path
- Designed for continuous operation (cycling until convergence)
- Thread-safe for concurrent access

The ShuffledBlocksScheduler additionally provides:
- Randomized iteration order to break access pattern correlations
- Automatic reshuffling at epoch boundaries
- Deterministic seeding for reproducibility

The TopKGSScheduler additionally provides:
- O(n) rebuild using nth_element for Top-K selection
- Two-phase dispatch: priority (hot set) then coverage (shuffled blocks)
- Approximates Gauss-Southwell coordinate selection efficiently
- Guaranteed coverage via fallback to ensure convergence

The CATopKGSScheduler additionally provides:
- Everything TopKGS provides, plus:
- Conflict grouping by cache locality to reduce false sharing
- G independent cursors instead of one, reducing contention
- Round-robin group access for load balancing across groups
- Better scaling with many threads compared to TopKGS

The ResidualBucketsScheduler additionally provides:
- O(n) rebuild with two-pass bucket sort
- Snapshot isolation for concurrent rebuild/dispatch
- Automatic fallback to round-robin when priority information is stale
