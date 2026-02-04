# Helios Schedulers: Detailed Implementation Guide

This document provides an in-depth explanation of the scheduling subsystem in Helios, covering the abstract interface, concrete implementations, and the algorithms used for coordinate selection in asynchronous fixed-point iteration.

---

## Table of Contents

1. [Overview](#overview)
2. [Scheduler Interface](#scheduler-interface)
3. [StaticBlocksScheduler](#staticblocksscheduler)
4. [ResidualBucketsScheduler](#residualbucketsscheduler)
   - [Bucket Assignment Algorithm](#bucket-assignment-algorithm)
   - [Two-Pass Rebuild Algorithm](#two-pass-rebuild-algorithm)
   - [Lock-Free Index Dispatch](#lock-free-index-dispatch)
   - [Data Structure Layout](#data-structure-layout)
5. [Thread Safety Guarantees](#thread-safety-guarantees)
6. [Performance Considerations](#performance-considerations)

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

    // Atomically publish the new data structure
    std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}
```

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
| ResidualBuckets | Priority-ordered | Moderate (indices not contiguous) |

### Contention Points

**ResidualBucketsScheduler**:
- `cursor[b].fetch_add()` - contention when multiple threads target same bucket
- Mitigated by: bucket hints skip exhausted buckets

### When to Use Each Scheduler

| Scenario | Recommended Scheduler |
|----------|----------------------|
| Uniform residuals | StaticBlocks (simpler, cache-friendly) |
| Skewed residuals | ResidualBuckets (prioritizes hot spots) |
| Many threads, small n | StaticBlocks (less contention) |
| Few threads, large n | ResidualBuckets (better convergence) |

---

## Summary

The Helios scheduling subsystem provides:

1. **StaticBlocksScheduler**: Simple, zero-contention partitioning for uniform workloads
2. **ResidualBucketsScheduler**: Priority-based scheduling using logarithmic bucketing

Both schedulers are:
- Fully lock-free in the dispatch hot path
- Designed for continuous operation (cycling until convergence)
- Thread-safe for concurrent access

The ResidualBucketsScheduler additionally provides:
- O(n) rebuild with two-pass bucket sort
- Snapshot isolation for concurrent rebuild/dispatch
- Automatic fallback to round-robin when priority information is stale
