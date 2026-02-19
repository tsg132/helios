# Conflict-Aware Top-K Gauss-Southwell Scheduler: Mathematical Analysis

## 1. Background: The Coordinate Selection Problem

We are solving the fixed-point equation x = F(x) for F: R^n -> R^n via asynchronous iteration. At each step, a worker thread selects a coordinate i and performs:

```
x_i <- (1 - alpha) * x_i + alpha * F_i(x)
```

The **residual** at coordinate i is:

```
rho_i(x) = |F_i(x) - x_i|
```

This measures how far coordinate i is from its fixed-point condition. The global residual is:

```
||F(x) - x||_inf = max_i rho_i(x)
```

Convergence is declared when ||F(x) - x||_inf <= eps.

The question: **which coordinate i should a worker update next?**

---

## 2. Exact Gauss-Southwell Rule

The classical Gauss-Southwell (GS) rule selects the coordinate with maximum residual:

```
i* = argmax_{i in {0,...,n-1}} rho_i(x)
```

**Why this is optimal.** The contraction mapping theorem guarantees that each application of F reduces the error. Updating the coordinate with the largest residual provides the greatest per-step reduction in ||F(x) - x||_inf, because:

```
Delta_i = rho_i(x)   (the residual eliminated by updating coordinate i)
```

Choosing i* = argmax rho_i maximizes Delta_i.

**Why this is impractical.** Computing argmax requires O(n) work per update, the same cost as a full sweep. For n = 10^6, this makes GS slower than naive Jacobi despite needing fewer iterations.

---

## 3. Top-K Approximation

Top-K GS replaces the per-update O(n) argmax with a **periodic** O(n) selection of the K coordinates with largest residuals, called the **hot set** H:

```
H = {i_1, i_2, ..., i_K}   where   rho_{i_j}(x) >= rho_{i}(x)   for all i not in H
```

This is computed via `nth_element`, which partitions the array such that:

```
For the sorted residual vector rho_sigma[0] >= rho_sigma[1] >= ... >= rho_sigma[n-1]:

    H = {sigma(0), sigma(1), ..., sigma(K-1)}
```

The default K is:

```
K = clamp(max(0.01 * n, 256 * T), 1, n)
```

where T is the number of threads. The 0.01n term ensures coverage of the top 1% of residuals; the 256T term ensures enough work to amortize rebuild cost across threads.

### Two-Phase Dispatch

After a rebuild, the scheduler operates in two phases:

1. **Priority phase**: Dispatch indices from H sequentially. Each call to `next()` atomically increments a shared cursor and returns the next hot index. This phase lasts K calls.

2. **Coverage phase**: After the hot set is exhausted, fall back to per-thread shuffled blocks over [0, n) to ensure all coordinates are updated at least once per epoch.

The coverage phase is essential: without it, coordinates with small residuals are never updated, and convergence stalls.

---

## 4. The Contention Problem

In Top-K GS, all T threads compete on a **single atomic cursor** during the priority phase:

```
k = hot_cursor.fetch_add(1)    // T threads contend here
if k < K: return hot[k]
```

This creates two problems:

**Problem 1: Cursor contention.** On hardware with T cores, `fetch_add` on a single cache line costs O(T) cycles per operation due to cache-coherence traffic (MESI protocol invalidations). At T = 8 with ~40 ns per contested atomic, the priority phase alone costs 40 * K nanoseconds of serialized time.

**Problem 2: False sharing on x[].** If hot indices i_1, i_2, ..., i_K happen to be close in memory (e.g., i_j and i_{j+1} differ by < 8, which means they share a 64-byte cache line), then threads updating x[i_j] and x[i_{j+1}] simultaneously cause cache-line bouncing even though they update different coordinates.

---

## 5. CA-TopK-GS: Conflict-Aware Extension

CA-TopK-GS addresses both problems by **partitioning the hot set into G conflict groups** based on memory locality, then spreading thread access across groups.

### 5.1 The Conflict Key Function

Define the key function kappa: {0, ..., n-1} -> {0, ..., G-1}:

```
kappa(i) = floor(i / B) mod G
```

where B is the cache-block size parameter (default 256) and G is the number of conflict groups (default 4T).

This maps the coordinate space into repeating stripes of width B:

```
Coordinates:  [0, B)     [B, 2B)    [2B, 3B)   ...  [(G-1)B, GB)   [GB, (G+1)B)  ...
Group:            0          1          2       ...      G-1              0         ...
```

**Why this works.** Two coordinates i and j are in the **same** group only if floor(i/B) === floor(j/B) (mod G). When B = 256 and `real_t = double` (8 bytes), each block spans 256 * 8 = 2048 bytes = 32 cache lines. Coordinates in **different** groups are guaranteed to be at least B doubles apart in memory, eliminating false sharing on x[].

The code ([ca_topk_gs.h:70-72](include/helios/schedulers/ca_topk_gs.h#L70-L72)):

```cpp
index_t key(index_t i) const {
    return (i / params_.block_size) % static_cast<index_t>(G_);
}
```

### 5.2 Group Construction (Rebuild)

Given the hot set H from Top-K selection, partition it into G groups:

```
C_g = {i in H : kappa(i) = g}     for g = 0, 1, ..., G-1
```

So H = C_0 union C_1 union ... union C_{G-1} (disjoint union) and |C_0| + |C_1| + ... + |C_{G-1}| = K.

The code that performs this partitioning ([ca_topk_gs.cc:116-120](src/schedulers/ca_topk_gs.cc#L116-L120)):

```cpp
for (index_t i = 0; i < K_; ++i) {
    const index_t idx = pairs[pivot_pos + i].second;
    const index_t g = key(idx);
    d->groups[g].push_back(idx);
}
```

Optionally, sort each group by descending residual for greedy-like behavior within each group ([ca_topk_gs.cc:123-128](src/schedulers/ca_topk_gs.cc#L123-L128)):

```cpp
if (params_.sort_within_group) {
    for (size_t g = 0; g < G_; ++g) {
        std::sort(d->groups[g].begin(), d->groups[g].end(),
                  [&residuals](index_t a, index_t b) { return residuals[a] > residuals[b]; });
    }
}
```

### 5.3 Group Visiting Order

To prevent all threads from targeting group 0 first (which would recreate the contention problem), the group visiting order is a **random permutation** pi of {0, 1, ..., G-1}:

```
pi: {0, ..., G-1} -> {0, ..., G-1}   (bijection, shuffled each epoch)
```

The code ([ca_topk_gs.cc:131-140](src/schedulers/ca_topk_gs.cc#L131-L140)):

```cpp
d->group_order.resize(G_);
std::iota(d->group_order.begin(), d->group_order.end(), size_t{0});

std::mt19937_64 order_rng(epoch_seed + epoch_counter * 0xCAFEBABE);
std::shuffle(d->group_order.begin(), d->group_order.end(), order_rng);
```

The seed varies by epoch to avoid systematic bias across rebuilds.

### 5.4 Two-Level Lock-Free Dispatch

The `next(tid)` function implements a two-level atomic dispatch. The full dispatch logic ([ca_topk_gs.cc:175-210](src/schedulers/ca_topk_gs.cc#L175-L210)):

**Level 1: Group selection** via a global round-robin cursor:

```
t = group_rr_cursor.fetch_add(1)      // atomic, relaxed
g = pi[t mod G]                        // map to shuffled group
```

**Level 2: Index selection** within the chosen group via a per-group cursor:

```
k = group_cursor[g].fetch_add(1)      // atomic, relaxed
if k < |C_g|: return C_g[k]           // success
else: group exhausted, try next group
```

The code:

```cpp
for (size_t attempt = 0; attempt < G_; ++attempt) {
    const size_t t = d->group_rr_cursor.fetch_add(1, std::memory_order_relaxed);
    const size_t g = d->group_order[t % G_];

    const index_t k = d->group_cursor[g].fetch_add(1, std::memory_order_relaxed);
    if (k < static_cast<index_t>(d->groups[g].size())) {
        return d->groups[g][k];
    }
}
```

If all G groups are exhausted after G attempts, fall back to the per-thread shuffled blocks (coverage phase).

---

## 6. Mathematical Properties

### 6.1 Contention Reduction

In standard Top-K GS with a single cursor, the expected number of threads contending on the same cache line is T (all threads).

In CA-TopK-GS, the round-robin mechanism distributes threads across G cursors. The expected contention per cursor is:

```
E[contention per group cursor] = T / G
```

With the default G = 4T:

```
E[contention per group cursor] = T / (4T) = 0.25
```

This means on average, fewer than 1 thread contends on any given group cursor at any moment. The cache-coherence cost drops from O(T) to O(1) per `fetch_add`.

### 6.2 False Sharing Elimination

Two coordinates i and j cause false sharing on x[] when:

```
|i - j| < L / sizeof(real_t) = 64 / 8 = 8     (L = cache line size = 64 bytes)
```

In CA-TopK-GS, if threads A and B pick from different groups g_A != g_B, then any i in C_{g_A} and j in C_{g_B} satisfy:

```
kappa(i) != kappa(j)
=> floor(i/B) mod G != floor(j/B) mod G
```

Since B = 256 >> 8, coordinates from different groups are separated by at least B - 8 = 248 coordinates in the worst case (when blocks are adjacent modulo G), or typically by B = 256+ coordinates. At 8 bytes per double, this is 2048+ bytes apart, spanning 32+ cache lines. False sharing is eliminated.

### 6.3 Priority Preservation

**Claim**: CA-TopK-GS dispatches the same set of indices as Top-K GS (the hot set H), just in a different order.

**Proof**: The hot set H is identical (same nth_element selection). The only difference is the dispatch order: Top-K GS dispatches H[0], H[1], ..., H[K-1] in a fixed order, while CA-TopK-GS dispatches them interleaved across groups. Since all indices in H satisfy rho_i >= rho_{K-th largest}, the priority guarantee is preserved:

```
For all i dispatched in the priority phase:
    rho_i(x) >= min_{j in H} rho_j(x)
```

The within-group sorting (when `sort_within_group = true`) provides an additional greedy-like property within each group: within C_g, indices are dispatched in decreasing residual order.

### 6.4 Epoch Snapshot Isolation

The `data_` pointer is published atomically via `std::atomic_store_explicit` with `memory_order_release`, and loaded with `memory_order_acquire`. This provides the guarantee:

```
All writes to EpochData fields in rebuild()
    HAPPEN-BEFORE
all reads to EpochData fields in next()
```

Workers see a **consistent snapshot**: either the old epoch (all old groups, old cursors) or the new epoch (all new groups, reset cursors). No torn reads are possible.

The code ([ca_topk_gs.cc:172](src/schedulers/ca_topk_gs.cc#L172)):

```cpp
std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
```

And in dispatch ([ca_topk_gs.cc:176](src/schedulers/ca_topk_gs.cc#L176)):

```cpp
auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);
```

---

## 7. Complexity Analysis

### Per-Call Cost of `next()`

| Operation | Cost |
|-----------|------|
| `atomic_load(data_)` | O(1), acquire fence |
| Group selection: `fetch_add(group_rr_cursor)` | O(1), relaxed atomic |
| Permutation lookup: `group_order[t % G]` | O(1), array index |
| Index claim: `fetch_add(group_cursor[g])` | O(1), relaxed atomic |
| Bounds check and return | O(1) |

**Best case** (first group has indices): 1 iteration of the loop = O(1).

**Worst case** (all groups exhausted): G iterations + fallback lookup = O(G) = O(T).

**Amortized** over K priority dispatches + coverage: O(1) per call.

### Rebuild Cost

| Step | Cost |
|------|------|
| Build (residual, index) pairs | O(n) |
| `nth_element` for Top-K | O(n) average, O(n^2) worst |
| Assign K indices to G groups | O(K) |
| Sort within groups (optional) | O(sum_g |C_g| log |C_g|) <= O(K log K) |
| Shuffle group order | O(G) |
| Initialize fallback blocks | O(n) |

**Total rebuild**: O(n) + O(K log K) if sorting is enabled.

Since K << n (default is 1% of n), the O(n) pair construction and nth_element dominate.

### Space

| Component | Space |
|-----------|-------|
| Groups (hot set) | O(K) total across all groups |
| Group cursors | O(G) atomic integers |
| Group order permutation | O(G) |
| Fallback shuffled blocks | O(n) |
| Per-thread RNGs | O(T) |

**Total**: O(n + K + G + T) = O(n) since K, G, T << n.

---

## 8. The Top-K Selection Algorithm

The rebuild uses `std::nth_element` for O(n) average-case selection of the K largest residuals. The key insight is that we don't need the top K sorted; we only need them **separated** from the remaining n - K.

Mathematically, `nth_element` at position p = n - K produces a partition:

```
pairs[0..p-1]     all have residual <= pairs[p].residual
pairs[p]           is the K-th largest (the pivot)
pairs[p+1..n-1]    all have residual >= pairs[p].residual
```

The code ([ca_topk_gs.cc:96-100](src/schedulers/ca_topk_gs.cc#L96-L100)):

```cpp
const size_t pivot_pos = n_ - K_;
std::nth_element(pairs.begin(), pairs.begin() + static_cast<ptrdiff_t>(pivot_pos), pairs.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
```

This uses Introselect (hybrid of quickselect + median-of-medians) internally, giving O(n) average and O(n) worst case in modern standard library implementations.

After partitioning, the top-K indices are at positions `[pivot_pos, n)`:

```cpp
for (index_t i = 0; i < K_; ++i) {
    const index_t idx = pairs[pivot_pos + i].second;
    const index_t g = key(idx);
    d->groups[g].push_back(idx);
}
```

---

## 9. Dispatch State Machine

Each epoch, the scheduler transitions through these states:

```
                    rebuild()
                       |
                       v
              +------------------+
              |  PRIORITY PHASE  |  <-- group_rr_cursor < K (approximately)
              |                  |
              |  Round-robin     |
              |  across G groups |
              |  via group_order |
              +--------+---------+
                       |
                       | all G groups exhausted
                       v
              +------------------+
              |  COVERAGE PHASE  |  <-- per-thread shuffled blocks
              |                  |
              |  Thread t cycles |
              |  through its own |
              |  block partition |
              +--------+---------+
                       |
                       | rebuild() called again
                       v
              +------------------+
              |  PRIORITY PHASE  |  (new epoch, new hot set)
              |  (new epoch)     |
              +------------------+
```

The priority phase drains K hot indices across all threads. Once exhausted, the coverage phase ensures full-coverage sweeps until the next rebuild.

### Fallback Coverage

The coverage phase partitions [0, n) into T contiguous blocks and shuffles each:

```
Thread 0: sigma_0([0, n/T))         -- shuffled permutation
Thread 1: sigma_1([n/T, 2n/T))
...
Thread T-1: sigma_{T-1}([(T-1)n/T, n))
```

When a thread exhausts its shuffled block, it reshuffles and starts over ([ca_topk_gs.cc:212-218](src/schedulers/ca_topk_gs.cc#L212-L218)):

```cpp
void CATopKGSScheduler::reshuffle_fallback_(EpochData* d, size_t tid) {
    auto& indices = d->shuffled_indices[tid];
    if (indices.size() > 1) {
        std::shuffle(indices.begin(), indices.end(), d->rngs[tid]);
    }
    d->cursor[tid] = 0;
}
```

This guarantees every coordinate in [0, n) is visited at least once per epoch, which is necessary for convergence of the overall fixed-point iteration.

---

## 10. Comparison to Exact Gauss-Southwell

| Property | Exact GS | Top-K GS | CA-TopK-GS |
|----------|----------|----------|------------|
| Selection rule | argmax_i rho_i | Top-K set, any order | Top-K set, grouped by cache locality |
| Per-update cost | O(n) | O(1) amortized | O(1) amortized |
| Periodic cost | None | O(n) rebuild | O(n) + O(K log K) rebuild |
| Optimality | Exact greedy | Approximate greedy | Approximate greedy |
| Parallelism | Sequential only | T threads, 1 cursor | T threads, G cursors |
| False sharing | N/A (sequential) | Possible | Eliminated by construction |
| Contention | N/A (sequential) | O(T) per atomic | O(T/G) ~ O(1) per atomic |
| Coverage | No (greedy only) | Yes (fallback phase) | Yes (fallback phase) |

---

## 11. Parameter Selection

### K (Hot Set Size)

```
K = clamp(max(0.01 * n, 256 * T), 1, n)
```

- **0.01n**: Captures the top 1% of residuals. For well-conditioned problems this is often sufficient.
- **256T**: Ensures at least 256 priority dispatches per thread per rebuild, amortizing the O(n) rebuild cost.
- The rebuild runs every ~100-200ms. If each update takes ~10ns, each thread performs ~10M-20M updates between rebuilds, so K = 256T means the priority phase covers < 0.1% of all updates. The coverage phase dominates.

### G (Number of Groups)

```
G = 4 * T
```

- With G = 4T, the expected load per group is K / (4T) indices.
- The round-robin ensures each thread visits a different group on each call, spreading contention across 4T independent atomic counters.
- More groups = less contention, but more overhead in the priority phase loop (up to G iterations if many groups are empty).

### B (Block Size)

```
B = 256   (default)
```

- Each block spans B * sizeof(double) = 2048 bytes = 32 cache lines.
- Coordinates in the same block are cache-local (likely share L1/L2 pages).
- The modular key kappa(i) = floor(i/B) mod G creates a repeating stripe pattern with period B * G = 256 * 4T.
- For n >> B * G, the groups are roughly balanced in expectation (assuming hot indices are uniformly distributed across memory).

---

## 12. Worked Example

**Setup**: n = 2048, T = 2, K = 8, G = 8, B = 256.

**Residuals** (only showing the 8 largest):

| Index i | rho_i | floor(i/256) | kappa(i) = floor(i/256) mod 8 |
|---------|-------|-------------|-------------------------------|
| 42 | 5.2 | 0 | 0 |
| 103 | 4.8 | 0 | 0 |
| 300 | 4.5 | 1 | 1 |
| 515 | 4.1 | 2 | 2 |
| 520 | 3.9 | 2 | 2 |
| 800 | 3.7 | 3 | 3 |
| 1400 | 3.5 | 5 | 5 |
| 1900 | 3.2 | 7 | 7 |

**Group assignment** (after rebuild):

```
C_0 = {42, 103}    (sorted by residual: 42 first since rho_42 > rho_103)
C_1 = {300}
C_2 = {515, 520}   (sorted: 515 first)
C_3 = {800}
C_4 = {}            (empty)
C_5 = {1400}
C_6 = {}            (empty)
C_7 = {1900}
```

**Shuffled group order** pi = [5, 2, 7, 0, 3, 1, 6, 4].

**Dispatch trace** (2 threads, A and B):

```
Call 1 (Thread A): t=0, g=pi[0]=5, k=0 < |C_5|=1  -> return 1400
Call 2 (Thread B): t=1, g=pi[1]=2, k=0 < |C_2|=2  -> return 515
Call 3 (Thread A): t=2, g=pi[2]=7, k=0 < |C_7|=1  -> return 1900
Call 4 (Thread B): t=3, g=pi[3]=0, k=0 < |C_0|=2  -> return 42
Call 5 (Thread A): t=4, g=pi[4]=3, k=0 < |C_3|=1  -> return 800
Call 6 (Thread B): t=5, g=pi[5]=1, k=0 < |C_1|=1  -> return 300
Call 7 (Thread A): t=6, g=pi[6]=6, k=0 >= |C_6|=0 -> exhausted, retry
                   t=7, g=pi[7]=4, k=0 >= |C_4|=0 -> exhausted, retry
                   ... all 8 groups tried -> COVERAGE PHASE
                   -> return fallback shuffled block for Thread A
Call 8 (Thread B): similarly enters coverage phase
```

Note how Thread A and Thread B never contend on the same group cursor: A hits groups 5, 7, 3 while B hits groups 2, 0, 1. And the memory regions they touch are well-separated (1400 vs 515, 1900 vs 42, 800 vs 300 -- all >200 indices apart, so >50 cache lines apart).
