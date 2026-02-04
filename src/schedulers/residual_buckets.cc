#include "helios/schedulers/residual_buckets.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace helios {

// Sentinel value indicating no valid index available
static constexpr index_t kInvalidIndex = std::numeric_limits<index_t>::max();

void ResidualBucketsScheduler::init(index_t n, size_t num_threads) {
    n_ = n;
    num_threads_ = num_threads;
    thread_bucket_hint_.assign(num_threads_, 0);

    auto d = std::make_shared<Data>();
    d->n = n_;
    d->B = params_.num_buckets;
    d->offsets.assign(d->B + 1, 0);

    // Construct cursor vector with correct size (atomics default to 0)
    d->cursor = std::vector<std::atomic<uint32_t>>(d->B);

    d->indices.clear();
    d->rr_cursor.store(0, std::memory_order_relaxed);
    d->all_tiny = true;

    std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}

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

void ResidualBucketsScheduler::rebuild(const std::vector<real_t>& residuals) {
    assert(static_cast<index_t>(residuals.size()) == n_);

    const uint32_t B = std::max<uint32_t>(1, params_.num_buckets);
    const real_t base = std::max<real_t>(params_.base, static_cast<real_t>(1e-300));

    // Pass 1: Count how many indices fall into each bucket
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

    // Build offsets array (exclusive prefix sum)
    d->offsets.resize(static_cast<size_t>(B) + 1);
    d->offsets[0] = 0;
    for (uint32_t b = 0; b < B; ++b) {
        d->offsets[b + 1] = d->offsets[b] + counts[b];
    }

    // Allocate indices array
    d->indices.resize(static_cast<size_t>(d->offsets[B]));

    // Construct cursor vector (each atomic initialized to 0)
    d->cursor = std::vector<std::atomic<uint32_t>>(static_cast<size_t>(B));

    d->rr_cursor.store(0, std::memory_order_relaxed);

    // Pass 2: Place each index into its bucket
    std::vector<uint32_t> write_ptr = d->offsets;

    for (index_t i = 0; i < n_; ++i) {
        const real_t r = residuals[i];
        const uint32_t b = bucket_of_(r, base, B);
        const uint32_t pos = write_ptr[b]++;
        d->indices[pos] = i;
    }

    // Atomically publish the new data structure
    std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}

index_t ResidualBucketsScheduler::next(size_t tid) {
    // Atomically load the current data snapshot
    auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);

    if (!d) return kInvalidIndex;

    // Fallback to round-robin if all residuals are tiny OR if buckets are exhausted
    if (params_.fallback_round_robin && d->all_tiny) {
        index_t i = d->rr_cursor.fetch_add(1, std::memory_order_relaxed);
        // Wrap around for continuous operation
        if (i >= d->n) {
            i = i % d->n;
        }
        return i;
    }

    const uint32_t B = d->B;
    if (B == 0) return kInvalidIndex;

    // Clamp thread id to valid range
    const size_t tt = std::min(tid, num_threads_ > 0 ? num_threads_ - 1 : 0);

    // Start searching from the thread's hint bucket
    uint32_t start_from = 0;
    if (tt < thread_bucket_hint_.size()) {
        start_from = thread_bucket_hint_[tt];
    }

    // Search buckets from highest priority to lowest
    for (uint32_t off = start_from; off < B; ++off) {
        // Map offset to bucket: highest bucket first
        const uint32_t b = (B - 1) - off;

        const uint32_t begin = d->offsets[b];
        const uint32_t end = d->offsets[b + 1];

        // Skip empty buckets
        if (begin == end) continue;

        // Atomically claim the next index in this bucket
        const uint32_t k = d->cursor[b].fetch_add(1, std::memory_order_relaxed);
        const uint32_t idx = begin + k;

        if (idx < end) {
            // Update hint for next call
            if (tt < thread_bucket_hint_.size()) {
                thread_bucket_hint_[tt] = off;
            }
            return d->indices[idx];
        }
        // Bucket exhausted, continue to next
    }

    // All buckets exhausted - fall back to round-robin for continuous operation
    // This allows the scheduler to keep running until rebuild() is called
    if (params_.fallback_round_robin) {
        index_t i = d->rr_cursor.fetch_add(1, std::memory_order_relaxed);
        return i % d->n;
    }

    return kInvalidIndex;
}

} // namespace helios
