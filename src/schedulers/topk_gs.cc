#include "helios/schedulers/topk_gs.h"

#include <algorithm>
#include <numeric>

namespace helios {

TopKGSScheduler::TopKGSScheduler(const Params& params) : params_(params) {}

void TopKGSScheduler::init(index_t n, size_t num_threads) {
  n_ = n;
  num_threads_ = (num_threads > 0) ? num_threads : 1;

  // Compute K: if params_.K == 0, use default formula
  if (params_.K == 0) {
    // K = max(n * 0.01, num_threads * 256), clamped to [1, n]
    const index_t k_pct = static_cast<index_t>(n_ * 0.01);
    const index_t k_threads = static_cast<index_t>(num_threads_ * 256);
    K_ = std::max(k_pct, k_threads);
    K_ = std::max(K_, index_t{1});
    K_ = std::min(K_, n_);
  } else {
    K_ = std::min(params_.K, n_);
  }

  // Create initial epoch data with empty hot set and initialized fallback
  auto d = std::make_shared<EpochData>();
  d->hot.reserve(K_);
  d->hot_cursor.store(0, std::memory_order_relaxed);

  // Initialize fallback shuffled blocks
  d->shuffled_indices.resize(num_threads_);
  d->cursor.resize(num_threads_);
  d->rngs.resize(num_threads_);

  // Partition [0, n) into contiguous blocks per thread
  const index_t base_size = n_ / static_cast<index_t>(num_threads_);
  const index_t remainder = n_ % static_cast<index_t>(num_threads_);

  // Base seed for RNGs
  const uint64_t base_seed = (params_.seed != 0) ? params_.seed : 42;

  index_t start = 0;
  for (size_t tid = 0; tid < num_threads_; ++tid) {
    const index_t size = base_size + (tid < remainder ? 1 : 0);

    // Initialize RNG for this thread
    d->rngs[tid].seed(base_seed + tid * 0x9E3779B97F4A7C15ULL);

    // Fill and shuffle indices for this thread's block
    d->shuffled_indices[tid].resize(size);
    std::iota(d->shuffled_indices[tid].begin(), d->shuffled_indices[tid].end(), start);

    if (size > 1) {
      std::shuffle(d->shuffled_indices[tid].begin(), d->shuffled_indices[tid].end(),
                   d->rngs[tid]);
    }

    d->cursor[tid] = 0;
    start += size;
  }

  data_ = std::move(d);
}

void TopKGSScheduler::rebuild(const std::vector<real_t>& residuals) {
  if (n_ == 0 || residuals.size() < n_) return;

  // Build (residual, index) pairs
  std::vector<std::pair<real_t, index_t>> pairs(n_);
  for (index_t i = 0; i < n_; ++i) {
    pairs[i] = {residuals[i], i};
  }

  // Use nth_element to partition: largest K elements at the end
  // We want the K largest, so partition at position (n_ - K_)
  const size_t pivot_pos = n_ - K_;
  std::nth_element(pairs.begin(), pairs.begin() + pivot_pos, pairs.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create new epoch data
  auto d = std::make_shared<EpochData>();
  d->hot.resize(K_);

  // Extract top-K indices (from pivot_pos to end)
  for (index_t i = 0; i < K_; ++i) {
    d->hot[i] = pairs[pivot_pos + i].second;
  }

  // Optionally sort hot set by descending residual for closer-to-greedy order
  if (params_.sort_hot) {
    // We need to re-fetch residuals for sorting since pairs order is arbitrary after nth_element
    std::sort(d->hot.begin(), d->hot.end(),
              [&residuals](index_t a, index_t b) { return residuals[a] > residuals[b]; });
  }

  d->hot_cursor.store(0, std::memory_order_relaxed);

  // Initialize fallback shuffled blocks (same as init)
  d->shuffled_indices.resize(num_threads_);
  d->cursor.resize(num_threads_);
  d->rngs.resize(num_threads_);

  const index_t base_size = n_ / static_cast<index_t>(num_threads_);
  const index_t remainder = n_ % static_cast<index_t>(num_threads_);

  // Use a different seed each epoch to vary the shuffle
  // We use a simple counter based on current hot_cursor of old data (or 0)
  auto old_data = std::atomic_load_explicit(&data_, std::memory_order_acquire);
  const uint64_t epoch_seed = (params_.seed != 0) ? params_.seed : 42;
  const uint64_t epoch_counter = old_data ? old_data->hot_cursor.load(std::memory_order_relaxed) : 0;

  index_t start = 0;
  for (size_t tid = 0; tid < num_threads_; ++tid) {
    const index_t size = base_size + (tid < remainder ? 1 : 0);

    // Seed with epoch variation
    d->rngs[tid].seed(epoch_seed + tid * 0x9E3779B97F4A7C15ULL + epoch_counter * 0xDEADBEEF);

    d->shuffled_indices[tid].resize(size);
    std::iota(d->shuffled_indices[tid].begin(), d->shuffled_indices[tid].end(), start);

    if (size > 1) {
      std::shuffle(d->shuffled_indices[tid].begin(), d->shuffled_indices[tid].end(),
                   d->rngs[tid]);
    }

    d->cursor[tid] = 0;
    start += size;
  }

  // Atomically publish new epoch data
  std::atomic_store_explicit(&data_, std::move(d), std::memory_order_release);
}

index_t TopKGSScheduler::next(size_t tid) {
  auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);
  if (!d || n_ == 0) return n_;  // Sentinel for invalid

  // Priority phase: try to get from hot set
  const index_t k = d->hot_cursor.fetch_add(1, std::memory_order_relaxed);
  if (k < K_ && k < static_cast<index_t>(d->hot.size())) {
    return d->hot[k];
  }

  // Coverage phase: fall back to shuffled blocks
  if (tid >= num_threads_) tid = num_threads_ - 1;

  auto& indices = d->shuffled_indices[tid];
  if (indices.empty()) return n_;

  const size_t pos = d->cursor[tid];
  const index_t i = indices[pos];

  // Advance cursor, reshuffle at epoch boundary
  d->cursor[tid] = pos + 1;
  if (d->cursor[tid] >= indices.size()) {
    reshuffle_fallback_(d.get(), tid);
  }

  return i;
}

void TopKGSScheduler::reshuffle_fallback_(EpochData* d, size_t tid) {
  auto& indices = d->shuffled_indices[tid];
  if (indices.size() > 1) {
    std::shuffle(indices.begin(), indices.end(), d->rngs[tid]);
  }
  d->cursor[tid] = 0;
}

} // namespace helios
