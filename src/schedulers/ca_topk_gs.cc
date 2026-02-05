#include "helios/schedulers/ca_topk_gs.h"

#include <algorithm>
#include <numeric>

namespace helios {

CATopKGSScheduler::CATopKGSScheduler(const Params& params) : params_(params) {}

void CATopKGSScheduler::init(index_t n, size_t num_threads) {
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

  // Compute G: if params_.G == 0, default to 4 * num_threads
  if (params_.G == 0) {
    G_ = 4 * num_threads_;
  } else {
    G_ = params_.G;
  }
  // Ensure G_ >= 1
  G_ = std::max(G_, size_t{1});

  // Create initial epoch data with empty groups and initialized fallback
  auto d = std::make_shared<EpochData>();
  d->num_groups = G_;

  // Initialize conflict groups (empty initially)
  d->groups.resize(G_);
  d->group_cursor = std::make_unique<std::atomic<index_t>[]>(G_);
  for (size_t g = 0; g < G_; ++g) {
    d->group_cursor[g].store(0, std::memory_order_relaxed);
  }

  // Initialize group order (identity permutation initially)
  d->group_order.resize(G_);
  std::iota(d->group_order.begin(), d->group_order.end(), size_t{0});

  d->group_rr_cursor.store(0, std::memory_order_relaxed);

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

void CATopKGSScheduler::rebuild(const std::vector<real_t>& residuals) {
  if (n_ == 0 || residuals.size() < n_) return;

  // Build (residual, index) pairs
  std::vector<std::pair<real_t, index_t>> pairs(n_);
  for (index_t i = 0; i < n_; ++i) {
    pairs[i] = {residuals[i], i};
  }

  // Use nth_element to partition: largest K elements at the end
  // We want the K largest, so partition at position (n_ - K_)
  const size_t pivot_pos = n_ - K_;
  std::nth_element(pairs.begin(), pairs.begin() + static_cast<ptrdiff_t>(pivot_pos), pairs.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create new epoch data
  auto d = std::make_shared<EpochData>();
  d->num_groups = G_;

  // Initialize conflict groups
  d->groups.resize(G_);
  d->group_cursor = std::make_unique<std::atomic<index_t>[]>(G_);
  for (size_t g = 0; g < G_; ++g) {
    d->groups[g].clear();
    d->groups[g].reserve(K_ / G_ + 1);  // Approximate expected size
    d->group_cursor[g].store(0, std::memory_order_relaxed);
  }

  // Extract top-K indices and assign to conflict groups
  for (index_t i = 0; i < K_; ++i) {
    const index_t idx = pairs[pivot_pos + i].second;
    const index_t g = key(idx);
    d->groups[g].push_back(idx);
  }

  // Optionally sort within each group by descending residual
  if (params_.sort_within_group) {
    for (size_t g = 0; g < G_; ++g) {
      std::sort(d->groups[g].begin(), d->groups[g].end(),
                [&residuals](index_t a, index_t b) { return residuals[a] > residuals[b]; });
    }
  }

  // Build group visiting order (shuffled permutation)
  d->group_order.resize(G_);
  std::iota(d->group_order.begin(), d->group_order.end(), size_t{0});

  // Use a different seed each epoch to vary the group order
  auto old_data = std::atomic_load_explicit(&data_, std::memory_order_acquire);
  const uint64_t epoch_seed = (params_.seed != 0) ? params_.seed : 42;
  const uint64_t epoch_counter = old_data ? old_data->group_rr_cursor.load(std::memory_order_relaxed) : 0;

  std::mt19937_64 order_rng(epoch_seed + epoch_counter * 0xCAFEBABE);
  std::shuffle(d->group_order.begin(), d->group_order.end(), order_rng);

  d->group_rr_cursor.store(0, std::memory_order_relaxed);

  // Initialize fallback shuffled blocks (same as init)
  d->shuffled_indices.resize(num_threads_);
  d->cursor.resize(num_threads_);
  d->rngs.resize(num_threads_);

  const index_t base_size = n_ / static_cast<index_t>(num_threads_);
  const index_t remainder = n_ % static_cast<index_t>(num_threads_);

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

index_t CATopKGSScheduler::next(size_t tid) {
  auto d = std::atomic_load_explicit(&data_, std::memory_order_acquire);
  if (!d || n_ == 0) return n_;  // Sentinel for invalid

  // Priority phase: try to pop from conflict-aware Top-K groups
  // Round-robin across groups to spread contention
  for (size_t attempt = 0; attempt < G_; ++attempt) {
    // Get next group in round-robin order
    const size_t t = d->group_rr_cursor.fetch_add(1, std::memory_order_relaxed);
    const size_t g = d->group_order[t % G_];

    // Try to pop from this group
    const index_t k = d->group_cursor[g].fetch_add(1, std::memory_order_relaxed);
    if (k < static_cast<index_t>(d->groups[g].size())) {
      return d->groups[g][k];
    }
    // Group exhausted, try next group
  }

  // Coverage phase: all groups exhausted, fall back to shuffled blocks
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

void CATopKGSScheduler::reshuffle_fallback_(EpochData* d, size_t tid) {
  auto& indices = d->shuffled_indices[tid];
  if (indices.size() > 1) {
    std::shuffle(indices.begin(), indices.end(), d->rngs[tid]);
  }
  d->cursor[tid] = 0;
}

} // namespace helios
