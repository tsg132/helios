#include "helios/schedulers/shuffled_blocks.h"

#include <algorithm>
#include <numeric>

namespace helios {

void ShuffledBlocksScheduler::init(index_t n, size_t num_threads) {
  n_ = n;
  num_threads_ = (num_threads > 0) ? num_threads : 1;

  // Partition [0, n) into num_threads contiguous blocks
  block_begin_.resize(num_threads_);
  block_end_.resize(num_threads_);
  shuffled_indices_.resize(num_threads_);
  cursor_.resize(num_threads_);
  rngs_.resize(num_threads_);

  const index_t base_size = n_ / static_cast<index_t>(num_threads_);
  const index_t remainder = n_ % static_cast<index_t>(num_threads_);

  index_t start = 0;
  for (size_t tid = 0; tid < num_threads_; ++tid) {
    // First 'remainder' threads get one extra element
    const index_t size = base_size + (tid < remainder ? 1 : 0);
    block_begin_[tid] = start;
    block_end_[tid] = start + size;

    // Initialize RNG for this thread with a seed based on tid
    rngs_[tid].seed(static_cast<uint64_t>(tid) * 0x9E3779B97F4A7C15ULL + 42);

    // Initialize shuffled indices for this thread
    shuffled_indices_[tid].resize(size);
    std::iota(shuffled_indices_[tid].begin(), shuffled_indices_[tid].end(), start);

    // Initial shuffle
    if (size > 1) {
      std::shuffle(shuffled_indices_[tid].begin(), shuffled_indices_[tid].end(), rngs_[tid]);
    }

    cursor_[tid] = 0;
    start += size;
  }
}

index_t ShuffledBlocksScheduler::next(size_t tid) {
  if (tid >= num_threads_ || n_ == 0) return n_;  // Invalid: return sentinel

  auto& indices = shuffled_indices_[tid];
  if (indices.empty()) return n_;  // Empty block for this thread

  const size_t pos = cursor_[tid];
  const index_t i = indices[pos];

  // Advance cursor
  cursor_[tid] = pos + 1;

  // If we've completed an epoch, reshuffle for next epoch
  if (cursor_[tid] >= indices.size()) {
    reshuffle(tid);
  }

  return i;
}

void ShuffledBlocksScheduler::reshuffle(size_t tid) {
  auto& indices = shuffled_indices_[tid];
  if (indices.size() > 1) {
    std::shuffle(indices.begin(), indices.end(), rngs_[tid]);
  }
  cursor_[tid] = 0;
}

} // namespace helios
