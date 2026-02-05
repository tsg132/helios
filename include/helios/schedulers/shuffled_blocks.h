#pragma once

#include <random>
#include <string_view>
#include <vector>
#include "helios/scheduler.h"
#include "helios/types.h"

namespace helios {

// ShuffledBlocksScheduler: each thread owns a contiguous block of indices,
// but iterates through them in shuffled order. At each epoch (full pass
// through the block), the order is reshuffled.
class ShuffledBlocksScheduler final : public Scheduler {
public:
  void init(index_t n, size_t num_threads) override;
  index_t next(size_t tid) override;

  std::string_view name() const noexcept override { return "shuffled_blocks"; }

private:
  index_t n_ = 0;
  size_t num_threads_ = 1;

  // Per-thread block boundaries: thread tid owns indices in [block_begin_[tid], block_end_[tid])
  std::vector<index_t> block_begin_;
  std::vector<index_t> block_end_;

  // Per-thread shuffled order of indices within the block
  // shuffled_indices_[tid] contains the indices owned by thread tid in shuffled order
  std::vector<std::vector<index_t>> shuffled_indices_;

  // Per-thread cursor within the shuffled order
  std::vector<size_t> cursor_;

  // Per-thread random number generators for reshuffling
  std::vector<std::mt19937_64> rngs_;

  // Reshuffle the indices for a given thread
  void reshuffle(size_t tid);
};

} // namespace helios
