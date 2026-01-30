#include "helios/schedulers/static_blocks.h"

namespace helios {

void StaticBlocksScheduler::init(index_t n, size_t num_threads) {
  n_ = n;
  num_threads_ = (num_threads > 0) ? num_threads : 1;

  // Partition [0, n) into num_threads contiguous blocks
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

index_t StaticBlocksScheduler::next(size_t tid) {
  if (tid >= num_threads_ || n_ == 0) return n_;  // Invalid: return sentinel

  const index_t begin = block_begin_[tid];
  const index_t end = block_end_[tid];

  if (begin >= end) return n_;  // Empty block for this thread

  const index_t i = cursor_[tid];

  // Advance cursor, wrapping around within the block
  cursor_[tid] = (i + 1 < end) ? (i + 1) : begin;

  return i;
}

} // namespace helios