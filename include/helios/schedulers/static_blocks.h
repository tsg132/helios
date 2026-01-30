#pragma once

#include <string_view>
#include <vector>
#include "helios/scheduler.h"
#include "helios/types.h"

namespace helios {

class StaticBlocksScheduler final : public Scheduler {
public:
  void init(index_t n, size_t num_threads) override;
  index_t next(size_t tid) override;

  std::string_view name() const noexcept override { return "static_blocks"; }

private:
  index_t n_ = 0;
  size_t num_threads_ = 1;

  // Per-thread block boundaries: thread tid owns [block_begin_[tid], block_end_[tid])
  std::vector<index_t> block_begin_;
  std::vector<index_t> block_end_;

  // Per-thread cursor within its block
  std::vector<index_t> cursor_;
};

} // namespace helios
