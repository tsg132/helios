#pragma once

#include <string_view>
#include "helios/scheduler.h"
#include "helios/types.h"

namespace helios {

class StaticBlocksScheduler final : public Scheduler {
public:
  void init(index_t n, int num_threads) override;
  index_t next(int tid) override;

  std::string_view name() const noexcept override { return "static_blocks"; }

private:
  index_t n_ = 0;
  int num_threads_ = 1;

  // Per-thread cursor within its block
  index_t block_begin_ = 0; // not used in this simple stub; kept for future
};

} // namespace helios