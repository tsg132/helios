#pragma once

#include <atomic>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

#include "helios/scheduler.h"
#include "helios/types.h"

namespace helios {

// TopKGSScheduler: Top-K Gauss-Southwell priority scheduler.
//
// Approximates Gauss-Southwell coordinate selection by maintaining a "hot set"
// of the K coordinates with largest residuals. Workers consume the hot set first
// (priority phase), then fall back to a shuffled blocks schedule (coverage phase)
// to ensure all coordinates are updated.
//
// The hot set is rebuilt periodically via rebuild() called by the runtime monitor.
// Parameters for TopKGSScheduler
struct TopKGSParams {
  // Hot set size. If 0, defaults to max(n * 0.01, num_threads * 256)
  index_t K = 0;

  // If true, sort hot set by descending residual for closer-to-greedy order
  bool sort_hot = false;

  // Seed for fallback shuffled blocks (0 = use default deterministic seed)
  uint64_t seed = 0;
};

class TopKGSScheduler final : public Scheduler {
public:
  using Params = TopKGSParams;

  explicit TopKGSScheduler(const Params& params = {});

  void init(index_t n, size_t num_threads) override;
  index_t next(size_t tid) override;

  // Rebuild hot set from current residuals. Called periodically by runtime.
  void rebuild(const std::vector<real_t>& residuals) override;

  bool supports_rebuild() const noexcept override { return true; }

  std::string_view name() const noexcept override { return "topk_gs"; }

private:
  Params params_;
  index_t n_ = 0;
  size_t num_threads_ = 1;
  index_t K_ = 0;  // Actual hot set size after init

  // Epoch snapshot data (atomically published)
  struct EpochData {
    std::vector<index_t> hot;              // Top-K indices by residual
    std::atomic<index_t> hot_cursor{0};    // Next index to dispatch from hot set

    // Fallback: shuffled blocks state
    std::vector<std::vector<index_t>> shuffled_indices;  // Per-thread shuffled block
    std::vector<size_t> cursor;                          // Per-thread cursor
    std::vector<std::mt19937_64> rngs;                   // Per-thread RNGs
  };

  std::shared_ptr<EpochData> data_;

  // Reshuffle fallback indices for a thread
  void reshuffle_fallback_(EpochData* d, size_t tid);
};

} // namespace helios
