#pragma once

#include <atomic>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

#include "helios/scheduler.h"
#include "helios/types.h"

namespace helios {

// Parameters for CATopKGSScheduler (Conflict-Aware Top-K Gauss-Southwell)
struct CATopKGSParams {
  // Hot set size. If 0, defaults to max(n * 0.01, num_threads * 256)
  index_t K = 0;

  // Number of conflict groups. If 0, defaults to 4 * num_threads
  size_t G = 0;

  // Block size for cache-block proxy key function. Indices in the same block
  // map to the same conflict group: key(i) = (i / block_size) % G
  index_t block_size = 256;

  // If true, sort indices within each group by descending residual
  bool sort_within_group = true;

  // Seed for group order shuffle and fallback (0 = use default deterministic seed)
  uint64_t seed = 0;
};

// CATopKGSScheduler: Conflict-Aware Top-K Gauss-Southwell priority scheduler.
//
// Extends Top-K GS by distributing hot indices into G conflict groups based on
// a key function (cache block proxy). Workers pick from groups in round-robin
// fashion to spread contention across cache lines.
//
// Algorithm:
// 1. REBUILD: Compute residuals, select Top-K, assign to conflict groups by key(i)
// 2. NEXT: Round-robin across groups, pop from each group's cursor, fallback when exhausted
//
// The conflict key function uses cache-block proxy: key(i) = (i / block_size) % G
// This groups indices that share cache blocks together, so different groups
// access different memory regions, reducing false sharing.
class CATopKGSScheduler final : public Scheduler {
public:
  using Params = CATopKGSParams;

  explicit CATopKGSScheduler(const Params& params = {});

  void init(index_t n, size_t num_threads) override;
  index_t next(size_t tid) override;

  // Rebuild hot set from current residuals. Called periodically by runtime.
  void rebuild(const std::vector<real_t>& residuals) override;

  bool supports_rebuild() const noexcept override { return true; }

  std::string_view name() const noexcept override { return "ca_topk_gs"; }

private:
  Params params_;
  index_t n_ = 0;
  size_t num_threads_ = 1;
  index_t K_ = 0;  // Actual hot set size after init
  size_t G_ = 0;   // Actual number of groups after init

  // Conflict key function: maps index to group
  index_t key(index_t i) const {
    return (i / params_.block_size) % static_cast<index_t>(G_);
  }

  // Epoch snapshot data (atomically published)
  struct EpochData {
    size_t num_groups = 0;

    // Conflict groups: groups[g] contains indices assigned to group g
    std::vector<std::vector<index_t>> groups;

    // Per-group cursor (next index to dispatch from each group)
    // Using unique_ptr to array since std::atomic is not move-constructible
    std::unique_ptr<std::atomic<index_t>[]> group_cursor;

    // Group visiting order (permutation of 0..G-1)
    std::vector<size_t> group_order;

    // Round-robin cursor over groups
    std::atomic<size_t> group_rr_cursor{0};

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
