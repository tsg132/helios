#pragma once

#include <string>
#include <vector>

#include "helios/types.h"
#include "helios/operator.h"
#include "helios/plan.h"

using namespace std;

namespace helios {

//=============================================================================
// PlannerConfig: parameters for plan compilation
//=============================================================================
struct PlannerConfig {
    size_t   threads = 1;                // T: number of worker threads
    index_t  blk = 256;                  // block size for grouping coordinates
    index_t  colors = 0;                 // C: number of colors (0 = auto = threads)
    index_t  K = 0;                      // top-K hot set size (0 = auto)
    bool     hot_phase_enabled = true;   // enable priority/hot phase
    bool     barrier_between_colors = false; // barrier after each color phase
    uint64_t seed = 42;                  // random seed
};

//=============================================================================
// Planner: abstract interface for schedule compilation
//=============================================================================
// A Planner takes an Operator + optional snapshot of x + config and produces
// an EpochPlan (the compiled schedule).
class Planner {
public:
    virtual ~Planner() = default;

    virtual EpochPlan build(const Operator& op,
                            const real_t* x_snapshot,
                            const PlannerConfig& cfg) = 0;

    virtual string_view name() const noexcept = 0;
};

//=============================================================================
// StaticPlanner: baseline - partition into blocks, assign to threads
//=============================================================================
// Single phase, no coloring, no priority. Equivalent to StaticBlocksScheduler.
class StaticPlanner final : public Planner {
public:
    EpochPlan build(const Operator& op,
                    const real_t* x_snapshot,
                    const PlannerConfig& cfg) override;

    string_view name() const noexcept override { return "StaticPlanner"; }
};

//=============================================================================
// ColoredPlanner: conflict-aware cache-block coloring
//=============================================================================
// Assigns blocks to colors via block_id % C, then creates C phases.
// Within each phase, only blocks of one color are active, reducing
// cache-line conflicts between threads.
class ColoredPlanner final : public Planner {
public:
    EpochPlan build(const Operator& op,
                    const real_t* x_snapshot,
                    const PlannerConfig& cfg) override;

    string_view name() const noexcept override { return "ColoredPlanner"; }
};

//=============================================================================
// PriorityPlanner: Top-K compiled hot phase + coverage phase
//=============================================================================
// Takes a residual snapshot, selects top-K high-residual blocks as "hot",
// emits a HOT phase (optionally colored) followed by a COVERAGE phase (full).
class PriorityPlanner final : public Planner {
public:
    EpochPlan build(const Operator& op,
                    const real_t* x_snapshot,
                    const PlannerConfig& cfg) override;

    string_view name() const noexcept override { return "PriorityPlanner"; }
};

} // namespace helios
