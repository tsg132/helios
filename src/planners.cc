#include "helios/planner.h"

#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

namespace helios {

//=============================================================================
// StaticPlanner
//=============================================================================

EpochPlan StaticPlanner::build(const Operator& op,
                                const real_t* /*x_snapshot*/,
                                const PlannerConfig& cfg) {
    const index_t n = op.n();
    const index_t blk = max(cfg.blk, index_t(1));
    const size_t T = max(cfg.threads, size_t(1));

    // Build list of all blocks
    const index_t num_blocks = (n + blk - 1) / blk;

    // Assign blocks to threads round-robin
    Phase phase;
    phase.worklist.resize(T);
    phase.barrier_after = false;

    for (index_t b = 0; b < num_blocks; ++b) {
        const index_t begin = b * blk;
        const index_t end = min(begin + blk, n);
        const size_t tid = b % T;

        Task task;
        task.kind = TaskKind::BLOCK;
        task.begin = begin;
        task.end = end;
        task.weight = static_cast<double>(end - begin);
        task.conflict_key = b;

        phase.worklist[tid].push_back(task);
    }

    EpochPlan plan;
    plan.phases.push_back(std::move(phase));
    plan.n = n;
    plan.threads = T;
    plan.blk = blk;
    plan.colors = 0;
    plan.K = 0;
    plan.seed = cfg.seed;
    plan.built_from = "StaticPlanner";

    return plan;
}

//=============================================================================
// ColoredPlanner
//=============================================================================

EpochPlan ColoredPlanner::build(const Operator& op,
                                 const real_t* /*x_snapshot*/,
                                 const PlannerConfig& cfg) {
    const index_t n = op.n();
    const index_t blk = max(cfg.blk, index_t(1));
    const size_t T = max(cfg.threads, size_t(1));

    // Number of colors: if not specified, use T
    const index_t C = (cfg.colors > 0) ? cfg.colors : static_cast<index_t>(T);

    // Build list of all blocks
    const index_t num_blocks = (n + blk - 1) / blk;

    // Group blocks by color: color(block_id) = block_id % C
    // Create C phases, one per color
    EpochPlan plan;
    plan.phases.resize(C);
    for (index_t c = 0; c < C; ++c) {
        plan.phases[c].worklist.resize(T);
        plan.phases[c].barrier_after = cfg.barrier_between_colors;
    }

    // Assign each block to its color phase and a thread
    // Within each color, assign to threads round-robin
    vector<size_t> color_thread_cursor(C, 0);

    for (index_t b = 0; b < num_blocks; ++b) {
        const index_t begin = b * blk;
        const index_t end = min(begin + blk, n);
        const index_t color = b % C;

        Task task;
        task.kind = TaskKind::BLOCK;
        task.begin = begin;
        task.end = end;
        task.weight = static_cast<double>(end - begin);
        task.conflict_key = b;

        const size_t tid = color_thread_cursor[color] % T;
        color_thread_cursor[color]++;

        plan.phases[color].worklist[tid].push_back(task);
    }

    plan.n = n;
    plan.threads = T;
    plan.blk = blk;
    plan.colors = C;
    plan.K = 0;
    plan.seed = cfg.seed;
    plan.built_from = "ColoredPlanner";

    return plan;
}

//=============================================================================
// PriorityPlanner
//=============================================================================

EpochPlan PriorityPlanner::build(const Operator& op,
                                  const real_t* x_snapshot,
                                  const PlannerConfig& cfg) {
    const index_t n = op.n();
    const index_t blk = max(cfg.blk, index_t(1));
    const size_t T = max(cfg.threads, size_t(1));
    const index_t num_blocks = (n + blk - 1) / blk;

    // Number of colors for hot phase coloring
    const index_t C = (cfg.colors > 0) ? cfg.colors : static_cast<index_t>(T);

    // Determine K (number of hot indices)
    index_t K = cfg.K;
    if (K == 0) {
        // Auto: 2% of n, clamped to [T*blk, n]
        K = max(static_cast<index_t>(n * 0.02),
                static_cast<index_t>(T * blk));
        K = min(K, n);
    }

    // Compute per-block residual scores
    vector<double> block_score(num_blocks, 0.0);

    if (x_snapshot) {
        for (index_t b = 0; b < num_blocks; ++b) {
            const index_t begin = b * blk;
            const index_t end = min(begin + blk, n);
            double score = 0.0;
            for (index_t i = begin; i < end; ++i) {
                score += static_cast<double>(op.residual_i(i, x_snapshot));
            }
            block_score[b] = score;
        }
    }

    // Select hot blocks: top-K indices mapped to blocks, deduped
    // Sort blocks by descending score, take enough to cover K indices
    vector<index_t> block_order(num_blocks);
    iota(block_order.begin(), block_order.end(), 0);
    sort(block_order.begin(), block_order.end(),
         [&](index_t a, index_t b_idx) { return block_score[a] > block_score[b_idx]; });

    // Select hot blocks until we cover at least K indices
    vector<bool> is_hot(num_blocks, false);
    index_t hot_coverage = 0;
    index_t num_hot_blocks = 0;
    for (index_t bi = 0; bi < num_blocks && hot_coverage < K; ++bi) {
        index_t b = block_order[bi];
        is_hot[b] = true;
        const index_t begin = b * blk;
        const index_t end = min(begin + blk, n);
        hot_coverage += (end - begin);
        num_hot_blocks++;
    }

    EpochPlan plan;

    // Phase A (HOT): hot blocks only, optionally colored
    if (cfg.hot_phase_enabled && num_hot_blocks > 0) {
        // Color hot blocks: hot_block_id % C
        vector<Phase> hot_phases(C);
        for (index_t c = 0; c < C; ++c) {
            hot_phases[c].worklist.resize(T);
            hot_phases[c].barrier_after = cfg.barrier_between_colors;
        }

        vector<size_t> color_cursor(C, 0);
        // Process hot blocks in score-descending order
        for (index_t bi = 0; bi < num_blocks; ++bi) {
            index_t b = block_order[bi];
            if (!is_hot[b]) continue;

            const index_t begin = b * blk;
            const index_t end = min(begin + blk, n);
            const index_t color = b % C;

            Task task;
            task.kind = TaskKind::BLOCK;
            task.begin = begin;
            task.end = end;
            task.weight = block_score[b];
            task.conflict_key = b;

            const size_t tid = color_cursor[color] % T;
            color_cursor[color]++;

            hot_phases[color].worklist[tid].push_back(task);
        }

        // Only add phases that have work
        for (index_t c = 0; c < C; ++c) {
            bool has_work = false;
            for (size_t tid = 0; tid < T; ++tid) {
                if (!hot_phases[c].worklist[tid].empty()) {
                    has_work = true;
                    break;
                }
            }
            if (has_work) {
                plan.phases.push_back(std::move(hot_phases[c]));
            }
        }
    }

    // Phase B (COVERAGE): full set of blocks (all blocks, colored)
    {
        vector<Phase> cov_phases(C);
        for (index_t c = 0; c < C; ++c) {
            cov_phases[c].worklist.resize(T);
            cov_phases[c].barrier_after = cfg.barrier_between_colors;
        }

        vector<size_t> cov_cursor(C, 0);
        for (index_t b = 0; b < num_blocks; ++b) {
            const index_t begin = b * blk;
            const index_t end = min(begin + blk, n);
            const index_t color = b % C;

            Task task;
            task.kind = TaskKind::BLOCK;
            task.begin = begin;
            task.end = end;
            task.weight = static_cast<double>(end - begin);
            task.conflict_key = b;

            const size_t tid = cov_cursor[color] % T;
            cov_cursor[color]++;

            cov_phases[color].worklist[tid].push_back(task);
        }

        for (index_t c = 0; c < C; ++c) {
            bool has_work = false;
            for (size_t tid = 0; tid < T; ++tid) {
                if (!cov_phases[c].worklist[tid].empty()) {
                    has_work = true;
                    break;
                }
            }
            if (has_work) {
                plan.phases.push_back(std::move(cov_phases[c]));
            }
        }
    }

    plan.n = n;
    plan.threads = T;
    plan.blk = blk;
    plan.colors = C;
    plan.K = K;
    plan.seed = cfg.seed;
    plan.built_from = "PriorityPlanner";

    return plan;
}

} // namespace helios
