#include "helios/autotune.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

using namespace std;

namespace helios {

using Clock = chrono::steady_clock;

AutotuneResult autotune(const Operator& op,
                         const MDP* mdp,
                         real_t* x,
                         const AutotuneConfig& atcfg) {
    AutotuneResult result;
    const auto t0 = Clock::now();

    const index_t n = op.n();
    const size_t T = max(atcfg.runtime_cfg.num_threads, size_t(1));

    // Take a snapshot of x for planner builds and pilot run restoration
    vector<real_t> x_snapshot(n);
    memcpy(x_snapshot.data(), x, n * sizeof(real_t));

    // Generate all candidates
    vector<AutotuneCandidate> candidates;

    for (auto blk : atcfg.blk_candidates) {
        if (blk > n) continue;

        // StaticPlanner candidate
        {
            AutotuneCandidate c;
            c.planner_name = "Static";
            c.planner_cfg.threads = T;
            c.planner_cfg.blk = blk;
            c.planner_cfg.colors = 0;
            c.planner_cfg.K = 0;
            c.planner_cfg.hot_phase_enabled = false;
            candidates.push_back(c);
        }

        // ColoredPlanner candidates
        for (auto cm : atcfg.color_multipliers) {
            AutotuneCandidate c;
            c.planner_name = "Colored";
            c.planner_cfg.threads = T;
            c.planner_cfg.blk = blk;
            c.planner_cfg.colors = static_cast<index_t>(cm * T);
            c.planner_cfg.K = 0;
            c.planner_cfg.hot_phase_enabled = false;
            c.planner_cfg.barrier_between_colors = false;
            candidates.push_back(c);
        }

        // PriorityPlanner candidates
        for (auto kf : atcfg.K_fractions) {
            index_t K = static_cast<index_t>(static_cast<double>(n) * kf);
            K = max(K, static_cast<index_t>(T * blk));
            K = min(K, n);

            for (auto cm : atcfg.color_multipliers) {
                AutotuneCandidate c;
                c.planner_name = "Priority";
                c.planner_cfg.threads = T;
                c.planner_cfg.blk = blk;
                c.planner_cfg.colors = static_cast<index_t>(cm * T);
                c.planner_cfg.K = K;
                c.planner_cfg.hot_phase_enabled = true;
                c.planner_cfg.barrier_between_colors = false;
                candidates.push_back(c);
            }
        }
    }

    // Build plans and estimate costs for all candidates
    CostModelConfig cost_cfg;

    for (auto& c : candidates) {
        EpochPlan plan;

        if (c.planner_name == "Static") {
            StaticPlanner planner;
            plan = planner.build(op, x_snapshot.data(), c.planner_cfg);
        } else if (c.planner_name == "Colored") {
            ColoredPlanner planner;
            plan = planner.build(op, x_snapshot.data(), c.planner_cfg);
        } else {
            PriorityPlanner planner;
            plan = planner.build(op, x_snapshot.data(), c.planner_cfg);
        }

        // Populate weights from MDP
        if (mdp) {
            populate_task_weights(plan, mdp);
        }

        c.cost_est = estimate_plan_cost(plan, mdp, cost_cfg);
    }

    // Sort by estimated cost (ascending)
    sort(candidates.begin(), candidates.end(),
         [](const AutotuneCandidate& a, const AutotuneCandidate& b) {
             return a.cost_est.estimated_cost < b.cost_est.estimated_cost;
         });

    // Take top M candidates for pilot runs
    const size_t M = min(atcfg.top_M, candidates.size());

    Runtime rt;

    for (size_t ci = 0; ci < M; ++ci) {
        auto& c = candidates[ci];

        // Restore x to snapshot
        memcpy(x, x_snapshot.data(), n * sizeof(real_t));

        // Build plan
        EpochPlan plan;
        if (c.planner_name == "Static") {
            StaticPlanner planner;
            plan = planner.build(op, x, c.planner_cfg);
        } else if (c.planner_name == "Colored") {
            ColoredPlanner planner;
            plan = planner.build(op, x, c.planner_cfg);
        } else {
            PriorityPlanner planner;
            plan = planner.build(op, x, c.planner_cfg);
        }

        if (mdp) populate_task_weights(plan, mdp);

        // Pilot run
        RuntimeConfig pilot_cfg = atcfg.runtime_cfg;
        pilot_cfg.max_seconds = atcfg.pilot_seconds;
        pilot_cfg.record_trace = false;

        real_t initial_residual = Runtime::residual_inf(op, x);

        RunResult pilot_result = rt.run_plan(op, plan, x, pilot_cfg);

        c.pilot_updates_per_sec = pilot_result.updates_per_sec;
        c.pilot_residual_drop = (pilot_result.final_residual_inf > 0.0)
            ? (initial_residual / pilot_result.final_residual_inf) : 1e12;
        c.piloted = true;
    }

    // Select best piloted candidate by residual drop rate (most convergence progress)
    size_t best_idx = 0;
    double best_score = 0.0;
    for (size_t ci = 0; ci < M; ++ci) {
        if (candidates[ci].piloted) {
            // Score = residual_drop * updates_per_sec (combined metric)
            double score = candidates[ci].pilot_residual_drop;
            if (score > best_score) {
                best_score = score;
                best_idx = ci;
            }
        }
    }

    // Restore x to snapshot for the real run
    memcpy(x, x_snapshot.data(), n * sizeof(real_t));

    // Build the best plan
    result.best = candidates[best_idx];
    if (result.best.planner_name == "Static") {
        StaticPlanner planner;
        result.best_plan = planner.build(op, x, result.best.planner_cfg);
    } else if (result.best.planner_name == "Colored") {
        ColoredPlanner planner;
        result.best_plan = planner.build(op, x, result.best.planner_cfg);
    } else {
        PriorityPlanner planner;
        result.best_plan = planner.build(op, x, result.best.planner_cfg);
    }

    if (mdp) populate_task_weights(result.best_plan, mdp);

    result.all_candidates = std::move(candidates);

    auto t1 = Clock::now();
    result.autotune_time_sec = chrono::duration<double>(t1 - t0).count();

    return result;
}

} // namespace helios
