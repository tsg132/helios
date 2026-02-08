#include "helios/cost_model.h"

#include <algorithm>

using namespace std;

namespace helios {

void populate_task_weights(EpochPlan& plan, const MDP* mdp) {
    if (!mdp) return;

    for (auto& phase : plan.phases) {
        for (auto& wl : phase.worklist) {
            for (auto& task : wl) {
                double w = 0.0;
                for (index_t i = task.begin; i < task.end; ++i) {
                    // Cost proxy: number of non-zeros in row i
                    w += static_cast<double>(mdp->row_ptr[i + 1] - mdp->row_ptr[i]);
                }
                task.weight = w;
            }
        }
    }
}

CostEstimate estimate_plan_cost(const EpochPlan& plan,
                                 const MDP* mdp,
                                 const CostModelConfig& cost_cfg) {
    CostEstimate est;

    // Populate weights from MDP if available and measured cost not provided
    // (We work with the weights already on the tasks)

    const size_t T = plan.threads;

    for (auto& phase : plan.phases) {
        // Compute per-thread cost for this phase
        double max_thread = 0.0;
        double total = 0.0;
        for (size_t tid = 0; tid < min(T, phase.worklist.size()); ++tid) {
            double thread_cost = 0.0;
            for (auto& task : phase.worklist[tid]) {
                if (cost_cfg.measured_update_ns > 0.0) {
                    // Use measured per-update cost
                    thread_cost += cost_cfg.measured_update_ns *
                                   static_cast<double>(task.size());
                } else if (task.weight > 0.0) {
                    thread_cost += task.weight;
                } else if (mdp) {
                    // Compute nnz proxy on the fly
                    for (index_t i = task.begin; i < task.end; ++i) {
                        thread_cost += static_cast<double>(
                            mdp->row_ptr[i + 1] - mdp->row_ptr[i]);
                    }
                } else {
                    thread_cost += static_cast<double>(task.size());
                }
            }
            total += thread_cost;
            max_thread = max(max_thread, thread_cost);
        }
        est.bottleneck_cost += max_thread;
        est.total_cost += total;

        if (phase.barrier_after) {
            est.barrier_penalty += cost_cfg.barrier_penalty;
        }
    }

    est.phase_penalty = cost_cfg.lambda * static_cast<double>(plan.phases.size());
    est.estimated_cost = est.bottleneck_cost + est.phase_penalty + est.barrier_penalty;

    return est;
}

} // namespace helios
