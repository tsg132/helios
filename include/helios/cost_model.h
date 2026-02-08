#pragma once

#include <cstdint>
#include <string>

#include "helios/types.h"
#include "helios/operator.h"
#include "helios/plan.h"
#include "helios/mdp.h"

using namespace std;

namespace helios {

//=============================================================================
// Cost Model: estimates the cost of executing an EpochPlan
//=============================================================================
// Cost of a BLOCK task = sum of nnz(row i) for i in [begin, end)
// (or a constant * block_len once avg_update_cost_ns is measured)
//
// Estimated plan cost = max_tid sum_{task in worklist[tid]} c(task)
//                       + lambda * num_phases
//                       + optional penalty for barriers

struct CostModelConfig {
    double lambda = 100.0;           // penalty per phase (nanoseconds)
    double barrier_penalty = 500.0;  // penalty per barrier
    double measured_update_ns = 0.0; // if > 0, use measured cost instead of nnz proxy
};

struct CostEstimate {
    double bottleneck_cost = 0.0;   // max thread cost (the real bottleneck)
    double total_cost = 0.0;        // sum of all thread costs
    double phase_penalty = 0.0;     // lambda * num_phases
    double barrier_penalty = 0.0;   // penalty for barriers
    double estimated_cost = 0.0;    // bottleneck + penalties

    string summary() const {
        string s;
        s += "CostEstimate: est=" + to_string(static_cast<uint64_t>(estimated_cost));
        s += " bottleneck=" + to_string(static_cast<uint64_t>(bottleneck_cost));
        s += " phase_pen=" + to_string(static_cast<uint64_t>(phase_penalty));
        s += " barrier_pen=" + to_string(static_cast<uint64_t>(barrier_penalty));
        return s;
    }
};

// Estimate cost of executing a plan given MDP sparsity
CostEstimate estimate_plan_cost(const EpochPlan& plan,
                                 const MDP* mdp,
                                 const CostModelConfig& cost_cfg);

// Populate task weights using MDP nnz info
void populate_task_weights(EpochPlan& plan, const MDP* mdp);

} // namespace helios
