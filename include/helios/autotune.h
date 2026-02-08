#pragma once

#include <string>
#include <vector>

#include "helios/types.h"
#include "helios/operator.h"
#include "helios/plan.h"
#include "helios/planner.h"
#include "helios/cost_model.h"
#include "helios/runtime.h"
#include "helios/mdp.h"

using namespace std;

namespace helios {

//=============================================================================
// AutotuneConfig: parameters for the autotuner
//=============================================================================
struct AutotuneConfig {
    // Candidate grid
    vector<index_t> blk_candidates   = {64, 128, 256, 512, 1024};
    vector<index_t> color_multipliers = {1, 2, 4};  // colors = mult * threads
    vector<double>  K_fractions       = {0.005, 0.01, 0.02, 0.05}; // fraction of n
    vector<size_t>  rebuild_ms_candidates = {50, 100, 200};

    // Pilot run parameters
    double pilot_seconds = 0.5;   // how long each pilot run lasts
    size_t top_M = 3;             // number of top candidates to pilot

    // Runtime config template (threads, alpha, eps, etc. come from here)
    RuntimeConfig runtime_cfg;
};

//=============================================================================
// AutotuneResult: the selected configuration + pilot results
//=============================================================================
struct AutotuneCandidate {
    PlannerConfig planner_cfg;
    string planner_name;         // "Static", "Colored", "Priority"
    CostEstimate cost_est;
    double pilot_updates_per_sec = 0.0;
    double pilot_residual_drop = 0.0;  // initial_residual / final_residual
    bool piloted = false;
};

struct AutotuneResult {
    AutotuneCandidate best;
    EpochPlan best_plan;
    vector<AutotuneCandidate> all_candidates;
    double autotune_time_sec = 0.0;

    string summary() const {
        string s;
        s += "Autotune: selected " + best.planner_name;
        s += " blk=" + to_string(best.planner_cfg.blk);
        s += " colors=" + to_string(best.planner_cfg.colors);
        s += " K=" + to_string(best.planner_cfg.K);
        s += " est_cost=" + to_string(static_cast<uint64_t>(best.cost_est.estimated_cost));
        if (best.piloted) {
            s += " pilot_ups=" + to_string(static_cast<uint64_t>(best.pilot_updates_per_sec));
        }
        s += " autotune_time=" + to_string(autotune_time_sec) + "s";
        s += "\n";
        return s;
    }
};

//=============================================================================
// autotune: select best planner config via cost model + pilot runs
//=============================================================================
AutotuneResult autotune(const Operator& op,
                         const MDP* mdp,
                         real_t* x,
                         const AutotuneConfig& atcfg);

} // namespace helios
