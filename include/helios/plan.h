#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "helios/types.h"

using namespace std;

namespace helios {

//=============================================================================
// Schedule IR: Task / Phase / EpochPlan / ScheduleProgram
//=============================================================================
// The "compiled schedule" representation. A Planner produces an EpochPlan
// from operator sparsity + optional residual snapshot + hardware knobs.
// The Runtime plan executor consumes EpochPlans with minimal atomics.

//-----------------------------------------------------------------------------
// Task: a unit of work = one or more coordinate updates
//-----------------------------------------------------------------------------
enum class TaskKind : uint8_t {
    ONE,    // Single coordinate update (begin only, end = begin+1)
    BLOCK   // Block of contiguous coordinates [begin, end)
};

struct Task {
    TaskKind kind = TaskKind::BLOCK;
    index_t begin = 0;
    index_t end   = 0;         // end = begin+1 for ONE
    double  weight = 0.0;      // optional: estimated cost (e.g., sum of nnz)
    uint32_t conflict_key = 0; // optional: cache-block proxy for conflict avoidance

    index_t size() const noexcept { return end - begin; }
};

//-----------------------------------------------------------------------------
// Phase: one parallel step = per-thread worklists + optional barrier
//-----------------------------------------------------------------------------
struct Phase {
    // worklist[tid] = vector of Tasks for thread tid
    vector<vector<Task>> worklist;   // size = num_threads

    bool barrier_after = false;      // if true, all threads sync after this phase

    // Total number of coordinate updates in this phase
    uint64_t total_updates() const noexcept {
        uint64_t sum = 0;
        for (auto& wl : worklist) {
            for (auto& t : wl) sum += t.size();
        }
        return sum;
    }

    // Max load across threads (bottleneck)
    uint64_t max_thread_updates() const noexcept {
        uint64_t mx = 0;
        for (auto& wl : worklist) {
            uint64_t s = 0;
            for (auto& t : wl) s += t.size();
            if (s > mx) mx = s;
        }
        return mx;
    }

    // Max weighted load across threads
    double max_thread_weight() const noexcept {
        double mx = 0.0;
        for (auto& wl : worklist) {
            double s = 0.0;
            for (auto& t : wl) s += t.weight;
            if (s > mx) mx = s;
        }
        return mx;
    }
};

//-----------------------------------------------------------------------------
// EpochPlan: a complete plan for one "epoch" of computation
//-----------------------------------------------------------------------------
struct EpochPlan {
    vector<Phase> phases;

    // Metadata
    index_t n = 0;              // problem dimension
    size_t  threads = 0;        // number of threads
    index_t blk = 0;            // block size used
    index_t colors = 0;         // number of colors (0 = no coloring)
    index_t K = 0;              // top-K size (0 = no priority phase)
    uint64_t seed = 0;          // random seed used
    string built_from;          // description of planner that built this

    // Total updates across all phases
    uint64_t total_updates() const noexcept {
        uint64_t sum = 0;
        for (auto& p : phases) sum += p.total_updates();
        return sum;
    }

    // Print a summary of the plan
    string summary() const {
        string s;
        s += "EpochPlan[" + built_from + "]: ";
        s += "n=" + to_string(n) + " T=" + to_string(threads);
        s += " blk=" + to_string(blk);
        if (colors > 0) s += " colors=" + to_string(colors);
        if (K > 0) s += " K=" + to_string(K);
        s += " phases=" + to_string(phases.size());
        s += " updates=" + to_string(total_updates());
        s += "\n";
        for (size_t pi = 0; pi < phases.size(); ++pi) {
            auto& ph = phases[pi];
            s += "  Phase " + to_string(pi) + ": ";
            s += to_string(ph.total_updates()) + " updates, ";
            s += "max_thread=" + to_string(ph.max_thread_updates());
            if (ph.barrier_after) s += " [BARRIER]";
            s += "\n";
            for (size_t tid = 0; tid < ph.worklist.size(); ++tid) {
                s += "    T" + to_string(tid) + ": " +
                     to_string(ph.worklist[tid].size()) + " tasks\n";
            }
        }
        return s;
    }
};

//-----------------------------------------------------------------------------
// ScheduleProgram: wrapper for repeated or multi-epoch plans
//-----------------------------------------------------------------------------
struct ScheduleProgram {
    // For now: one EpochPlan repeated each epoch
    EpochPlan epoch_plan;

    // Future: vector<EpochPlan> for multi-epoch programs

    const EpochPlan& current_plan() const noexcept { return epoch_plan; }
};

} // namespace helios
