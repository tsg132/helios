#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "helios/types.h"

using namespace std;

namespace helios {

//=============================================================================
// Per-thread profiling counters
//=============================================================================
struct alignas(kCacheLine) ThreadCounters {
    uint64_t updates_completed = 0;
    uint64_t time_in_update_ns = 0;  // aggregate nanoseconds spent in updates

    void reset() {
        updates_completed = 0;
        time_in_update_ns = 0;
    }
};

//=============================================================================
// Global profiling counters
//=============================================================================
struct ProfilingResult {
    vector<ThreadCounters> per_thread;  // per-thread counters
    uint64_t time_in_residual_scan_ns = 0;
    uint64_t num_residual_scans = 0;
    uint64_t total_updates = 0;

    // Derived metrics
    double avg_update_cost_ns() const {
        if (total_updates == 0) return 0.0;
        uint64_t total_ns = 0;
        for (auto& tc : per_thread) total_ns += tc.time_in_update_ns;
        return static_cast<double>(total_ns) / static_cast<double>(total_updates);
    }

    double avg_residual_scan_ns() const {
        if (num_residual_scans == 0) return 0.0;
        return static_cast<double>(time_in_residual_scan_ns) /
               static_cast<double>(num_residual_scans);
    }

    string summary() const {
        string s;
        s += "Profiling: total_updates=" + to_string(total_updates);
        s += " avg_update_ns=" + to_string(static_cast<uint64_t>(avg_update_cost_ns()));
        s += " residual_scans=" + to_string(num_residual_scans);
        s += " avg_scan_ns=" + to_string(static_cast<uint64_t>(avg_residual_scan_ns()));
        s += "\n";
        for (size_t t = 0; t < per_thread.size(); ++t) {
            s += "  T" + to_string(t) + ": updates=" +
                 to_string(per_thread[t].updates_completed) +
                 " time_ns=" + to_string(per_thread[t].time_in_update_ns) + "\n";
        }
        return s;
    }

    // ── Runtime breakdown ──────────────────────────────────────────────────
    // Decomposes wall time into operator compute, residual scanning, and
    // scheduling / synchronisation overhead.  Fractions sum to 1.0 (modulo
    // timer granularity).
    struct BreakdownReport {
        double op_compute_frac = 0.0;  // operator apply_i time / wall_ns
        double residual_frac   = 0.0;  // residual scan time  / wall_ns
        double overhead_frac   = 0.0;  // scheduler + sync + other

        double op_compute_pct() const { return op_compute_frac * 100.0; }
        double residual_pct()   const { return residual_frac   * 100.0; }
        double overhead_pct()   const { return overhead_frac   * 100.0; }
    };

    // wall_time_sec: total wall-clock time from RunResult::wall_time_sec
    BreakdownReport breakdown(double wall_time_sec) const {
        if (wall_time_sec <= 0.0) return {};
        const double wall_ns = wall_time_sec * 1e9;

        // Sum operator compute time across all threads.
        // For async, threads overlap so sum > wall; divide by thread count
        // to get the "per-wall-second effective compute fraction".
        uint64_t sum_update_ns = 0;
        for (const auto& tc : per_thread)
            sum_update_ns += tc.time_in_update_ns;
        const double T = per_thread.empty() ? 1.0 : (double)per_thread.size();
        double op_frac  = static_cast<double>(sum_update_ns) / (wall_ns * T);
        double res_frac = static_cast<double>(time_in_residual_scan_ns) / wall_ns;

        // Clamp to [0,1] to absorb timer granularity noise
        op_frac  = std::min(op_frac,  1.0);
        res_frac = std::min(res_frac, std::max(0.0, 1.0 - op_frac));

        BreakdownReport r;
        r.op_compute_frac = op_frac;
        r.residual_frac   = res_frac;
        r.overhead_frac   = std::max(0.0, 1.0 - op_frac - res_frac);
        return r;
    }
};

} // namespace helios
