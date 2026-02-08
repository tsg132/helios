#pragma once

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
};

} // namespace helios
