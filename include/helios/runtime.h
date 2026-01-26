/*

The runtime must run one of the several execution modes:

{Jacobi, Gauss-Seidel, Asynchronous}

and provide:

stopping condition, monitoring, performance counters, deterministic reporting.

Runtime configuration must include:

num_threads, alpha, eps, max_seconds, max_updates, monitor_interval_ms, mode enum.

For phase, we simply stop when the infinite norm is smaller or equal than eps.

This norm is computeed by either scanning all i and calling residual_i or computing apply_i and compareing to x[i] if residual not available.

The result should include:

converged (bool), final_residual, wall_time_sec, total_updates (count of coordinate updates),

updates_per_sec, residual_trace.

*/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "helios/types.h"
#include "helios/operator.h"
#include "helios/scheduler.h"

using namespace std;

namespace helios {

    enum class Mode : uint8_t {
        Jacobi,
        GaussSeidel,
        Async
    };

    struct ResidualSample {
        double time_sec = 0.0;
        real_t residual = 0.0;
    };

    struct RuntimeConfig {
        size_t num_threads = 1;
        real_t alpha = 1.0;
        real_t eps = 1e-6;
        double max_seconds = 0.0;
        uint64_t max_updates = 0;
        size_t monitor_interval_ms = 100;

        int residual_scan_stride = 1;
    
        bool verify_invariants = true;
    };

    struct RunResult {

        bool converged = false;

        real_t final_residual_inf = 0.0;

        double wall_time_sec = 0.0;

        uint64_t total_updates = 0;

        double updates_per_sec = 0.0;

        vector<ResidualSample> trace;

    };

    class Runtime {
        public:

            RunResult run(const Operator& op, Scheduler& scheduler, const RuntimeConfig& config);

            static real_t residual_inf(const Operator& op, const real_t* x, int stride = 1);

        private:

            RunResult run_jacobi_(const Operator& op, real_t* x, const RuntimeConfig& config);

            RunResult run_gauss_seidel_(const Operator& op, real_t* x, const RuntimeConfig& config);

            RunResult run_async_(const Operator& op, Scheduler& scheduler, real_t* x, const RuntimeConfig& config);
    };

}  // namespace helios