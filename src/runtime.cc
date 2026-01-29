#include "helios/runtime.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

using namespace std;

namespace helios {

using Clock = chrono::steady_clock;

RunResult Runtime::run(const Operator& op, Scheduler& sched, real_t* x, const RuntimeConfig& cfg) {

    #ifndef NDEBUG

        if (cfg.verify_invariants) {

            op.check_invariants();

        }

    #endif

    RunResult out{};

    if (!x) return out;

    switch (cfg.mode) {

        case Mode::Jacobi:

            return run_jacobi_(op, x, cfg);

        case Mode::GaussSeidel:
        
            return run_gauss_seidel_(op, x, cfg);

        case Mode::Async:  
        
            sched.init(op.n(), max(1, (int) cfg.num_threads));

            return run_async_(op, sched, x, cfg);

        default:

            return out;
    }

}

real_t Runtime::residual_inf(const Operator& op, const real_t* x, int stride) {
  

    const index_t n = op.n();

    if (n == 0) return 0.0;

    if (stride <= 0) stride = 1;

    real_t max_residual = 0.0;

    for (index_t i = 0; i < n; i = static_cast<index_t>(i + stride)) {

        const real_t res_i = op.residual_i(i, x);

        if (res_i > max_residual) {

            max_residual = res_i;

        }

    }


    return max_residual;

}

RunResult Runtime::run_jacobi_(const Operator& op, real_t* x, const RuntimeConfig& cfg) {

    /*

    Jacobi (syncronous) fixed-point iteration:

    x_i^{k + 1} = (1 - alpha) x_i^{k} + alpha F_{i}(x^{k}) for i = 0,...,n-1

    All coordinates see the same x^{k} when computing F(x^{k})

    convergence: |x^{k+1} - x^{k}|_{\infty} < \alpha\beta |x^{k} - x^{\star}|_{\infty}
    
    */

    const index_t n = op.n();

    RunResult out{};

    if (n == 0) return out;

    vector<real_t> bufA(n), bufB(n);

    memcpy(bufA.data(), x, n * sizeof(real_t));

    real_t* x_curr = bufA.data(); 

    real_t* x_next = bufB.data();

    const real_t alpha = cfg.alpha;

    const int stride = cfg.residual_scan_stride <= 0 ? 1 : cfg.residual_scan_stride;

    const auto t0 = Clock::now();

    auto t_last_sample = t0;

    auto now_sec = [&]() {

        return chrono::duration<double>(Clock::now() - t0).count(); // returns the number of ticks as a double

    };

    auto should_time_out = [&]() -> bool {

        if (cfg.max_seconds <= 0.0) return false;

        return now_sec() >= cfg.max_seconds;

    };

    auto should_update_out = [&] (uint64_t total_updates) -> bool {

        if (cfg.max_updates == 0) return false;

        return total_updates >= cfg.max_updates;
    };

    // Initial residual sample at t=0

    {
        const real_t r0 = residual_inf(op, x_curr, stride);

        out.final_residual_inf = r0;

        if (cfg.record_trace) out.trace.push_back({0.0, r0});

        if (r0 < cfg.eps) {

            out.converged = true;

            out.wall_time_sec = 0.0;

            out.total_updates = 0;

            out.updates_per_sec = 0.0;

            memcpy(x, x_curr, n * sizeof(real_t)); 

            return out;
        }
    }

    uint64_t total_updates = 0;

    while (true) {

        if (should_time_out()) break;

        if (should_update_out(total_updates)) break;

        // Perform Jacobi sweep:

        // x_next[i] = (1 - alpha) x_curr[i] + alpha * F_{i}(x_curr)

        for (index_t i = 0; i < n; ++i) {

            const real_t fi = op.apply_i(i, x_curr);

            x_next[i] = (real_t(1.0) - alpha) * x_curr[i] + alpha * fi;

        }

        total_updates += static_cast<uint64_t>(n);

        // Swap buffers

    swap(x_curr, x_next); // so that x_curr is the latest

    const auto t_now = Clock::now();

    const int interval = cfg.monitor_interval_ms;

    const bool sample_now =
        (interval <= 0) || (chrono::duration_cast<chrono::milliseconds>(t_now - t_last_sample).count() >= interval);

        if (sample_now) {

            t_last_sample = t_now;

            const real_t r = residual_inf(op, x_curr, stride);

            out.final_residual_inf = r;

            if (cfg.record_trace) out.trace.push_back({now_sec(), r});

            if (r <= cfg.eps) {

                out.converged = true;

                break;

            }

        }

    }

    out.wall_time_sec = now_sec();

    out.total_updates = total_updates;

    out.updates_per_sec = (out.wall_time_sec > 0.0) ? (static_cast<double>(total_updates) / out.wall_time_sec) : 0.0;

    if (out.converged) {

        out.final_residual_inf = residual_inf(op, x_curr, stride);

        if (cfg.record_trace) out.trace.push_back({out.wall_time_sec, out.final_residual_inf});

        if (out.final_residual_inf <= cfg.eps) out.converged = true;

    }

    memcpy(x, x_curr, n * sizeof(real_t));

    return out;


  
}

RunResult Runtime::run_gauss_seidel_(const Operator& op, real_t* x, const RuntimeConfig& cfg) {

    const index_t n = op.n();

    RunResult out{};

    if (!x || n == 0) return out;

    const real_t alpha = cfg.alpha;

    const int stride = cfg.residual_scan_stride <= 0 ? 1 : cfg.residual_scan_stride;

    const auto t0 = Clock::now();

    auto t_last_sample = t0;

    auto now_sec = [&]() {

        return chrono::duration<double>(Clock::now() - t0).count(); // returns the number of ticks as a double

    };

    auto should_time_out = [&]() -> bool {

        if (cfg.max_seconds <= 0.0) return false;

        return now_sec() >= cfg.max_seconds;

    };

    auto should_update_out = [&] (uint64_t total_updates) -> bool {

        if (cfg.max_updates == 0) return false;

        return total_updates >= cfg.max_updates;
    };

    // Initial residual sample at t=0

    {
        const real_t r0 = residual_inf(op, x, stride);

        out.final_residual_inf = r0;

        if (cfg.record_trace) out.trace.push_back({0.0, r0});

        if (r0 < cfg.eps) {

            out.converged = true;

            out.wall_time_sec = 0.0;

            out.total_updates = 0;

            out.updates_per_sec = 0.0;

            return out;
        }
    }

    uint64_t total_updates = 0;

    while (true) {

        if (should_time_out()) break;

        if (should_update_out(total_updates)) break;

        for (index_t i = 0; i < n; ++i) {

            const real_t fi = op.apply_i(i, x); // uses current x, including earlier updates in this sweep.

            x[i] = (real_t(1.0) - alpha) * x[i] + alpha * fi;
        }

        total_updates += static_cast<uint64_t>(n);

        const auto t_now = Clock::now();

        const int interval = cfg.monitor_interval_ms;

        const bool sample_now =
            (interval <= 0) || (chrono::duration_cast<chrono::milliseconds>(t_now - t_last_sample).count() >= interval);

        if (sample_now) {

            t_last_sample = t_now;

            const real_t r = residual_inf(op, x, stride);

            out.final_residual_inf = r;

            if (cfg.record_trace) out.trace.push_back({now_sec(), r});

            if (r <= cfg.eps) {

                out.converged = true;

                break;

            }

        }

    }
  
}

RunResult Runtime::run_async_(const Operator& op, Scheduler& sched, real_t* x, const RuntimeConfig& cfg) {
    
    const index_t n = op.n();

    RunResult out{};

    if (!x || n == 0) return out;

    #ifndef NDEBUG

        if (cfg.verify_invariants) op.check_invariants();

    #endif

    const int T = max(1, (int) cfg.num_threads);

    sched.init(n, T);

    const real_t alpha = cfg.alpha;

    const int stride = (cfg.residual_scan_stride <= 0) ? 1 : cfg.residual_scan_stride;

    const auto t0 = Clock::now();

    auto now_sec = [&]() {

        return chrono::duration<double>(Clock::now() - t0).count(); // returns the number of ticks as a double

    };

    auto timed_out = [&]() -> bool {

        if (cfg.max_seconds <= 0.0) return false;

        return now_sec() >= cfg.max_seconds;

    };

    atomic<bool> stop{false};

    atomic<uint64_t> total_updates{0};

    // Store best known residual for reporting/trace

    atomic<real_t> residual_inf_atomic{numeric_limits<real_t>::infinity()};

    // Monitor thread: periodically compute ||F(x) - x||_{\infty} and check for convergence/time limit

    thread monitor([&]() {

        auto last = Clock::now();

        {

            real_t mx = 0.0;

            for (index_t i = 0; i < n; i = static_cast<index_t>(i + stride)) {

                const real_t r = op.residual_i_async(i, x);

                if (r > mx) mx = r;

            }

            residual_inf_atomic.store(mx, memory_order_relaxed);

            if (cfg.record_trace) out.trace.push_back({0.0, mx});

            if (mx <= cfg.eps) stop.store(true, memory_order_relaxed);

        }

        while (!stop.load(memory_order_relaxed)) {

            if (timed_out()) {

                stop.store(true, memory_order_relaxed);

                break;

            }

            const int interval = cfg.monitor_interval_ms;

            if (interval > 0) {

                this_thread::sleep_for(chrono::milliseconds(interval));
            
            } else {

                this_thread::yield();

            }

        }

    });



    
}

} // namespace helios