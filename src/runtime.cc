#include "helios/runtime.h"

#include <algorithm>
#include <atomic>
#include <barrier>
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

    out.wall_time_sec = now_sec();

    out.total_updates = total_updates;

    out.updates_per_sec = (out.wall_time_sec > 0.0) ? (static_cast<double>(total_updates) / out.wall_time_sec) : 0.0;

    out.final_residual_inf = residual_inf(op, x, stride);

    out.converged = (out.final_residual_inf <= cfg.eps);

    return out;
}

RunResult Runtime::run_async_(const Operator& op, Scheduler& sched, real_t* x, const RuntimeConfig& cfg) {
    
    const index_t n = op.n();

    RunResult out{};

    if (!x || n == 0) return out;

    #ifndef NDEBUG

        if (cfg.verify_invariants) op.check_invariants();

    #endif

    const int T = max(1, (int) cfg.num_threads);

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

    // Per-thread update counters, cache-line-padded to avoid false sharing.
    // Each counter is atomic for correct concurrent access from the monitor thread.
    struct alignas(128) PaddedCounter { atomic<uint64_t> count{0}; };
    vector<PaddedCounter> local_updates(T);

    auto get_total_updates = [&]() -> uint64_t {
        uint64_t sum = 0;
        for (int t = 0; t < T; ++t) sum += local_updates[t].count.load(memory_order_relaxed);
        return sum;
    };

    // Store best known residual for reporting/trace
    atomic<real_t> residual_inf_atomic{numeric_limits<real_t>::infinity()};

    // Monitor thread: periodically compute ||F(x) - x||_inf and check convergence
    const bool do_rebuild = sched.supports_rebuild() && cfg.rebuild_interval_ms > 0;

    thread monitor([&]() {
        vector<real_t> residuals_buf;
        if (do_rebuild) residuals_buf.resize(n);

        auto t_last_rebuild = Clock::now();

        auto scan_residuals = [&](bool collect) -> real_t {
            real_t mx = 0.0;
            for (index_t i = 0; i < n; ++i) {
                const real_t r = op.residual_i_async(i, x);
                if (collect) residuals_buf[i] = r;
                if (i % stride == 0 && r > mx) mx = r;
            }
            return mx;
        };

        // Initial scan
        {
            const bool need_rebuild = do_rebuild;
            real_t mx = scan_residuals(need_rebuild);
            residual_inf_atomic.store(mx, memory_order_relaxed);
            if (cfg.record_trace) out.trace.push_back({0.0, mx});
            if (need_rebuild) { sched.rebuild(residuals_buf); t_last_rebuild = Clock::now(); }
            if (mx <= cfg.eps) stop.store(true, memory_order_relaxed);
        }

        while (!stop.load(memory_order_relaxed)) {
            if (timed_out()) { stop.store(true, memory_order_relaxed); break; }

            const int interval = cfg.monitor_interval_ms;
            if (interval > 0) this_thread::sleep_for(chrono::milliseconds(interval));
            else this_thread::yield();

            const bool need_rebuild = do_rebuild &&
                chrono::duration_cast<chrono::milliseconds>(Clock::now() - t_last_rebuild).count()
                    >= static_cast<long long>(cfg.rebuild_interval_ms);

            real_t mx = scan_residuals(need_rebuild);
            residual_inf_atomic.store(mx, memory_order_relaxed);
            if (cfg.record_trace) out.trace.push_back({now_sec(), mx});
            if (need_rebuild) { sched.rebuild(residuals_buf); t_last_rebuild = Clock::now(); }
            if (mx <= cfg.eps) { stop.store(true, memory_order_relaxed); break; }
            if (cfg.max_updates != 0 && get_total_updates() >= cfg.max_updates) {
                stop.store(true, memory_order_relaxed); break;
            }
        }
    });

    // Worker threads: asynchronous coordinate updates
    vector<thread> workers;
    workers.reserve(T);

    for (int tid = 0; tid < T; ++tid) {
        workers.emplace_back([&, tid]() {
            auto& my_counter = local_updates[tid].count;
            uint64_t batch = 0;

            while (!stop.load(memory_order_relaxed)) {
                const index_t i = sched.next(tid);
                if (i >= n) { this_thread::yield(); continue; }

                const real_t fi = op.apply_i_async(i, x);
                atomic_ref<real_t> xi_ref(x[i]);
                const real_t xi = xi_ref.load(memory_order_relaxed);
                const real_t xnew = (real_t(1.0) - alpha) * xi + alpha * fi;
                xi_ref.store(xnew, memory_order_relaxed);

                // Batch counter updates to reduce atomic store frequency
                if (++batch >= 256) {
                    my_counter.fetch_add(batch, memory_order_relaxed);
                    batch = 0;
                }
            }
            // Flush remaining
            my_counter.fetch_add(batch, memory_order_relaxed);
        });
    }

    for (auto& th : workers) th.join();
    stop.store(true, memory_order_relaxed);
    if (monitor.joinable()) monitor.join();

    out.wall_time_sec = now_sec();
    out.total_updates = get_total_updates();  // safe: workers joined, no concurrent writes
    out.updates_per_sec = (out.wall_time_sec > 0.0) ? (out.total_updates / out.wall_time_sec) : 0.0;
    out.final_residual_inf = residual_inf_atomic.load(memory_order_relaxed);
    out.converged = (out.final_residual_inf <= cfg.eps);

    if (cfg.record_trace) {
        if (out.trace.empty() || out.trace.back().time_sec < out.wall_time_sec) {
            out.trace.push_back({out.wall_time_sec, out.final_residual_inf});
        }
    }

    return out;
}

RunResult Runtime::run_plan(const Operator& op, const EpochPlan& plan,
                             real_t* x, const RuntimeConfig& cfg) {
    const index_t n = op.n();
    RunResult out{};

    if (!x || n == 0) return out;
    if (plan.phases.empty()) return out;

    const size_t T = max(plan.threads, size_t(1));
    const real_t alpha = cfg.alpha;
    const int stride = (cfg.residual_scan_stride <= 0) ? 1 : cfg.residual_scan_stride;

    const auto t0 = Clock::now();
    auto t_last_sample = t0;

    auto now_sec = [&]() {
        return chrono::duration<double>(Clock::now() - t0).count();
    };

    // Set up profiling counters
    out.profiling.per_thread.resize(T);
    for (auto& tc : out.profiling.per_thread) tc.reset();

    // Initial residual
    {
        auto scan_t0 = Clock::now();
        const real_t r0 = residual_inf(op, x, stride);
        auto scan_t1 = Clock::now();
        out.profiling.time_in_residual_scan_ns +=
            static_cast<uint64_t>(chrono::duration_cast<chrono::nanoseconds>(scan_t1 - scan_t0).count());
        out.profiling.num_residual_scans++;

        out.final_residual_inf = r0;
        if (cfg.record_trace) out.trace.push_back({0.0, r0});

        if (r0 <= cfg.eps) {
            out.converged = true;
            return out;
        }
    }

    uint64_t total_updates = 0;
    bool should_stop = false;

    // Residual check helper (used by both paths)
    auto check_residual = [&](bool force) {
        const auto t_now = Clock::now();
        const int interval = cfg.monitor_interval_ms;
        const bool sample_now = force || (interval <= 0) ||
            (chrono::duration_cast<chrono::milliseconds>(t_now - t_last_sample).count() >= interval);
        if (sample_now) {
            t_last_sample = t_now;
            auto scan_t0 = Clock::now();
            const real_t r = residual_inf(op, x, stride);
            auto scan_t1 = Clock::now();
            out.profiling.time_in_residual_scan_ns +=
                static_cast<uint64_t>(chrono::duration_cast<chrono::nanoseconds>(scan_t1 - scan_t0).count());
            out.profiling.num_residual_scans++;
            out.final_residual_inf = r;
            if (cfg.record_trace) out.trace.push_back({now_sec(), r});
            if (r <= cfg.eps) { out.converged = true; should_stop = true; }
        }
    };

    if (T == 1) {
        // ── Single-threaded fast path: no thread overhead, no atomics ──
        while (!should_stop) {
            for (size_t pi = 0; pi < plan.phases.size() && !should_stop; ++pi) {
                const Phase& phase = plan.phases[pi];
                if (phase.worklist.size() > 0) {
                    auto update_t0 = Clock::now();
                    for (auto& task : phase.worklist[0]) {
                        for (index_t i = task.begin; i < task.end; ++i) {
                            const real_t fi = op.apply_i(i, x);
                            x[i] = (real_t(1.0) - alpha) * x[i] + alpha * fi;
                        }
                        total_updates += task.size();
                    }
                    auto update_t1 = Clock::now();
                    out.profiling.per_thread[0].updates_completed += phase.total_updates();
                    out.profiling.per_thread[0].time_in_update_ns +=
                        static_cast<uint64_t>(chrono::duration_cast<chrono::nanoseconds>(update_t1 - update_t0).count());
                }
                if (cfg.max_seconds > 0.0 && now_sec() >= cfg.max_seconds) { should_stop = true; break; }
                if (cfg.max_updates != 0 && total_updates >= cfg.max_updates) { should_stop = true; break; }
            }
            check_residual(should_stop);
        }
    } else {
        // ── Multi-threaded: persistent workers with barrier sync ──
        // Workers stay alive across all phases/epochs, eliminating thread
        // creation overhead (~50-100μs/thread) that killed scaling at small n.
        const size_t num_phases = plan.phases.size();
        atomic<size_t> current_phase{0};
        atomic<bool> workers_done{false};
        vector<uint64_t> thread_updates(T, 0);  // thread-local counters

        barrier phase_start(static_cast<ptrdiff_t>(T + 1));
        barrier phase_end(static_cast<ptrdiff_t>(T + 1));

        vector<thread> workers;
        workers.reserve(T);

        for (size_t tid = 0; tid < T; ++tid) {
            workers.emplace_back([&, tid]() {
                while (true) {
                    phase_start.arrive_and_wait();
                    if (workers_done.load(memory_order_relaxed)) return;

                    const size_t pi = current_phase.load(memory_order_relaxed);
                    const Phase& phase = plan.phases[pi];

                    if (tid < phase.worklist.size() && !phase.worklist[tid].empty()) {
                        auto update_t0 = Clock::now();
                        uint64_t my_updates = 0;
                        for (auto& task : phase.worklist[tid]) {
                            for (index_t i = task.begin; i < task.end; ++i) {
                                const real_t fi = op.apply_i_async(i, x);
                                atomic_ref<real_t> xi_ref(x[i]);
                                const real_t xi = xi_ref.load(memory_order_relaxed);
                                const real_t xnew = (real_t(1.0) - alpha) * xi + alpha * fi;
                                xi_ref.store(xnew, memory_order_relaxed);
                                my_updates++;
                            }
                        }
                        auto update_t1 = Clock::now();
                        out.profiling.per_thread[tid].updates_completed += my_updates;
                        out.profiling.per_thread[tid].time_in_update_ns +=
                            static_cast<uint64_t>(chrono::duration_cast<chrono::nanoseconds>(update_t1 - update_t0).count());
                        thread_updates[tid] += my_updates;
                    }

                    phase_end.arrive_and_wait();
                }
            });
        }

        while (!should_stop) {
            for (size_t pi = 0; pi < num_phases && !should_stop; ++pi) {
                current_phase.store(pi, memory_order_relaxed);
                phase_start.arrive_and_wait();  // wake workers
                phase_end.arrive_and_wait();    // wait for completion

                for (size_t tid = 0; tid < T; ++tid) {
                    total_updates += thread_updates[tid];
                    thread_updates[tid] = 0;
                }
                if (cfg.max_seconds > 0.0 && now_sec() >= cfg.max_seconds) { should_stop = true; break; }
                if (cfg.max_updates != 0 && total_updates >= cfg.max_updates) { should_stop = true; break; }
            }
            check_residual(should_stop);
        }

        workers_done.store(true, memory_order_relaxed);
        phase_start.arrive_and_wait();  // wake workers so they see done flag
        for (auto& th : workers) th.join();
    }

    out.wall_time_sec = now_sec();
    out.total_updates = total_updates;
    out.updates_per_sec = (out.wall_time_sec > 0.0)
        ? (static_cast<double>(total_updates) / out.wall_time_sec) : 0.0;
    out.profiling.total_updates = total_updates;

    // Final residual
    out.final_residual_inf = residual_inf(op, x, stride);
    out.converged = (out.final_residual_inf <= cfg.eps);

    if (cfg.record_trace) {
        if (out.trace.empty() || out.trace.back().time_sec < out.wall_time_sec) {
            out.trace.push_back({out.wall_time_sec, out.final_residual_inf});
        }
    }

    return out;
}

} // namespace helios