#include "helios/schedulers/residual_buckets.h"
#include "helios/runtime.h"
#include "helios/mdp.h"
#include "helios/policy_eval_op.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <set>
#include <thread>
#include <vector>

using namespace helios;

// Forward declaration - defined in test_runtime_smoke.cc
MDP build_ring_mdp(index_t n, real_t beta);

//-----------------------------------------------------------------------------
// Test: ResidualBucketsScheduler basic init and next
//-----------------------------------------------------------------------------
bool test_residual_buckets_init() {
    constexpr index_t n = 100;
    constexpr size_t num_threads = 4;

    ResidualBucketsScheduler::Params params;
    params.num_buckets = 8;
    params.base = 1e-6;

    ResidualBucketsScheduler sched(params);
    sched.init(n, num_threads);

    // After init with no rebuild, all_tiny = true, should use round-robin
    std::set<index_t> seen;
    for (index_t i = 0; i < n; ++i) {
        index_t idx = sched.next(i % num_threads);
        if (idx >= n) {
            std::printf("FAIL: next() returned invalid index %" PRIu32 " >= n\n", idx);
            return false;
        }
        seen.insert(idx);
    }

    if (seen.size() != n) {
        std::printf("FAIL: Expected %u unique indices, got %zu\n", n, seen.size());
        return false;
    }

    std::printf("PASS: ResidualBucketsScheduler init and round-robin fallback\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: rebuild assigns indices to correct buckets based on residuals
//-----------------------------------------------------------------------------
bool test_residual_buckets_rebuild() {
    constexpr index_t n = 16;
    constexpr size_t num_threads = 2;

    ResidualBucketsScheduler::Params params;
    params.num_buckets = 4;  // Buckets 0, 1, 2, 3
    params.base = 1.0;       // base = 1.0, so bucket = floor(log2(r))
    params.fallback_round_robin = false;

    ResidualBucketsScheduler sched(params);
    sched.init(n, num_threads);

    // Create residuals with known bucket assignments:
    // r < 1 or r == 0 → bucket 0 (log2 < 0)
    // 1 <= r < 2     → bucket 0 (log2 in [0,1))
    // 2 <= r < 4     → bucket 1 (log2 in [1,2))
    // 4 <= r < 8     → bucket 2 (log2 in [2,3))
    // r >= 8         → bucket 3 (clamped)
    std::vector<real_t> residuals(n);

    // Indices 0-3: residual = 0.5 → bucket 0
    residuals[0] = 0.5; residuals[1] = 0.5; residuals[2] = 0.5; residuals[3] = 0.5;

    // Indices 4-7: residual = 2.0 → bucket 1 (log2(2) = 1)
    residuals[4] = 2.0; residuals[5] = 2.0; residuals[6] = 2.0; residuals[7] = 2.0;

    // Indices 8-11: residual = 5.0 → bucket 2 (log2(5) ≈ 2.32)
    residuals[8] = 5.0; residuals[9] = 5.0; residuals[10] = 5.0; residuals[11] = 5.0;

    // Indices 12-15: residual = 100.0 → bucket 3 (clamped)
    residuals[12] = 100.0; residuals[13] = 100.0; residuals[14] = 100.0; residuals[15] = 100.0;

    sched.rebuild(residuals);

    // next() should return high-bucket indices first (12-15), then 8-11, etc.
    std::vector<index_t> order;
    for (index_t i = 0; i < n; ++i) {
        index_t idx = sched.next(0);
        if (idx == std::numeric_limits<index_t>::max()) break;
        order.push_back(idx);
    }

    if (order.size() != n) {
        std::printf("FAIL: Expected %u indices, got %zu\n", n, order.size());
        return false;
    }

    // First 4 should be from bucket 3 (indices 12-15)
    for (size_t i = 0; i < 4; ++i) {
        if (order[i] < 12 || order[i] > 15) {
            std::printf("FAIL: Expected index 12-15 at position %zu, got %" PRIu32 "\n",
                        i, order[i]);
            return false;
        }
    }

    // Next 4 should be from bucket 2 (indices 8-11)
    for (size_t i = 4; i < 8; ++i) {
        if (order[i] < 8 || order[i] > 11) {
            std::printf("FAIL: Expected index 8-11 at position %zu, got %" PRIu32 "\n",
                        i, order[i]);
            return false;
        }
    }

    // Next 4 should be from bucket 1 (indices 4-7)
    for (size_t i = 8; i < 12; ++i) {
        if (order[i] < 4 || order[i] > 7) {
            std::printf("FAIL: Expected index 4-7 at position %zu, got %" PRIu32 "\n",
                        i, order[i]);
            return false;
        }
    }

    // Last 4 should be from bucket 0 (indices 0-3)
    for (size_t i = 12; i < 16; ++i) {
        if (order[i] > 3) {
            std::printf("FAIL: Expected index 0-3 at position %zu, got %" PRIu32 "\n",
                        i, order[i]);
            return false;
        }
    }

    std::printf("PASS: ResidualBucketsScheduler rebuild and priority ordering\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: Multi-threaded concurrent access to scheduler
//-----------------------------------------------------------------------------
bool test_residual_buckets_concurrent() {
    constexpr index_t n = 1000;
    constexpr size_t num_threads = 4;

    ResidualBucketsScheduler::Params params;
    params.num_buckets = 16;
    params.base = 1e-8;
    params.fallback_round_robin = false;  // Disable fallback to test bucket exhaustion

    ResidualBucketsScheduler sched(params);
    sched.init(n, num_threads);

    // Create varying residuals - all above base so they go into buckets
    std::vector<real_t> residuals(n);
    for (index_t i = 0; i < n; ++i) {
        residuals[i] = static_cast<real_t>(i + 1) * 1e-4;  // Well above base of 1e-8
    }
    sched.rebuild(residuals);

    // Simulate concurrent access - each thread grabs indices until buckets exhausted
    std::atomic<size_t> total_count{0};
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t count = 0;
            // Get indices until we've collectively processed enough
            // With fallback_round_robin=false, we get exactly n indices from buckets
            while (total_count.load(std::memory_order_relaxed) < n) {
                index_t idx = sched.next(t);
                if (idx < n) {
                    count++;
                    total_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    break;  // Bucket exhausted
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // With proper bucket scheduling, we should get exactly n indices in the first pass
    size_t final_count = total_count.load();
    if (final_count != n) {
        std::printf("FAIL: Expected %u indices, got %zu\n", n, final_count);
        return false;
    }

    std::printf("PASS: ResidualBucketsScheduler concurrent access (%zu threads, %u indices)\n",
                num_threads, n);
    return true;
}

//-----------------------------------------------------------------------------
// Test: Integration with Async runtime using residual buckets scheduler
//-----------------------------------------------------------------------------
bool test_async_with_residual_buckets() {
    constexpr index_t n = 64;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);

    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(n, 0.0);

    RuntimeConfig cfg;
    cfg.mode = Mode::Async;
    cfg.num_threads = 2;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.max_updates = 0;
    cfg.monitor_interval_ms = 10;
    cfg.record_trace = false;

    Runtime rt;
    ResidualBucketsScheduler sched;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Async with ResidualBuckets did not converge\n");
        std::printf("  final_residual_inf = %.9e (eps = %.9e)\n",
                    result.final_residual_inf, eps);
        return false;
    }

    const real_t expected = 1.0 / (1.0 - beta);
    real_t max_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t err = std::abs(x[i] - expected);
        if (err > max_err) max_err = err;
    }

    if (max_err > eps * 10) {
        std::printf("FAIL: Solution not close to analytical value\n");
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: Async with ResidualBucketsScheduler\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e, threads = %zu\n",
                n, beta, eps, cfg.num_threads);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);

    return true;
}

//-----------------------------------------------------------------------------
// Run all scheduler tests (called from test_runtime_smoke.cc)
//-----------------------------------------------------------------------------
int run_scheduler_tests() {
    int failures = 0;

    if (!test_residual_buckets_init()) failures++;
    if (!test_residual_buckets_rebuild()) failures++;
    if (!test_residual_buckets_concurrent()) failures++;
    if (!test_async_with_residual_buckets()) failures++;

    return failures;
}
