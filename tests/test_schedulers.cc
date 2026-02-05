#include "helios/schedulers/residual_buckets.h"
#include "helios/schedulers/shuffled_blocks.h"
#include "helios/schedulers/topk_gs.h"
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
// Test: ShuffledBlocksScheduler basic init and coverage
//-----------------------------------------------------------------------------
bool test_shuffled_blocks_init() {
    constexpr index_t n = 100;
    constexpr size_t num_threads = 4;

    ShuffledBlocksScheduler sched;
    sched.init(n, num_threads);

    // Each thread should be able to get indices, and together they should cover all n indices
    std::set<index_t> seen;

    // Each thread processes its block
    for (size_t tid = 0; tid < num_threads; ++tid) {
        // Each thread's block size is approximately n/num_threads
        const index_t expected_block_size = n / num_threads + (tid < n % num_threads ? 1 : 0);
        for (index_t j = 0; j < expected_block_size; ++j) {
            index_t idx = sched.next(tid);
            if (idx >= n) {
                std::printf("FAIL: next() returned invalid index %" PRIu32 " >= n\n", idx);
                return false;
            }
            seen.insert(idx);
        }
    }

    if (seen.size() != n) {
        std::printf("FAIL: Expected %u unique indices, got %zu\n", n, seen.size());
        return false;
    }

    std::printf("PASS: ShuffledBlocksScheduler init and full coverage\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: ShuffledBlocksScheduler reshuffles after epoch
//-----------------------------------------------------------------------------
bool test_shuffled_blocks_reshuffle() {
    constexpr index_t n = 20;
    constexpr size_t num_threads = 2;

    ShuffledBlocksScheduler sched;
    sched.init(n, num_threads);

    // Thread 0 should have indices 0-9 (10 elements)
    // Collect first epoch order
    std::vector<index_t> epoch1;
    for (index_t i = 0; i < 10; ++i) {
        epoch1.push_back(sched.next(0));
    }

    // Collect second epoch order (should be reshuffled)
    std::vector<index_t> epoch2;
    for (index_t i = 0; i < 10; ++i) {
        epoch2.push_back(sched.next(0));
    }

    // Both epochs should contain the same indices (0-9)
    std::set<index_t> set1(epoch1.begin(), epoch1.end());
    std::set<index_t> set2(epoch2.begin(), epoch2.end());

    if (set1.size() != 10 || set2.size() != 10) {
        std::printf("FAIL: Epochs don't contain 10 unique indices\n");
        return false;
    }

    // Both should have the same elements
    if (set1 != set2) {
        std::printf("FAIL: Epochs don't contain the same indices\n");
        return false;
    }

    // Orders should differ (with very high probability due to shuffling)
    // For 10! permutations, probability of same order is 1/10! ≈ 2.8e-7
    bool different = false;
    for (size_t i = 0; i < 10; ++i) {
        if (epoch1[i] != epoch2[i]) {
            different = true;
            break;
        }
    }

    if (!different) {
        std::printf("WARN: Shuffle produced same order (very unlikely but possible)\n");
    }

    std::printf("PASS: ShuffledBlocksScheduler reshuffles after epoch\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: ShuffledBlocksScheduler multi-threaded concurrent access
//-----------------------------------------------------------------------------
bool test_shuffled_blocks_concurrent() {
    constexpr index_t n = 1000;
    constexpr size_t num_threads = 4;

    ShuffledBlocksScheduler sched;
    sched.init(n, num_threads);

    // Each thread collects its indices concurrently
    std::vector<std::set<index_t>> thread_indices(num_threads);
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread processes one full epoch of its block
            const index_t block_size = n / num_threads + (t < n % num_threads ? 1 : 0);
            for (index_t i = 0; i < block_size; ++i) {
                index_t idx = sched.next(t);
                if (idx < n) {
                    thread_indices[t].insert(idx);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Verify no overlap between threads and complete coverage
    std::set<index_t> all_indices;
    for (size_t t = 0; t < num_threads; ++t) {
        for (index_t idx : thread_indices[t]) {
            if (all_indices.count(idx)) {
                std::printf("FAIL: Index %" PRIu32 " appears in multiple threads\n", idx);
                return false;
            }
            all_indices.insert(idx);
        }
    }

    if (all_indices.size() != n) {
        std::printf("FAIL: Expected %u total indices, got %zu\n", n, all_indices.size());
        return false;
    }

    std::printf("PASS: ShuffledBlocksScheduler concurrent access (%zu threads, %u indices)\n",
                num_threads, n);
    return true;
}

//-----------------------------------------------------------------------------
// Test: Integration with Async runtime using shuffled blocks scheduler
//-----------------------------------------------------------------------------
bool test_async_with_shuffled_blocks() {
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
    ShuffledBlocksScheduler sched;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Async with ShuffledBlocks did not converge\n");
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

    std::printf("PASS: Async with ShuffledBlocksScheduler\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e, threads = %zu\n",
                n, beta, eps, cfg.num_threads);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);

    return true;
}

//-----------------------------------------------------------------------------
// Test: TopKGSScheduler basic init and coverage
//-----------------------------------------------------------------------------
bool test_topk_gs_init() {
    constexpr index_t n = 100;
    constexpr size_t num_threads = 4;

    TopKGSScheduler::Params params;
    params.K = 10;  // Small hot set for testing

    TopKGSScheduler sched(params);
    sched.init(n, num_threads);

    // Without rebuild, hot set is empty, so all indices come from fallback
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
        std::printf("FAIL: Expected %u unique indices in first pass, got %zu\n", n, seen.size());
        return false;
    }

    std::printf("PASS: TopKGSScheduler init and fallback coverage\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: TopKGSScheduler rebuild prioritizes high-residual indices
//-----------------------------------------------------------------------------
bool test_topk_gs_rebuild() {
    constexpr index_t n = 100;
    constexpr size_t num_threads = 2;

    TopKGSScheduler::Params params;
    params.K = 10;
    params.sort_hot = true;  // Sort for deterministic ordering

    TopKGSScheduler sched(params);
    sched.init(n, num_threads);

    // Create residuals: indices 90-99 have high residuals (10.0), rest are 0.1
    std::vector<real_t> residuals(n, 0.1);
    for (index_t i = 90; i < 100; ++i) {
        residuals[i] = 10.0;
    }

    sched.rebuild(residuals);

    // First K calls to next() should return high-residual indices (90-99)
    std::set<index_t> hot_indices;
    for (index_t i = 0; i < 10; ++i) {
        index_t idx = sched.next(0);
        hot_indices.insert(idx);
    }

    // All should be from the high-residual set
    for (index_t idx : hot_indices) {
        if (idx < 90) {
            std::printf("FAIL: Hot set contains low-residual index %" PRIu32 "\n", idx);
            return false;
        }
    }

    if (hot_indices.size() != 10) {
        std::printf("FAIL: Expected 10 hot indices, got %zu\n", hot_indices.size());
        return false;
    }

    std::printf("PASS: TopKGSScheduler rebuild prioritizes high-residual indices\n");
    return true;
}

//-----------------------------------------------------------------------------
// Test: TopKGSScheduler multi-threaded concurrent access
//-----------------------------------------------------------------------------
bool test_topk_gs_concurrent() {
    constexpr index_t n = 1000;
    constexpr size_t num_threads = 4;

    TopKGSScheduler::Params params;
    params.K = 100;

    TopKGSScheduler sched(params);
    sched.init(n, num_threads);

    // Create residuals with top 100 having high values
    std::vector<real_t> residuals(n);
    for (index_t i = 0; i < n; ++i) {
        residuals[i] = static_cast<real_t>(i + 1);  // Higher index = higher residual
    }
    sched.rebuild(residuals);

    // Collect indices from multiple threads concurrently
    std::atomic<size_t> hot_consumed{0};
    std::vector<std::set<index_t>> thread_indices(num_threads);
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread gets some indices
            for (index_t i = 0; i < 300; ++i) {
                index_t idx = sched.next(t);
                if (idx < n) {
                    thread_indices[t].insert(idx);
                    if (idx >= 900) {  // Top 100 indices
                        hot_consumed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // All hot indices should have been consumed (100 total, distributed among threads)
    // Note: some may have been consumed multiple times due to fallback, but at least 100
    size_t total_hot = hot_consumed.load();
    if (total_hot < 100) {
        std::printf("FAIL: Expected at least 100 hot indices consumed, got %zu\n", total_hot);
        return false;
    }

    std::printf("PASS: TopKGSScheduler concurrent access (%zu threads, K=%u)\n",
                num_threads, params.K);
    return true;
}

//-----------------------------------------------------------------------------
// Test: Integration with Async runtime using TopK GS scheduler
//-----------------------------------------------------------------------------
bool test_async_with_topk_gs() {
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
    cfg.rebuild_interval_ms = 50;  // Rebuild every 50ms
    cfg.record_trace = false;

    Runtime rt;
    TopKGSScheduler::Params params;
    params.K = 10;
    TopKGSScheduler sched(params);
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Async with TopKGS did not converge\n");
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

    std::printf("PASS: Async with TopKGSScheduler\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e, threads = %zu, K = %u\n",
                n, beta, eps, cfg.num_threads, params.K);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);

    return true;
}

//-----------------------------------------------------------------------------
// Run all scheduler tests (called from test_runtime_smoke.cc)
//-----------------------------------------------------------------------------
int run_scheduler_tests() {
    int failures = 0;

    // Shuffled blocks tests
    if (!test_shuffled_blocks_init()) failures++;
    if (!test_shuffled_blocks_reshuffle()) failures++;
    if (!test_shuffled_blocks_concurrent()) failures++;
    if (!test_async_with_shuffled_blocks()) failures++;

    // TopK GS tests
    if (!test_topk_gs_init()) failures++;
    if (!test_topk_gs_rebuild()) failures++;
    if (!test_topk_gs_concurrent()) failures++;
    if (!test_async_with_topk_gs()) failures++;

    // Residual buckets tests
    if (!test_residual_buckets_init()) failures++;
    if (!test_residual_buckets_rebuild()) failures++;
    if (!test_residual_buckets_concurrent()) failures++;
    if (!test_async_with_residual_buckets()) failures++;

    return failures;
}
