#include "helios/mdp_generators.h"
#include "helios/policy_eval_op.h"
#include "helios/runtime.h"
#include "helios/schedulers/static_blocks.h"
#include "helios/schedulers/shuffled_blocks.h"
#include "helios/schedulers/topk_gs.h"
#include "helios/schedulers/ca_topk_gs.h"
#include "helios/schedulers/residual_buckets.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace helios;

//=============================================================================
// Test Utilities
//=============================================================================

// Verify that the solution satisfies the Bellman equation: V = r + beta * P * V
// Returns the maximum absolute error: max_i |V_i - (r_i + beta * sum_j P_ij * V_j)|
real_t verify_bellman_equation(const MDP& mdp, const real_t* V) {
    real_t max_err = 0.0;
    for (index_t i = 0; i < mdp.n; ++i) {
        // Compute (P * V)_i = sum_j P_ij * V_j
        real_t pv = 0.0;
        for (index_t idx = mdp.row_ptr[i]; idx < mdp.row_ptr[i + 1]; ++idx) {
            const index_t j = mdp.col_idx[idx];
            const real_t p = mdp.probs[idx];
            pv += p * V[j];
        }

        // Bellman: V_i = r_i + beta * (P * V)_i
        const real_t expected = mdp.rewards[i] + mdp.beta * pv;
        const real_t err = std::abs(V[i] - expected);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Run convergence test with a given scheduler
template<typename SchedulerT>
bool run_convergence_test(const MDP& mdp, SchedulerT& sched,
                          const char* mdp_name, const char* sched_name,
                          real_t eps, size_t num_threads, Mode mode,
                          real_t max_seconds = 30.0) {
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);

    RuntimeConfig cfg;
    cfg.mode = mode;
    cfg.num_threads = num_threads;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = max_seconds;
    cfg.max_updates = 0;
    cfg.monitor_interval_ms = 50;
    cfg.rebuild_interval_ms = 100;
    cfg.record_trace = false;

    Runtime rt;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: %s with %s did not converge\n", mdp_name, sched_name);
        std::printf("  n = %u, final_residual = %.3e (eps = %.1e), time = %.2f sec\n",
                    mdp.n, result.final_residual_inf, eps, result.wall_time_sec);
        return false;
    }

    // Verify the solution satisfies the Bellman equation
    const real_t bellman_err = verify_bellman_equation(mdp, x.data());
    if (bellman_err > eps * 10) {
        std::printf("FAIL: %s with %s - Bellman equation not satisfied\n", mdp_name, sched_name);
        std::printf("  bellman_err = %.3e (eps = %.1e)\n", bellman_err, eps);
        return false;
    }

    std::printf("PASS: %s with %s\n", mdp_name, sched_name);
    std::printf("  n = %u, converged in %.3f sec, %" PRIu64 " updates (%.2e/sec)\n",
                mdp.n, result.wall_time_sec, result.total_updates, result.updates_per_sec);
    std::printf("  final_residual = %.3e, bellman_err = %.3e\n",
                result.final_residual_inf, bellman_err);

    return true;
}

//=============================================================================
// Grid MDP Tests
//=============================================================================

bool test_grid_mdp_jacobi() {
    std::printf("\n--- Grid MDP Tests ---\n");

    // 16x16 grid = 256 states
    MDP mdp = build_grid_mdp(16, 16, 0.9, 0.2, 1.0, 0.5);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Grid(16x16)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_grid_mdp_gauss_seidel() {
    MDP mdp = build_grid_mdp(16, 16, 0.9, 0.2, 1.0, 0.5);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Grid(16x16)", "GaussSeidel",
                                1e-6, 1, Mode::GaussSeidel);
}

bool test_grid_mdp_async() {
    MDP mdp = build_grid_mdp(32, 32, 0.9, 0.2, 1.0, 0.5);  // 1024 states
    mdp.validate(true);

    ShuffledBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Grid(32x32)", "Async(Shuffled)",
                                1e-5, 4, Mode::Async);
}

bool test_grid_mdp_async_topk() {
    MDP mdp = build_grid_mdp(32, 32, 0.95, 0.3, 1.0, 1.0);  // Higher beta, harder
    mdp.validate(true);

    TopKGSScheduler::Params params;
    params.K = 100;
    TopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "Grid(32x32,beta=0.95)", "Async(TopKGS)",
                                1e-5, 4, Mode::Async);
}

//=============================================================================
// Metastable MDP Tests (Challenging!)
//=============================================================================

bool test_metastable_mdp_jacobi() {
    std::printf("\n--- Metastable MDP Tests (Two Clusters) ---\n");

    // Small metastable problem
    MDP mdp = build_metastable_mdp(32, 0.9, 0.95, 0.05, 1.0, 2.0, 42);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Metastable(32)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_metastable_mdp_gauss_seidel() {
    MDP mdp = build_metastable_mdp(32, 0.9, 0.95, 0.05, 1.0, 2.0, 42);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Metastable(32)", "GaussSeidel",
                                1e-6, 1, Mode::GaussSeidel);
}

bool test_metastable_mdp_async() {
    // Larger metastable problem - this tests slow mixing
    MDP mdp = build_metastable_mdp(64, 0.9, 0.98, 0.02, 1.0, 3.0, 123);
    mdp.validate(true);

    CATopKGSScheduler::Params params;
    params.K = 20;
    params.G = 8;
    CATopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "Metastable(64,p_bridge=0.02)", "Async(CA-TopKGS)",
                                1e-5, 4, Mode::Async, 60.0);
}

bool test_metastable_mdp_high_beta() {
    // Very challenging: high discount + weak bridges
    MDP mdp = build_metastable_mdp(64, 0.95, 0.97, 0.03, 1.0, 2.0, 999);
    mdp.validate(true);

    ResidualBucketsScheduler::Params params;
    ResidualBucketsScheduler sched(params);
    return run_convergence_test(mdp, sched, "Metastable(64,beta=0.95)", "Async(ResidualBuckets)",
                                1e-4, 4, Mode::Async, 60.0);
}

//=============================================================================
// Star MDP Tests
//=============================================================================

bool test_star_mdp_jacobi() {
    std::printf("\n--- Star MDP Tests ---\n");

    MDP mdp = build_star_mdp(100, 0.9, 0.8, 1.0, 0.5);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Star(100)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_star_mdp_async() {
    MDP mdp = build_star_mdp(500, 0.9, 0.9, 2.0, 0.5);  // High p_to_hub
    mdp.validate(true);

    TopKGSScheduler::Params params;
    params.K = 50;
    params.sort_hot = true;
    TopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "Star(500)", "Async(TopKGS)",
                                1e-5, 4, Mode::Async);
}

//=============================================================================
// Chain MDP Tests
//=============================================================================

bool test_chain_mdp_jacobi() {
    std::printf("\n--- Chain MDP Tests ---\n");

    // Symmetric random walk on chain
    MDP mdp = build_chain_mdp(100, 0.9, 0.25, 0.5, 0.25, 1, true);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Chain(100,symmetric)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_chain_mdp_biased() {
    // Biased drift to the right - tests directional convergence
    MDP mdp = build_chain_mdp(100, 0.9, 0.1, 0.3, 0.6, 2, true);  // Quadratic rewards
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Chain(100,biased)", "GaussSeidel",
                                1e-6, 1, Mode::GaussSeidel);
}

bool test_chain_mdp_async() {
    // Longer chain with async
    MDP mdp = build_chain_mdp(500, 0.9, 0.2, 0.4, 0.4, 1, true);
    mdp.validate(true);

    ShuffledBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "Chain(500)", "Async(Shuffled)",
                                1e-5, 4, Mode::Async);
}

//=============================================================================
// Random Sparse MDP Tests
//=============================================================================

bool test_random_sparse_mdp_jacobi() {
    std::printf("\n--- Random Sparse MDP Tests ---\n");

    MDP mdp = build_random_sparse_mdp(200, 5, 0.9, 1.0, 42);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "RandomSparse(200,nnz=5)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_random_sparse_mdp_async() {
    MDP mdp = build_random_sparse_mdp(1000, 10, 0.9, 2.0, 123);
    mdp.validate(true);

    CATopKGSScheduler::Params params;
    params.K = 100;
    params.G = 8;
    CATopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "RandomSparse(1000,nnz=10)", "Async(CA-TopKGS)",
                                1e-5, 4, Mode::Async);
}

bool test_random_sparse_mdp_high_beta() {
    // Challenging: high beta + sparse random
    MDP mdp = build_random_sparse_mdp(500, 8, 0.95, 1.5, 999);
    mdp.validate(true);

    TopKGSScheduler::Params params;
    params.K = 50;
    params.sort_hot = true;
    TopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "RandomSparse(500,beta=0.95)", "Async(TopKGS)",
                                1e-4, 4, Mode::Async, 60.0);
}

//=============================================================================
// Multi-cluster MDP Tests
//=============================================================================

bool test_multi_cluster_mdp() {
    std::printf("\n--- Multi-cluster MDP Tests ---\n");

    // 4 clusters with different rewards
    std::vector<real_t> rewards = {1.0, 2.0, 3.0, 4.0};
    MDP mdp = build_multi_cluster_mdp(100, 4, 0.9, 0.9, rewards, 42);
    mdp.validate(true);

    StaticBlocksScheduler sched;
    return run_convergence_test(mdp, sched, "MultiCluster(100,k=4)", "Jacobi",
                                1e-6, 1, Mode::Jacobi);
}

bool test_multi_cluster_mdp_async() {
    std::vector<real_t> rewards = {1.0, 1.5, 2.0, 2.5, 3.0};
    MDP mdp = build_multi_cluster_mdp(200, 5, 0.9, 0.95, rewards, 123);
    mdp.validate(true);

    ResidualBucketsScheduler::Params params;
    ResidualBucketsScheduler sched(params);
    return run_convergence_test(mdp, sched, "MultiCluster(200,k=5)", "Async(ResidualBuckets)",
                                1e-5, 4, Mode::Async);
}

//=============================================================================
// Stress Tests (Larger problems)
//=============================================================================

bool test_large_grid_stress() {
    std::printf("\n--- Stress Tests (Larger Problems) ---\n");

    // 64x64 grid = 4096 states
    MDP mdp = build_grid_mdp(64, 64, 0.9, 0.25, 1.0, 0.0);
    mdp.validate(true);

    CATopKGSScheduler::Params params;
    params.K = 200;
    params.G = 16;
    CATopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "Grid(64x64=4096)", "Async(CA-TopKGS)",
                                1e-5, 4, Mode::Async, 60.0);
}

bool test_large_random_stress() {
    // 5000 states, moderate sparsity
    MDP mdp = build_random_sparse_mdp(5000, 15, 0.9, 1.0, 42);
    mdp.validate(true);

    TopKGSScheduler::Params params;
    params.K = 250;
    TopKGSScheduler sched(params);
    return run_convergence_test(mdp, sched, "RandomSparse(5000)", "Async(TopKGS)",
                                1e-4, 4, Mode::Async, 120.0);
}

//=============================================================================
// Scheduler Comparison (same problem, different schedulers)
//=============================================================================

bool test_scheduler_comparison() {
    std::printf("\n--- Scheduler Comparison (Grid 32x32, beta=0.9) ---\n");

    MDP mdp = build_grid_mdp(32, 32, 0.9, 0.2, 1.0, 0.0);
    mdp.validate(true);

    const real_t eps = 1e-5;
    const size_t num_threads = 4;
    int failures = 0;

    // Test each scheduler on the same problem
    {
        StaticBlocksScheduler sched;
        if (!run_convergence_test(mdp, sched, "Grid(32x32)", "StaticBlocks",
                                  eps, num_threads, Mode::Async)) failures++;
    }
    {
        ShuffledBlocksScheduler sched;
        if (!run_convergence_test(mdp, sched, "Grid(32x32)", "ShuffledBlocks",
                                  eps, num_threads, Mode::Async)) failures++;
    }
    {
        TopKGSScheduler::Params params;
        params.K = 100;
        TopKGSScheduler sched(params);
        if (!run_convergence_test(mdp, sched, "Grid(32x32)", "TopKGS(K=100)",
                                  eps, num_threads, Mode::Async)) failures++;
    }
    {
        CATopKGSScheduler::Params params;
        params.K = 100;
        params.G = 8;
        CATopKGSScheduler sched(params);
        if (!run_convergence_test(mdp, sched, "Grid(32x32)", "CA-TopKGS(K=100,G=8)",
                                  eps, num_threads, Mode::Async)) failures++;
    }
    {
        ResidualBucketsScheduler::Params params;
        ResidualBucketsScheduler sched(params);
        if (!run_convergence_test(mdp, sched, "Grid(32x32)", "ResidualBuckets",
                                  eps, num_threads, Mode::Async)) failures++;
    }

    return failures == 0;
}

//=============================================================================
// Main entry point (called from test_runtime_smoke.cc)
//=============================================================================

int run_complex_mdp_tests() {
    std::printf("\n========================================\n");
    std::printf("Complex MDP Convergence Tests\n");
    std::printf("========================================\n");

    int failures = 0;

    // Grid MDP tests
    if (!test_grid_mdp_jacobi()) failures++;
    if (!test_grid_mdp_gauss_seidel()) failures++;
    if (!test_grid_mdp_async()) failures++;
    if (!test_grid_mdp_async_topk()) failures++;

    // Metastable MDP tests
    if (!test_metastable_mdp_jacobi()) failures++;
    if (!test_metastable_mdp_gauss_seidel()) failures++;
    if (!test_metastable_mdp_async()) failures++;
    if (!test_metastable_mdp_high_beta()) failures++;

    // Star MDP tests
    if (!test_star_mdp_jacobi()) failures++;
    if (!test_star_mdp_async()) failures++;

    // Chain MDP tests
    if (!test_chain_mdp_jacobi()) failures++;
    if (!test_chain_mdp_biased()) failures++;
    if (!test_chain_mdp_async()) failures++;

    // Random sparse tests
    if (!test_random_sparse_mdp_jacobi()) failures++;
    if (!test_random_sparse_mdp_async()) failures++;
    if (!test_random_sparse_mdp_high_beta()) failures++;

    // Multi-cluster tests
    if (!test_multi_cluster_mdp()) failures++;
    if (!test_multi_cluster_mdp_async()) failures++;

    // Stress tests
    if (!test_large_grid_stress()) failures++;
    if (!test_large_random_stress()) failures++;

    // Scheduler comparison
    if (!test_scheduler_comparison()) failures++;

    std::printf("\n========================================\n");
    if (failures == 0) {
        std::printf("All complex MDP tests passed!\n");
    } else {
        std::printf("%d complex MDP test(s) failed.\n", failures);
    }
    std::printf("========================================\n");

    return failures;
}
