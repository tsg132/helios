#include "helios/runtime.h"
#include "helios/mdp.h"
#include "helios/policy_eval_op.h"
#include "helios/schedulers/static_blocks.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace helios;

// Build a ring MDP: state i transitions to self (0.5) and (i+1) mod n (0.5)
// All rewards = 1.0, beta = discount factor
MDP build_ring_mdp(index_t n, real_t beta) {
    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    // Each state has exactly 2 transitions (nnz = 2*n)
    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(2 * n);
    mdp.probs.resize(2 * n);
    mdp.rewards.resize(n, 1.0);

    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = 2 * i;

        // Transition to self with prob 0.5
        mdp.col_idx[2 * i] = i;
        mdp.probs[2 * i] = 0.5;

        // Transition to next state with prob 0.5
        mdp.col_idx[2 * i + 1] = (i + 1) % n;
        mdp.probs[2 * i + 1] = 0.5;
    }
    mdp.row_ptr[n] = 2 * n;

    return mdp;
}

bool test_jacobi_ring_convergence() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    // Build MDP and validate
    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);

    // Create operator
    PolicyEvalOp op(&mdp);

    // Initial state vector (zeros)
    std::vector<real_t> x(n, 0.0);

    // Configure runtime
    RuntimeConfig cfg;
    cfg.mode = Mode::Jacobi;
    cfg.alpha = 1.0;          // No relaxation
    cfg.eps = eps;
    cfg.max_seconds = 10.0;   // Timeout after 10s
    cfg.max_updates = 0;      // No update limit
    cfg.monitor_interval_ms = 10;
    cfg.record_trace = true;

    // Run
    Runtime rt;
    StaticBlocksScheduler sched;  // Not used for Jacobi, but required by API
    RunResult result = rt.run(op, sched, x.data(), cfg);

    // Check convergence
    if (!result.converged) {
        std::printf("FAIL: Jacobi did not converge\n");
        std::printf("  final_residual_inf = %.9e (eps = %.9e)\n",
                    result.final_residual_inf, eps);
        return false;
    }

    // Analytical solution: V = 1 / (1 - beta) = 10.0
    const real_t expected = 1.0 / (1.0 - beta);
    real_t max_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t err = std::abs(x[i] - expected);
        if (err > max_err) max_err = err;
    }

    // Solution should be within eps of analytical value
    if (max_err > eps * 10) {  // Allow some slack
        std::printf("FAIL: Solution not close to analytical value\n");
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: Jacobi ring convergence\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e\n", n, beta, eps);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);
    std::printf("  final_residual_inf = %.9e\n", result.final_residual_inf);
    std::printf("  solution x[0] = %.6f (expected %.6f, max_err = %.9e)\n",
                x[0], expected, max_err);

    return true;
}

bool test_residual_computation() {
    constexpr index_t n = 4;
    constexpr real_t beta = 0.5;

    MDP mdp = build_ring_mdp(n, beta);
    PolicyEvalOp op(&mdp);

    // x = [1, 2, 3, 4]
    std::vector<real_t> x = {1.0, 2.0, 3.0, 4.0};

    // F_0(x) = r_0 + beta * (0.5*x[0] + 0.5*x[1]) = 1 + 0.5*(0.5*1 + 0.5*2) = 1 + 0.75 = 1.75
    // residual_0 = |F_0(x) - x[0]| = |1.75 - 1| = 0.75
    real_t r0 = op.residual_i(0, x.data());
    real_t expected_r0 = 0.75;

    if (std::abs(r0 - expected_r0) > 1e-12) {
        std::printf("FAIL: residual_i(0) = %.9f, expected %.9f\n", r0, expected_r0);
        return false;
    }

    // Test global residual
    real_t r_inf = Runtime::residual_inf(op, x.data());

    // Compute expected max residual manually
    // F_1(x) = 1 + 0.5*(0.5*2 + 0.5*3) = 1 + 1.25 = 2.25, residual = |2.25 - 2| = 0.25
    // F_2(x) = 1 + 0.5*(0.5*3 + 0.5*4) = 1 + 1.75 = 2.75, residual = |2.75 - 3| = 0.25
    // F_3(x) = 1 + 0.5*(0.5*4 + 0.5*1) = 1 + 1.25 = 2.25, residual = |2.25 - 4| = 1.75
    // max = 1.75
    real_t expected_r_inf = 1.75;

    if (std::abs(r_inf - expected_r_inf) > 1e-12) {
        std::printf("FAIL: residual_inf = %.9f, expected %.9f\n", r_inf, expected_r_inf);
        return false;
    }

    std::printf("PASS: Residual computation\n");
    return true;
}

bool test_gauss_seidel_ring_convergence() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);

    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(n, 0.0);

    RuntimeConfig cfg;
    cfg.mode = Mode::GaussSeidel;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.max_updates = 0;
    cfg.monitor_interval_ms = 10;
    cfg.record_trace = true;

    Runtime rt;
    StaticBlocksScheduler sched;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Gauss-Seidel did not converge\n");
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
        std::printf("FAIL: Gauss-Seidel solution not close to analytical value\n");
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: Gauss-Seidel ring convergence\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e\n", n, beta, eps);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);
    std::printf("  solution x[0] = %.6f (expected %.6f, max_err = %.9e)\n",
                x[0], expected, max_err);

    return true;
}

bool test_async_ring_convergence() {
    constexpr index_t n = 16;
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
    cfg.record_trace = true;

    Runtime rt;
    StaticBlocksScheduler sched;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Async did not converge\n");
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
        std::printf("FAIL: Async solution not close to analytical value\n");
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: Async ring convergence\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e, threads = %zu\n", n, beta, eps, cfg.num_threads);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);
    std::printf("  solution x[0] = %.6f (expected %.6f, max_err = %.9e)\n",
                x[0], expected, max_err);

    return true;
}

bool test_async_multithread_stress() {
    // Larger problem with more threads to stress-test concurrency
    constexpr index_t n = 256;
    constexpr real_t beta = 0.95;
    constexpr real_t eps = 1e-5;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);

    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(n, 0.0);

    RuntimeConfig cfg;
    cfg.mode = Mode::Async;
    cfg.num_threads = 4;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 30.0;
    cfg.max_updates = 0;
    cfg.monitor_interval_ms = 50;
    cfg.record_trace = false;

    Runtime rt;
    StaticBlocksScheduler sched;
    RunResult result = rt.run(op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Async stress test did not converge\n");
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
        std::printf("FAIL: Async stress solution not close to analytical value\n");
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: Async multithread stress test\n");
    std::printf("  n = %u, beta = %.2f, eps = %.1e, threads = %zu\n", n, beta, eps, cfg.num_threads);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);

    return true;
}

int main() {
    int failures = 0;

    if (!test_residual_computation()) failures++;
    if (!test_jacobi_ring_convergence()) failures++;
    if (!test_gauss_seidel_ring_convergence()) failures++;
    if (!test_async_ring_convergence()) failures++;
    if (!test_async_multithread_stress()) failures++;

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    } else {
        std::printf("\n%d test(s) failed.\n", failures);
        return 1;
    }
}
