// tests/test_simd.cc — Correctness tests for NEON SIMD kernels.
// Verifies that NEON-optimized operators produce identical results to
// the scalar PolicyEvalOp within floating-point tolerance.

#include "helios/policy_eval_op.h"
#include "helios/policy_eval_op_neon.h"
#include "helios/simd/neon_kernels.h"
#include "helios/mdp_generators.h"
#include "helios/runtime.h"
#include "helios/schedulers/static_blocks.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <vector>

using namespace helios;

// FMA reordering causes last-bit differences; never require bitwise equality.
static constexpr real_t TOL = 1e-12;

// ── Test 1: apply_i matches scalar across multiple MDP types ────────────
static bool test_apply_i_match() {
    struct Case { const char* name; MDP mdp; };
    std::vector<Case> cases;
    cases.push_back({"ring_16", build_ring_mdp(16, 0.9)});
    cases.push_back({"grid_10x10", build_grid_mdp(10, 10, 0.95)});
    cases.push_back({"meta_100", build_metastable_mdp(100, 0.99, 0.9, 0.1, 1.0, 2.0, 42)});
    cases.push_back({"rand_500", build_random_sparse_mdp(500, 20, 0.99, 1.0, 42)});
    cases.push_back({"star_200", build_star_mdp(200, 0.95)});
    cases.push_back({"chain_300", build_chain_mdp(300, 0.9)});

    for (auto& c : cases) {
        PolicyEvalOp scalar_op(&c.mdp);
        PolicyEvalOpNeon neon_op(&c.mdp);

        std::vector<real_t> x(c.mdp.n);
        for (index_t i = 0; i < c.mdp.n; ++i)
            x[i] = 1.0 + 0.1 * static_cast<real_t>(i % 17);

        for (index_t i = 0; i < c.mdp.n; ++i) {
            real_t s = scalar_op.apply_i(i, x.data());
            real_t n = neon_op.apply_i(i, x.data());
            if (std::abs(s - n) > TOL) {
                std::printf("FAIL: apply_i mismatch [%s] row %u: scalar=%.15e neon=%.15e diff=%.2e\n",
                            c.name, i, s, n, std::abs(s - n));
                return false;
            }
        }
    }
    std::printf("PASS: apply_i NEON matches scalar for all MDPs\n");
    return true;
}

// ── Test 2: residual_i matches scalar ───────────────────────────────────
static bool test_residual_i_match() {
    MDP mdp = build_random_sparse_mdp(1000, 15, 0.99, 1.0, 123);
    PolicyEvalOp scalar_op(&mdp);
    PolicyEvalOpNeon neon_op(&mdp);

    std::vector<real_t> x(mdp.n);
    for (index_t i = 0; i < mdp.n; ++i)
        x[i] = 0.5 * static_cast<real_t>(i);

    for (index_t i = 0; i < mdp.n; ++i) {
        real_t s = scalar_op.residual_i(i, x.data());
        real_t n = neon_op.residual_i(i, x.data());
        if (std::abs(s - n) > TOL) {
            std::printf("FAIL: residual_i mismatch row %u: scalar=%.15e neon=%.15e\n", i, s, n);
            return false;
        }
    }
    std::printf("PASS: residual_i NEON matches scalar\n");
    return true;
}

// ── Test 3: neon_jacobi_sweep matches scalar sweep ──────────────────────
static bool test_jacobi_sweep_match() {
    MDP mdp = build_random_sparse_mdp(500, 12, 0.95, 1.0, 77);
    std::vector<real_t> x(mdp.n);
    for (index_t i = 0; i < mdp.n; ++i) x[i] = 1.0;

    const real_t alpha = 0.7;

    // Scalar sweep
    PolicyEvalOp scalar_op(&mdp);
    std::vector<real_t> x_scalar(mdp.n);
    for (index_t i = 0; i < mdp.n; ++i) {
        real_t fi = scalar_op.apply_i(i, x.data());
        x_scalar[i] = (1.0 - alpha) * x[i] + alpha * fi;
    }

    // NEON batch sweep
    std::vector<real_t> x_neon(mdp.n);
    neon::neon_jacobi_sweep(mdp, x.data(), x_neon.data(), alpha);

    for (index_t i = 0; i < mdp.n; ++i) {
        if (std::abs(x_scalar[i] - x_neon[i]) > TOL) {
            std::printf("FAIL: jacobi_sweep mismatch row %u: scalar=%.15e neon=%.15e diff=%.2e\n",
                        i, x_scalar[i], x_neon[i], std::abs(x_scalar[i] - x_neon[i]));
            return false;
        }
    }
    std::printf("PASS: neon_jacobi_sweep matches scalar\n");
    return true;
}

// ── Test 4: neon_residual_max matches Runtime::residual_inf ─────────────
static bool test_residual_max_match() {
    MDP mdp = build_random_sparse_mdp(800, 10, 0.99, 1.0, 99);
    PolicyEvalOp scalar_op(&mdp);

    std::vector<real_t> x(mdp.n);
    for (index_t i = 0; i < mdp.n; ++i) x[i] = 2.0 * (i % 7);

    real_t scalar_max = Runtime::residual_inf(scalar_op, x.data(), 1);
    real_t neon_max = neon::neon_residual_max(mdp, x.data());

    if (std::abs(scalar_max - neon_max) > TOL) {
        std::printf("FAIL: residual_max mismatch: scalar=%.15e neon=%.15e diff=%.2e\n",
                    scalar_max, neon_max, std::abs(scalar_max - neon_max));
        return false;
    }
    std::printf("PASS: neon_residual_max matches scalar\n");
    return true;
}

// ── Test 5: Full convergence — NEON operator through Runtime ────────────
static bool test_neon_convergence() {
    MDP mdp = build_ring_mdp(64, 0.9);
    PolicyEvalOpNeon neon_op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);

    RuntimeConfig cfg;
    cfg.mode = Mode::Jacobi;
    cfg.alpha = 1.0;
    cfg.eps = 1e-8;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;
    cfg.record_trace = false;

    Runtime rt;
    StaticBlocksScheduler sched;
    RunResult result = rt.run(neon_op, sched, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: NEON operator did not converge via Runtime\n");
        std::printf("  final_residual_inf = %.9e\n", result.final_residual_inf);
        return false;
    }

    const real_t expected = 1.0 / (1.0 - 0.9);  // 10.0
    real_t max_err = 0.0;
    for (index_t i = 0; i < mdp.n; ++i) {
        real_t err = std::abs(x[i] - expected);
        if (err > max_err) max_err = err;
    }

    if (max_err > 1e-6) {
        std::printf("FAIL: NEON convergence solution wrong: expected %.6f, max_err=%.9e\n",
                    expected, max_err);
        return false;
    }

    std::printf("PASS: NEON operator converges correctly through Runtime\n");
    std::printf("  converged in %.3f sec, %" PRIu64 " updates, max_err=%.2e\n",
                result.wall_time_sec, result.total_updates, max_err);
    return true;
}

// ── Test 6: Edge cases — empty row, single-element row, odd lengths ─────
static bool test_edge_cases() {
    // Custom MDP with varying row lengths: 0, 1, 3, 5
    MDP mdp;
    mdp.n = 4;
    mdp.beta = 0.9;
    mdp.rewards = {1.0, 2.0, 3.0, 4.0};
    mdp.row_ptr = {0, 0, 1, 4, 9};
    mdp.col_idx = {0,  0, 1, 2,  0, 1, 2, 3, 0};
    mdp.probs   = {1.0,  0.3, 0.3, 0.4,  0.1, 0.2, 0.2, 0.3, 0.2};

    PolicyEvalOp scalar_op(&mdp);
    PolicyEvalOpNeon neon_op(&mdp);
    std::vector<real_t> x = {1.0, 2.0, 3.0, 4.0};

    for (index_t i = 0; i < mdp.n; ++i) {
        real_t s = scalar_op.apply_i(i, x.data());
        real_t n = neon_op.apply_i(i, x.data());
        if (std::abs(s - n) > TOL) {
            std::printf("FAIL: edge case row %u (len=%u): scalar=%.15e neon=%.15e\n",
                        i, mdp.row_ptr[i + 1] - mdp.row_ptr[i], s, n);
            return false;
        }
    }

    // Also test the batch kernels on this edge-case MDP
    std::vector<real_t> x_scalar(mdp.n), x_neon(mdp.n);
    for (index_t i = 0; i < mdp.n; ++i) {
        real_t fi = scalar_op.apply_i(i, x.data());
        x_scalar[i] = fi;
    }
    neon::neon_jacobi_sweep(mdp, x.data(), x_neon.data(), 1.0);  // alpha=1.0 => x_next = fi
    for (index_t i = 0; i < mdp.n; ++i) {
        if (std::abs(x_scalar[i] - x_neon[i]) > TOL) {
            std::printf("FAIL: edge case batch sweep row %u: scalar=%.15e neon=%.15e\n",
                        i, x_scalar[i], x_neon[i]);
            return false;
        }
    }

    real_t scalar_max = Runtime::residual_inf(scalar_op, x.data(), 1);
    real_t neon_max = neon::neon_residual_max(mdp, x.data());
    if (std::abs(scalar_max - neon_max) > TOL) {
        std::printf("FAIL: edge case residual_max: scalar=%.15e neon=%.15e\n",
                    scalar_max, neon_max);
        return false;
    }

    std::printf("PASS: NEON handles edge cases (empty/short rows)\n");
    return true;
}

int main() {
    int failures = 0;

    std::printf("=== Helios SIMD Correctness Tests ===\n\n");

#ifdef __ARM_NEON
    std::printf("ARM NEON: ENABLED\n\n");
#else
    std::printf("ARM NEON: NOT AVAILABLE (testing scalar fallbacks)\n\n");
#endif

    if (!test_apply_i_match()) failures++;
    if (!test_residual_i_match()) failures++;
    if (!test_jacobi_sweep_match()) failures++;
    if (!test_residual_max_match()) failures++;
    if (!test_neon_convergence()) failures++;
    if (!test_edge_cases()) failures++;

    if (failures == 0) {
        std::printf("\nAll SIMD tests passed.\n");
        return 0;
    } else {
        std::printf("\n%d SIMD test(s) failed.\n", failures);
        return 1;
    }
}
