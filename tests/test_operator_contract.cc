// test_operator_contract.cc
// Contract tests for the helios::Operator interface.
// Verifies every invariant that any concrete Operator must satisfy.

#include "helios/policy_eval_op.h"
#include "helios/mdp.h"
#include "helios/types.h"

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

using namespace helios;

// ─── MDP helpers ─────────────────────────────────────────────────────────────

// 2-state MDP with fully known analytical fixed point.
//   State 0: transitions to state 1 with prob 1.0, reward = 2.
//   State 1: transitions to state 0 with prob 1.0, reward = 3.
//   beta = 0.5
// Solving V0 = 2 + 0.5*V1,  V1 = 3 + 0.5*V0:
//   V0 = 14/3,  V1 = 16/3.
static MDP build_two_state_mdp() {
    MDP mdp;
    mdp.n       = 2;
    mdp.beta    = 0.5;
    mdp.row_ptr = {0, 1, 2};
    mdp.col_idx = {1, 0};
    mdp.probs   = {1.0, 1.0};
    mdp.rewards = {2.0, 3.0};
    return mdp;
}

// Ring MDP: all rewards = 1, beta given.
// Analytical V* = 1/(1-beta) everywhere.
static MDP build_ring_mdp(index_t n, real_t beta) {
    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;
    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(2 * n);
    mdp.probs.resize(2 * n);
    mdp.rewards.resize(n, 1.0);
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i]     = 2 * i;
        mdp.col_idx[2*i]   = i;
        mdp.probs[2*i]     = 0.5;
        mdp.col_idx[2*i+1] = (i + 1) % n;
        mdp.probs[2*i+1]   = 0.5;
    }
    mdp.row_ptr[n] = 2 * n;
    return mdp;
}

// ─── Contract 1: n() is positive and stable ──────────────────────────────────

static bool test_n_positive_and_stable() {
    MDP mdp = build_ring_mdp(16, 0.9);
    PolicyEvalOp op(&mdp);

    if (op.n() == 0) {
        std::printf("FAIL: test_n_positive_and_stable: n() returned 0\n");
        return false;
    }
    if (op.n() != 16) {
        std::printf("FAIL: test_n_positive_and_stable: n()=%u, expected 16\n", op.n());
        return false;
    }
    for (int k = 0; k < 100; ++k) {
        if (op.n() != 16) {
            std::printf("FAIL: test_n_positive_and_stable: n() not stable on call %d\n", k);
            return false;
        }
    }
    std::printf("PASS: test_n_positive_and_stable\n");
    return true;
}

// ─── Contract 2: residual_i(i,x) == |apply_i(i,x) - x[i]| exactly ───────────

static bool test_residual_equals_abs_diff() {
    MDP mdp = build_ring_mdp(32, 0.9);
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> x(n);
    for (index_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.3 * (i % 7);

    for (index_t i = 0; i < n; ++i) {
        const real_t fi  = op.apply_i(i, x.data());
        const real_t ri  = op.residual_i(i, x.data());
        const real_t ref = std::abs(fi - x[i]);
        if (std::abs(ri - ref) > 1e-14) {
            std::printf(
                "FAIL: test_residual_equals_abs_diff at i=%u: "
                "residual_i=%.15e, |apply_i-x[i]|=%.15e, diff=%.3e\n",
                i, ri, ref, std::abs(ri - ref));
            return false;
        }
    }
    std::printf("PASS: test_residual_equals_abs_diff\n");
    return true;
}

// ─── Contract 3: residual_i >= 0 for all x ───────────────────────────────────

static bool test_residual_nonnegative() {
    MDP mdp = build_ring_mdp(64, 0.99);
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> x(n);
    for (index_t i = 0; i < n; ++i)
        x[i] = -5.0 + 0.17 * i;   // includes negative values

    for (index_t i = 0; i < n; ++i) {
        if (op.residual_i(i, x.data()) < 0.0) {
            std::printf("FAIL: test_residual_nonnegative at i=%u\n", i);
            return false;
        }
    }
    std::printf("PASS: test_residual_nonnegative\n");
    return true;
}

// ─── Contract 4: apply_i(i, 0) == rewards[i]  (since beta*P*0 = 0) ───────────

static bool test_apply_at_zero_equals_reward() {
    MDP mdp = build_two_state_mdp();
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> zeros(n, 0.0);

    for (index_t i = 0; i < n; ++i) {
        const real_t fi  = op.apply_i(i, zeros.data());
        const real_t ref = mdp.rewards[i];
        if (std::abs(fi - ref) > 1e-13) {
            std::printf(
                "FAIL: test_apply_at_zero_equals_reward at i=%u: "
                "apply_i=%.15e, reward=%.15e\n", i, fi, ref);
            return false;
        }
    }
    std::printf("PASS: test_apply_at_zero_equals_reward\n");
    return true;
}

// ─── Contract 5: residual_i ≈ 0 at the known fixed point ────────────────────

static bool test_residual_at_fixed_point() {
    // Two-state MDP: V* = [14/3, 16/3]
    {
        MDP mdp = build_two_state_mdp();
        PolicyEvalOp op(&mdp);
        std::vector<real_t> Vstar = {14.0 / 3.0, 16.0 / 3.0};
        for (index_t i = 0; i < 2; ++i) {
            const real_t ri = op.residual_i(i, Vstar.data());
            if (ri > 1e-12) {
                std::printf(
                    "FAIL: test_residual_at_fixed_point (2-state) at i=%u: "
                    "residual=%.3e\n", i, ri);
                return false;
            }
        }
    }
    // Ring MDP: V* = 1/(1-beta) everywhere
    {
        constexpr real_t beta = 0.9;
        MDP mdp = build_ring_mdp(16, beta);
        PolicyEvalOp op(&mdp);
        const real_t vstar = 1.0 / (1.0 - beta);
        std::vector<real_t> V(16, vstar);
        for (index_t i = 0; i < 16; ++i) {
            const real_t ri = op.residual_i(i, V.data());
            if (ri > 1e-12) {
                std::printf(
                    "FAIL: test_residual_at_fixed_point (ring) at i=%u: "
                    "residual=%.3e\n", i, ri);
                return false;
            }
        }
    }
    std::printf("PASS: test_residual_at_fixed_point\n");
    return true;
}

// ─── Contract 6: Contraction — ||F(V)-F(W)||_inf <= beta*||V-W||_inf ─────────

static bool test_contraction_property() {
    MDP mdp = build_ring_mdp(64, 0.9);
    PolicyEvalOp op(&mdp);
    const index_t n   = op.n();
    const real_t beta = mdp.beta;

    std::vector<real_t> V(n), W(n);
    for (index_t i = 0; i < n; ++i) {
        V[i] = 0.1 * i;
        W[i] = 1.0 + 0.05 * ((i * 13) % 17);
    }

    real_t diff_VW = 0.0;
    for (index_t i = 0; i < n; ++i)
        diff_VW = std::max(diff_VW, std::abs(V[i] - W[i]));

    real_t diff_FV_FW = 0.0;
    for (index_t i = 0; i < n; ++i) {
        const real_t fv = op.apply_i(i, V.data());
        const real_t fw = op.apply_i(i, W.data());
        diff_FV_FW = std::max(diff_FV_FW, std::abs(fv - fw));
    }

    if (diff_FV_FW > beta * diff_VW + 1e-12) {
        std::printf(
            "FAIL: test_contraction_property: "
            "||F(V)-F(W)||=%.15e > beta*||V-W||=%.15e\n",
            diff_FV_FW, beta * diff_VW);
        return false;
    }
    std::printf("PASS: test_contraction_property\n");
    return true;
}

// ─── Contract 7: Monotonicity — V >= W componentwise => F(V) >= F(W) ─────────

static bool test_monotonicity() {
    MDP mdp = build_ring_mdp(32, 0.8);
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> V(n), W(n);
    for (index_t i = 0; i < n; ++i) {
        W[i] = (real_t)i;
        V[i] = W[i] + 0.5 + 0.1 * (i % 5);   // V strictly > W everywhere
    }

    for (index_t i = 0; i < n; ++i) {
        const real_t fv = op.apply_i(i, V.data());
        const real_t fw = op.apply_i(i, W.data());
        if (fv < fw - 1e-13) {
            std::printf(
                "FAIL: test_monotonicity at i=%u: F(V)=%.15e < F(W)=%.15e\n",
                i, fv, fw);
            return false;
        }
    }
    std::printf("PASS: test_monotonicity\n");
    return true;
}

// ─── Contract 8: apply_i is deterministic (identical value on repeated calls) ─

static bool test_apply_i_deterministic() {
    MDP mdp = build_ring_mdp(16, 0.9);
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> x(n);
    for (index_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.1 * i;

    for (index_t i = 0; i < n; ++i) {
        const real_t first = op.apply_i(i, x.data());
        for (int k = 0; k < 10; ++k) {
            const real_t again = op.apply_i(i, x.data());
            if (again != first) {
                std::printf(
                    "FAIL: test_apply_i_deterministic at i=%u call %d: "
                    "%.15e != %.15e\n", i, k, again, first);
                return false;
            }
        }
    }
    std::printf("PASS: test_apply_i_deterministic\n");
    return true;
}

// ─── Contract 9: No cross-coordinate side effects (order-independent) ─────────

static bool test_no_stateful_side_effects() {
    MDP mdp = build_ring_mdp(16, 0.9);
    PolicyEvalOp op(&mdp);
    const index_t n = op.n();

    std::vector<real_t> x(n);
    for (index_t i = 0; i < n; ++i)
        x[i] = 2.0 + 0.1 * i;

    // Forward pass: 0 → n-1
    std::vector<real_t> forward(n), backward(n);
    for (index_t i = 0; i < n; ++i)
        forward[i] = op.apply_i(i, x.data());

    // Backward pass: n-1 → 0
    for (index_t i = n; i-- > 0;)
        backward[i] = op.apply_i(i, x.data());

    for (index_t i = 0; i < n; ++i) {
        if (forward[i] != backward[i]) {
            std::printf(
                "FAIL: test_no_stateful_side_effects at i=%u: "
                "forward=%.15e, backward=%.15e\n",
                i, forward[i], backward[i]);
            return false;
        }
    }
    std::printf("PASS: test_no_stateful_side_effects\n");
    return true;
}

// ─── Contract 10: Thread safety — concurrent apply_i matches sequential ───────

static bool test_thread_safety() {
    constexpr index_t n = 128;
    MDP mdp = build_ring_mdp(n, 0.9);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n);
    for (index_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.05 * i;

    // Sequential reference
    std::vector<real_t> seq(n);
    for (index_t i = 0; i < n; ++i)
        seq[i] = op.apply_i(i, x.data());

    // Parallel execution across 4 threads, each owns a disjoint slice
    std::vector<real_t> par(n, 0.0);
    constexpr int T = 4;
    {
        std::vector<std::thread> threads;
        threads.reserve(T);
        for (int t = 0; t < T; ++t) {
            threads.emplace_back([&, t]() {
                const index_t stride = (n + T - 1) / T;
                const index_t start  = (index_t)t * stride;
                const index_t end    = std::min(start + stride, n);
                for (index_t i = start; i < end; ++i)
                    par[i] = op.apply_i(i, x.data());
            });
        }
        for (auto& th : threads) th.join();
    }

    int bad = 0;
    for (index_t i = 0; i < n; ++i) {
        if (par[i] != seq[i]) {
            std::printf(
                "FAIL: test_thread_safety at i=%u: par=%.15e seq=%.15e\n",
                i, par[i], seq[i]);
            bad++;
        }
    }
    if (bad > 0) return false;
    std::printf("PASS: test_thread_safety (4 threads, n=%u)\n", n);
    return true;
}

// ─── Contract 11: name() returns a non-empty string ──────────────────────────

static bool test_name_nonempty() {
    MDP mdp = build_ring_mdp(8, 0.9);
    PolicyEvalOp op(&mdp);

    if (op.name().empty()) {
        std::printf("FAIL: test_name_nonempty: name() is empty\n");
        return false;
    }
    std::printf("PASS: test_name_nonempty (name=\"%.*s\")\n",
                (int)op.name().size(), op.name().data());
    return true;
}

// ─── Contract 12: check_invariants() does not throw for a valid operator ──────

static bool test_check_invariants_valid() {
    MDP mdp = build_ring_mdp(16, 0.9);
    PolicyEvalOp op(&mdp);

    try {
        op.check_invariants();
    } catch (const std::exception& e) {
        std::printf("FAIL: test_check_invariants_valid threw: %s\n", e.what());
        return false;
    }
    std::printf("PASS: test_check_invariants_valid\n");
    return true;
}

// ─── Contract 13: F(x) is linear in rewards at x=0 ──────────────────────────
// apply_i(i, 0) = r_i, so scaling rewards by c scales F_i(0) by c.

static bool test_linearity_in_rewards_at_zero() {
    MDP mdp_base   = build_two_state_mdp();
    MDP mdp_scaled = mdp_base;
    const real_t c = 3.7;
    for (auto& r : mdp_scaled.rewards) r *= c;

    PolicyEvalOp op_base(&mdp_base);
    PolicyEvalOp op_scaled(&mdp_scaled);

    std::vector<real_t> zeros(2, 0.0);

    for (index_t i = 0; i < 2; ++i) {
        const real_t f_base   = op_base.apply_i(i, zeros.data());
        const real_t f_scaled = op_scaled.apply_i(i, zeros.data());
        const real_t expected = c * f_base;
        if (std::abs(f_scaled - expected) > 1e-13) {
            std::printf(
                "FAIL: test_linearity_in_rewards_at_zero at i=%u: "
                "scaled=%.15e, c*base=%.15e\n", i, f_scaled, expected);
            return false;
        }
    }
    std::printf("PASS: test_linearity_in_rewards_at_zero\n");
    return true;
}

// ─── Entry point ─────────────────────────────────────────────────────────────

int run_operator_contract_tests() {
    std::printf("\n--- Operator Contract Tests ---\n");
    int failures = 0;

    if (!test_n_positive_and_stable())          failures++;
    if (!test_residual_equals_abs_diff())        failures++;
    if (!test_residual_nonnegative())            failures++;
    if (!test_apply_at_zero_equals_reward())     failures++;
    if (!test_residual_at_fixed_point())         failures++;
    if (!test_contraction_property())            failures++;
    if (!test_monotonicity())                    failures++;
    if (!test_apply_i_deterministic())           failures++;
    if (!test_no_stateful_side_effects())        failures++;
    if (!test_thread_safety())                   failures++;
    if (!test_name_nonempty())                   failures++;
    if (!test_check_invariants_valid())          failures++;
    if (!test_linearity_in_rewards_at_zero())    failures++;

    std::printf("--- Operator Contract Tests: %d failure(s) ---\n\n", failures);
    return failures;
}
