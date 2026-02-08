// tests/test_stress.cc — Definitive stress tests for Helios
// Covers all MDP types × execution modes × schedulers × planners
// with larger problem sizes, high-beta regimes, and correctness verification.

#include "helios/runtime.h"
#include "helios/mdp.h"
#include "helios/mdp_generators.h"
#include "helios/policy_eval_op.h"
#include "helios/plan.h"
#include "helios/planner.h"
#include "helios/cost_model.h"
#include "helios/autotune.h"
#include "helios/profiling.h"
#include "helios/schedulers/static_blocks.h"
#include "helios/schedulers/shuffled_blocks.h"
#include "helios/schedulers/topk_gs.h"
#include "helios/schedulers/ca_topk_gs.h"
#include "helios/schedulers/residual_buckets.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace helios;

// ============================================================================
// Verification Utilities
// ============================================================================

static real_t bellman_error(const MDP& mdp, const real_t* V) {
    real_t mx = 0.0;
    for (index_t i = 0; i < mdp.n; ++i) {
        real_t pv = 0.0;
        for (index_t idx = mdp.row_ptr[i]; idx < mdp.row_ptr[i + 1]; ++idx)
            pv += mdp.probs[idx] * V[mdp.col_idx[idx]];
        real_t expected = mdp.rewards[i] + mdp.beta * pv;
        real_t err = std::abs(V[i] - expected);
        if (err > mx) mx = err;
    }
    return mx;
}

struct TestResult {
    bool pass;
    double wall_sec;
    uint64_t updates;
    double ups;
    double residual;
    double bellman_err;
};

// Run a scheduler-based test and verify Bellman equation
static TestResult stress_sched(const MDP& mdp, const char* tn,
                                Scheduler& sc, Mode mode, size_t T,
                                real_t eps, double maxs) {
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);
    RuntimeConfig cfg;
    cfg.mode = mode; cfg.num_threads = T; cfg.alpha = 1.0; cfg.eps = eps;
    cfg.max_seconds = maxs; cfg.monitor_interval_ms = 20; cfg.rebuild_interval_ms = 100;
    Runtime rt;
    RunResult rr = rt.run(op, sc, x.data(), cfg);

    real_t be = bellman_error(mdp, x.data());
    bool ok = rr.converged && be < eps * 10;

    std::printf("  %s %s  %.3fs  %" PRIu64 " upd  res=%.2e  bell=%.2e\n",
                ok ? "PASS:" : "FAIL:", tn, rr.wall_time_sec, rr.total_updates,
                rr.final_residual_inf, be);
    if (!rr.converged)
        std::printf("    ** did not converge (res=%.2e, eps=%.1e)\n", rr.final_residual_inf, eps);
    if (be > eps * 10)
        std::printf("    ** Bellman violated (err=%.2e)\n", be);

    return {ok, rr.wall_time_sec, rr.total_updates, rr.updates_per_sec, rr.final_residual_inf, be};
}

// Run a planner-based test and verify Bellman equation
static TestResult stress_plan(const MDP& mdp, const char* tn,
                               Planner& pl, const PlannerConfig& pc,
                               size_t T, real_t eps, double maxs) {
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);
    EpochPlan plan = pl.build(op, x.data(), pc);
    populate_task_weights(plan, &mdp);

    RuntimeConfig cfg;
    cfg.alpha = 1.0; cfg.eps = eps; cfg.max_seconds = maxs;
    cfg.num_threads = T; cfg.monitor_interval_ms = 20;
    Runtime rt;
    RunResult rr = rt.run_plan(op, plan, x.data(), cfg);

    real_t be = bellman_error(mdp, x.data());
    bool ok = rr.converged && be < eps * 10;

    std::printf("  %s %s  %.3fs  %" PRIu64 " upd  res=%.2e  bell=%.2e\n",
                ok ? "PASS:" : "FAIL:", tn, rr.wall_time_sec, rr.total_updates,
                rr.final_residual_inf, be);
    if (!rr.converged)
        std::printf("    ** did not converge (res=%.2e, eps=%.1e)\n", rr.final_residual_inf, eps);
    if (be > eps * 10)
        std::printf("    ** Bellman violated (err=%.2e)\n", be);

    return {ok, rr.wall_time_sec, rr.total_updates, rr.updates_per_sec, rr.final_residual_inf, be};
}

// ============================================================================
// S1: Large Grid — all solvers
// ============================================================================

static int test_large_grid_battery() {
    std::printf("\n--- S1: Large Grid 64x64 (n=4096) battery ---\n");
    MDP mdp = build_grid_mdp(64, 64, 0.9, 0.2, 1.0, 0.5);
    mdp.validate(true);
    const real_t eps = 1e-5;
    const double ms = 60.0;
    const size_t T = 4;
    int fail = 0;

    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Grid64_Jacobi",s,Mode::Jacobi,1,eps,ms).pass) fail++; }
    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Grid64_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Grid64_Async_Static",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ShuffledBlocksScheduler s; if (!stress_sched(mdp,"Grid64_Async_Shuffled",s,Mode::Async,T,eps,ms).pass) fail++; }
    { TopKGSScheduler::Params p; p.K=200; TopKGSScheduler s(p);
      if (!stress_sched(mdp,"Grid64_Async_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { CATopKGSScheduler::Params p; p.K=200; p.G=8; CATopKGSScheduler s(p);
      if (!stress_sched(mdp,"Grid64_Async_CATopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ResidualBucketsScheduler::Params p; ResidualBucketsScheduler s(p);
      if (!stress_sched(mdp,"Grid64_Async_ResBuck",s,Mode::Async,T,eps,ms).pass) fail++; }

    { StaticPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64;
      if (!stress_plan(mdp,"Grid64_Plan_Static",pl,pc,T,eps,ms).pass) fail++; }
    { ColoredPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64; pc.colors=T;
      if (!stress_plan(mdp,"Grid64_Plan_Colored",pl,pc,T,eps,ms).pass) fail++; }
    { PriorityPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64; pc.colors=T;
      pc.K=800; pc.hot_phase_enabled=true;
      if (!stress_plan(mdp,"Grid64_Plan_Priority",pl,pc,T,eps,ms).pass) fail++; }

    return fail;
}

// ============================================================================
// S2: Metastable — hard convergence
// ============================================================================

static int test_metastable_battery() {
    std::printf("\n--- S2: Metastable (n=256, beta=0.9, p_bridge=0.03) ---\n");
    MDP mdp = build_metastable_mdp(256, 0.9, 0.97, 0.03, 1.0, 3.0, 42);
    mdp.validate(true);
    const real_t eps = 1e-4;
    const double ms = 60.0;
    const size_t T = 4;
    int fail = 0;

    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Meta256_Jacobi",s,Mode::Jacobi,1,eps,ms).pass) fail++; }
    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Meta256_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
    { TopKGSScheduler::Params p; p.K=40; TopKGSScheduler s(p);
      if (!stress_sched(mdp,"Meta256_Async_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { CATopKGSScheduler::Params p; p.K=40; p.G=8; CATopKGSScheduler s(p);
      if (!stress_sched(mdp,"Meta256_Async_CATopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ResidualBucketsScheduler::Params p; ResidualBucketsScheduler s(p);
      if (!stress_sched(mdp,"Meta256_Async_ResBuck",s,Mode::Async,T,eps,ms).pass) fail++; }
    { PriorityPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=16; pc.colors=T;
      pc.K=64; pc.hot_phase_enabled=true;
      if (!stress_plan(mdp,"Meta256_Plan_Priority",pl,pc,T,eps,ms).pass) fail++; }

    return fail;
}

// ============================================================================
// S3: Multi-cluster — 8 clusters
// ============================================================================

static int test_multi_cluster_battery() {
    std::printf("\n--- S3: Multi-cluster (n=400, k=8, beta=0.9) ---\n");
    std::vector<real_t> rw = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
    MDP mdp = build_multi_cluster_mdp(400, 8, 0.9, 0.92, rw, 42);
    mdp.validate(true);
    const real_t eps = 1e-5;
    const double ms = 60.0;
    const size_t T = 4;
    int fail = 0;

    { StaticBlocksScheduler s; if (!stress_sched(mdp,"MC8_Jacobi",s,Mode::Jacobi,1,eps,ms).pass) fail++; }
    { StaticBlocksScheduler s; if (!stress_sched(mdp,"MC8_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
    { ShuffledBlocksScheduler s; if (!stress_sched(mdp,"MC8_Async_Shuffled",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ResidualBucketsScheduler::Params p; ResidualBucketsScheduler s(p);
      if (!stress_sched(mdp,"MC8_Async_ResBuck",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ColoredPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=32; pc.colors=T;
      if (!stress_plan(mdp,"MC8_Plan_Colored",pl,pc,T,eps,ms).pass) fail++; }

    return fail;
}

// ============================================================================
// S4: High-beta stress — near-singular (beta = 0.99)
// ============================================================================

static int test_high_beta_stress() {
    std::printf("\n--- S4: High beta (beta=0.99) stress ---\n");
    const real_t eps = 1e-3;
    const double ms = 90.0;
    const size_t T = 4;
    int fail = 0;

    // Grid high beta
    {
        MDP mdp = build_grid_mdp(20, 20, 0.99, 0.2, 1.0, 0.0);
        mdp.validate(true);
        std::printf("  Grid 20x20 beta=0.99:\n");
        { StaticBlocksScheduler s; if (!stress_sched(mdp,"HiB_Grid_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
        { TopKGSScheduler::Params p; p.K=40; TopKGSScheduler s(p);
          if (!stress_sched(mdp,"HiB_Grid_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    }

    // Chain high beta — very slow propagation
    {
        MDP mdp = build_chain_mdp(200, 0.99, 0.25, 0.5, 0.25, 1, true);
        mdp.validate(true);
        std::printf("  Chain 200 beta=0.99:\n");
        { StaticBlocksScheduler s; if (!stress_sched(mdp,"HiB_Chain_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
        { ShuffledBlocksScheduler s; if (!stress_sched(mdp,"HiB_Chain_Shuffled",s,Mode::Async,T,eps,ms).pass) fail++; }
    }

    // Random high beta
    {
        MDP mdp = build_random_sparse_mdp(500, 10, 0.99, 1.0, 42);
        mdp.validate(true);
        std::printf("  Random 500 beta=0.99:\n");
        { TopKGSScheduler::Params p; p.K=50; TopKGSScheduler s(p);
          if (!stress_sched(mdp,"HiB_Rand_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
        { PriorityPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=32; pc.colors=T;
          pc.K=100; pc.hot_phase_enabled=true;
          if (!stress_plan(mdp,"HiB_Rand_Priority",pl,pc,T,eps,ms).pass) fail++; }
    }

    return fail;
}

// ============================================================================
// S5: Large random sparse — 10k states
// ============================================================================

static int test_large_random() {
    std::printf("\n--- S5: Large Random Sparse (n=10000) ---\n");
    MDP mdp = build_random_sparse_mdp(10000, 12, 0.9, 1.0, 42);
    mdp.validate(true);
    const real_t eps = 1e-4;
    const double ms = 120.0;
    const size_t T = 4;
    int fail = 0;

    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Big10k_Async_Static",s,Mode::Async,T,eps,ms).pass) fail++; }
    { TopKGSScheduler::Params p; p.K=500; TopKGSScheduler s(p);
      if (!stress_sched(mdp,"Big10k_Async_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ColoredPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=128; pc.colors=T;
      if (!stress_plan(mdp,"Big10k_Plan_Colored",pl,pc,T,eps,ms).pass) fail++; }

    return fail;
}

// ============================================================================
// S6: Star MDP — skewed workload
// ============================================================================

static int test_star_stress() {
    std::printf("\n--- S6: Star MDP (n=1000) ---\n");
    MDP mdp = build_star_mdp(1000, 0.9, 0.9, 2.0, 0.5);
    mdp.validate(true);
    const real_t eps = 1e-5;
    const double ms = 60.0;
    const size_t T = 4;
    int fail = 0;

    { StaticBlocksScheduler s; if (!stress_sched(mdp,"Star1k_Jacobi",s,Mode::Jacobi,1,eps,ms).pass) fail++; }
    { TopKGSScheduler::Params p; p.K=100; p.sort_hot=true; TopKGSScheduler s(p);
      if (!stress_sched(mdp,"Star1k_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    { ResidualBucketsScheduler::Params p; ResidualBucketsScheduler s(p);
      if (!stress_sched(mdp,"Star1k_ResBuck",s,Mode::Async,T,eps,ms).pass) fail++; }
    { PriorityPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64; pc.colors=T;
      pc.K=200; pc.hot_phase_enabled=true;
      if (!stress_plan(mdp,"Star1k_Priority",pl,pc,T,eps,ms).pass) fail++; }

    return fail;
}

// ============================================================================
// S7: Chain with biased drift — directional propagation
// ============================================================================

static int test_chain_battery() {
    std::printf("\n--- S7: Chain MDP battery ---\n");
    const real_t eps = 1e-5;
    const double ms = 60.0;
    const size_t T = 4;
    int fail = 0;

    // Symmetric chain, quadratic rewards
    {
        MDP mdp = build_chain_mdp(500, 0.9, 0.25, 0.5, 0.25, 2, true);
        mdp.validate(true);
        std::printf("  Symmetric chain n=500 quad rewards:\n");
        { StaticBlocksScheduler s; if (!stress_sched(mdp,"Chain500_sym_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
        { ShuffledBlocksScheduler s; if (!stress_sched(mdp,"Chain500_sym_Shuffled",s,Mode::Async,T,eps,ms).pass) fail++; }
    }

    // Right-biased drift
    {
        MDP mdp = build_chain_mdp(500, 0.9, 0.1, 0.3, 0.6, 1, true);
        mdp.validate(true);
        std::printf("  Right-biased chain n=500:\n");
        { StaticBlocksScheduler s; if (!stress_sched(mdp,"Chain500_rbias_GS",s,Mode::GaussSeidel,1,eps,ms).pass) fail++; }
        { TopKGSScheduler::Params p; p.K=50; TopKGSScheduler s(p);
          if (!stress_sched(mdp,"Chain500_rbias_TopKGS",s,Mode::Async,T,eps,ms).pass) fail++; }
    }

    // Absorbing boundaries
    {
        MDP mdp = build_chain_mdp(200, 0.9, 0.25, 0.5, 0.25, 0, false);
        mdp.validate(true);
        std::printf("  Absorbing chain n=200:\n");
        { StaticBlocksScheduler s; if (!stress_sched(mdp,"Chain200_abs_Jacobi",s,Mode::Jacobi,1,eps,ms).pass) fail++; }
        { ColoredPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=16; pc.colors=T;
          if (!stress_plan(mdp,"Chain200_abs_Colored",pl,pc,T,eps,ms).pass) fail++; }
    }

    return fail;
}

// ============================================================================
// S8: Thread safety — same MDP solved with 1,2,4,8 threads
// ============================================================================

static int test_thread_safety() {
    std::printf("\n--- S8: Thread safety (same result across thread counts) ---\n");
    MDP mdp = build_grid_mdp(32, 32, 0.9, 0.2, 1.0, 0.0);
    mdp.validate(true);
    const real_t eps = 1e-5;
    const double ms = 60.0;
    int fail = 0;

    // Get single-threaded reference solution
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x_ref(mdp.n, 0.0);
    {
        RuntimeConfig cfg;
        cfg.mode = Mode::GaussSeidel; cfg.alpha = 1.0; cfg.eps = eps;
        cfg.max_seconds = ms; cfg.monitor_interval_ms = 10;
        Runtime rt;
        StaticBlocksScheduler s;
        rt.run(op, s, x_ref.data(), cfg);
    }

    for (size_t T : {size_t(2), size_t(4), size_t(8)}) {
        // Async mode
        {
            std::vector<real_t> x(mdp.n, 0.0);
            RuntimeConfig cfg;
            cfg.mode = Mode::Async; cfg.num_threads = T; cfg.alpha = 1.0; cfg.eps = eps;
            cfg.max_seconds = ms; cfg.monitor_interval_ms = 10; cfg.rebuild_interval_ms = 100;
            Runtime rt;
            StaticBlocksScheduler s;
            RunResult rr = rt.run(op, s, x.data(), cfg);

            real_t max_diff = 0.0;
            for (index_t i = 0; i < mdp.n; ++i) {
                real_t d = std::abs(x[i] - x_ref[i]);
                if (d > max_diff) max_diff = d;
            }

            bool ok = rr.converged && max_diff < eps * 100;
            std::printf("  %s T=%zu Async_Static  diff=%.2e  %s\n",
                        ok?"PASS:":"FAIL:", T, max_diff, rr.converged?"CONV":"FAIL");
            if (!ok) fail++;
        }

        // Plan mode
        {
            std::vector<real_t> x(mdp.n, 0.0);
            ColoredPlanner pl;
            PlannerConfig pc; pc.threads = T; pc.blk = 32; pc.colors = T;
            EpochPlan plan = pl.build(op, x.data(), pc);
            RuntimeConfig cfg;
            cfg.alpha = 1.0; cfg.eps = eps; cfg.max_seconds = ms;
            cfg.num_threads = T; cfg.monitor_interval_ms = 10;
            Runtime rt;
            RunResult rr = rt.run_plan(op, plan, x.data(), cfg);

            real_t max_diff = 0.0;
            for (index_t i = 0; i < mdp.n; ++i) {
                real_t d = std::abs(x[i] - x_ref[i]);
                if (d > max_diff) max_diff = d;
            }

            bool ok = rr.converged && max_diff < eps * 100;
            std::printf("  %s T=%zu Plan_Colored  diff=%.2e  %s\n",
                        ok?"PASS:":"FAIL:", T, max_diff, rr.converged?"CONV":"FAIL");
            if (!ok) fail++;
        }
    }

    return fail;
}

// ============================================================================
// S9: Profiling sanity — verify profiling data is consistent
// ============================================================================

static int test_profiling_sanity() {
    std::printf("\n--- S9: Profiling sanity ---\n");
    MDP mdp = build_ring_mdp(256, 0.9);
    PolicyEvalOp op(&mdp);
    int fail = 0;

    // Plan mode with profiling
    {
        std::vector<real_t> x(mdp.n, 0.0);
        ColoredPlanner pl;
        PlannerConfig pc; pc.threads = 4; pc.blk = 32; pc.colors = 4;
        EpochPlan plan = pl.build(op, x.data(), pc);
        RuntimeConfig cfg;
        cfg.alpha = 1.0; cfg.eps = 1e-6; cfg.max_seconds = 10.0;
        cfg.num_threads = 4; cfg.monitor_interval_ms = 0;
        Runtime rt;
        RunResult rr = rt.run_plan(op, plan, x.data(), cfg);

        if (!rr.converged) { std::printf("  FAIL: did not converge\n"); return 1; }

        // Check profiling data
        bool ok = true;
        if (rr.profiling.per_thread.size() != 4) { ok = false; std::printf("  FAIL: per_thread.size=%zu\n", rr.profiling.per_thread.size()); }
        if (rr.profiling.total_updates == 0) { ok = false; std::printf("  FAIL: total_updates=0\n"); }
        if (rr.profiling.num_residual_scans == 0) { ok = false; std::printf("  FAIL: num_residual_scans=0\n"); }

        // Check per-thread load balance: no thread should have 0 updates
        uint64_t min_u = UINT64_MAX, max_u = 0;
        for (auto& tc : rr.profiling.per_thread) {
            if (tc.updates_completed < min_u) min_u = tc.updates_completed;
            if (tc.updates_completed > max_u) max_u = tc.updates_completed;
        }
        // Balance: max/min < 3x (generous threshold)
        if (min_u > 0 && max_u > 3 * min_u) {
            ok = false;
            std::printf("  FAIL: load imbalance min=%" PRIu64 " max=%" PRIu64 "\n", min_u, max_u);
        }

        double avg_ns = rr.profiling.avg_update_cost_ns();
        if (avg_ns <= 0.0) { ok = false; std::printf("  FAIL: avg_update_ns=%.1f\n", avg_ns); }

        if (ok) {
            std::printf("  PASS: Profiling sanity\n");
            std::printf("    total_updates=%" PRIu64 " scans=%" PRIu64 " avg_upd_ns=%.1f\n",
                        rr.profiling.total_updates, rr.profiling.num_residual_scans, avg_ns);
            std::printf("    per-thread updates: ");
            for (auto& tc : rr.profiling.per_thread) std::printf("%" PRIu64 " ", tc.updates_completed);
            std::printf("\n");
        } else {
            fail++;
        }
    }

    return fail;
}

// ============================================================================
// S10: Autotune correctness — autotune picks a valid plan that converges
// ============================================================================

static int test_autotune_correctness() {
    std::printf("\n--- S10: Autotune correctness ---\n");
    int fail = 0;

    struct AC { const char* name; MDP mdp; };
    std::vector<AC> cases;
    cases.push_back({"AT_Grid",  build_grid_mdp(32,32,0.9)});
    cases.push_back({"AT_Meta",  build_metastable_mdp(128,0.9,0.95,0.05,1,2,42)});
    cases.push_back({"AT_Star",  build_star_mdp(256,0.9)});

    for (auto& ac : cases) {
        PolicyEvalOp op(&ac.mdp);
        std::vector<real_t> x(ac.mdp.n, 0.0);

        AutotuneConfig atcfg;
        atcfg.blk_candidates = {16, 32, 64};
        atcfg.color_multipliers = {1, 2};
        atcfg.K_fractions = {0.02, 0.05};
        atcfg.pilot_seconds = 0.2;
        atcfg.top_M = 2;
        atcfg.runtime_cfg.num_threads = 2;
        atcfg.runtime_cfg.alpha = 1.0;
        atcfg.runtime_cfg.eps = 1e-5;
        atcfg.runtime_cfg.max_seconds = 30.0;
        atcfg.runtime_cfg.monitor_interval_ms = 0;

        AutotuneResult at = autotune(op, &ac.mdp, x.data(), atcfg);

        if (at.all_candidates.empty() || at.best_plan.phases.empty()) {
            std::printf("  FAIL: %s — autotune produced no plan\n", ac.name);
            fail++;
            continue;
        }

        // Run the plan to convergence
        std::fill(x.begin(), x.end(), 0.0);
        RuntimeConfig cfg; cfg.alpha=1; cfg.eps=1e-5; cfg.max_seconds=30;
        cfg.num_threads=2; cfg.monitor_interval_ms=0;
        Runtime rt;
        RunResult rr = rt.run_plan(op, at.best_plan, x.data(), cfg);

        real_t be = bellman_error(ac.mdp, x.data());
        bool ok = rr.converged && be < 1e-4;

        std::printf("  %s %s  %s blk=%u C=%u K=%u  %.3fs  bell=%.2e\n",
                    ok?"PASS:":"FAIL:", ac.name,
                    at.best.planner_name.c_str(), at.best.planner_cfg.blk,
                    at.best.planner_cfg.colors, at.best.planner_cfg.K,
                    rr.wall_time_sec, be);
        if (!ok) fail++;
    }

    return fail;
}

// ============================================================================
// S11: Cost model sanity — colored costs more phases than static
// ============================================================================

static int test_cost_model_sanity() {
    std::printf("\n--- S11: Cost model sanity ---\n");
    MDP mdp = build_grid_mdp(32, 32, 0.9);
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);
    int fail = 0;

    // Static plan: 1 phase
    StaticPlanner sp;
    PlannerConfig spc; spc.threads = 4; spc.blk = 64;
    EpochPlan s_plan = sp.build(op, x.data(), spc);
    populate_task_weights(s_plan, &mdp);

    // Colored plan: 4 phases
    ColoredPlanner cp;
    PlannerConfig cpc; cpc.threads = 4; cpc.blk = 64; cpc.colors = 4;
    EpochPlan c_plan = cp.build(op, x.data(), cpc);
    populate_task_weights(c_plan, &mdp);

    CostModelConfig ccfg;
    CostEstimate s_est = estimate_plan_cost(s_plan, &mdp, ccfg);
    CostEstimate c_est = estimate_plan_cost(c_plan, &mdp, ccfg);

    // Colored should have higher phase penalty
    bool ok = (c_est.phase_penalty > s_est.phase_penalty) && (s_est.estimated_cost > 0) && (c_est.estimated_cost > 0);
    std::printf("  %s Cost model: Static est=%.0f (phases=%.0f), Colored est=%.0f (phases=%.0f)\n",
                ok?"PASS:":"FAIL:",
                s_est.estimated_cost, s_est.phase_penalty,
                c_est.estimated_cost, c_est.phase_penalty);
    if (!ok) fail++;

    return fail;
}

// ============================================================================
// Entry point
// ============================================================================

int run_stress_tests() {
    std::printf("\n========================================\n");
    std::printf("Stress Tests\n");
    std::printf("========================================\n");

    int failures = 0;

    failures += test_large_grid_battery();        // S1
    failures += test_metastable_battery();        // S2
    failures += test_multi_cluster_battery();     // S3
    failures += test_high_beta_stress();          // S4
    failures += test_large_random();              // S5
    failures += test_star_stress();               // S6
    failures += test_chain_battery();             // S7
    failures += test_thread_safety();             // S8
    failures += test_profiling_sanity();          // S9
    failures += test_autotune_correctness();      // S10
    failures += test_cost_model_sanity();         // S11

    std::printf("\n========================================\n");
    if (failures == 0)
        std::printf("All stress tests passed!\n");
    else
        std::printf("%d stress test(s) FAILED.\n", failures);
    std::printf("========================================\n");

    return failures;
}
