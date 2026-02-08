#include "helios/runtime.h"
#include "helios/mdp.h"
#include "helios/policy_eval_op.h"
#include "helios/plan.h"
#include "helios/planner.h"
#include "helios/cost_model.h"
#include "helios/profiling.h"
#include "helios/autotune.h"
#include "helios/mdp_generators.h"
#include "helios/schedulers/static_blocks.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace helios;

//=============================================================================
// Test helpers
//=============================================================================

static bool check_convergence(const char* test_name, const RunResult& result,
                               const real_t* x, index_t n, real_t expected,
                               real_t eps) {
    if (!result.converged) {
        std::printf("FAIL: %s did not converge\n", test_name);
        std::printf("  final_residual_inf = %.9e (eps = %.9e)\n",
                    result.final_residual_inf, eps);
        return false;
    }

    real_t max_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t err = std::abs(x[i] - expected);
        if (err > max_err) max_err = err;
    }

    if (max_err > eps * 10) {
        std::printf("FAIL: %s solution not close to analytical value\n", test_name);
        std::printf("  expected = %.6f, max_err = %.9e\n", expected, max_err);
        return false;
    }

    std::printf("PASS: %s\n", test_name);
    std::printf("  converged in %.3f sec, %" PRIu64 " updates (%.2e updates/sec)\n",
                result.wall_time_sec, result.total_updates, result.updates_per_sec);
    std::printf("  solution x[0] = %.6f (expected %.6f, max_err = %.9e)\n",
                x[0], expected, max_err);
    return true;
}

//=============================================================================
// D1: Schedule IR tests
//=============================================================================

static bool test_plan_ir_basic() {
    // Test basic Task/Phase/EpochPlan creation
    EpochPlan plan;
    plan.n = 100;
    plan.threads = 2;
    plan.blk = 10;
    plan.built_from = "test";

    Phase phase;
    phase.worklist.resize(2);

    // Thread 0: blocks [0,10), [20,30), [40,50)
    phase.worklist[0].push_back({TaskKind::BLOCK, 0, 10, 10.0, 0});
    phase.worklist[0].push_back({TaskKind::BLOCK, 20, 30, 10.0, 2});
    phase.worklist[0].push_back({TaskKind::BLOCK, 40, 50, 10.0, 4});

    // Thread 1: blocks [10,20), [30,40), [50,60)
    phase.worklist[1].push_back({TaskKind::BLOCK, 10, 20, 10.0, 1});
    phase.worklist[1].push_back({TaskKind::BLOCK, 30, 40, 10.0, 3});
    phase.worklist[1].push_back({TaskKind::BLOCK, 50, 60, 10.0, 5});

    plan.phases.push_back(phase);

    if (plan.total_updates() != 60) {
        std::printf("FAIL: Plan IR total_updates = %" PRIu64 ", expected 60\n",
                    plan.total_updates());
        return false;
    }

    if (plan.phases[0].max_thread_updates() != 30) {
        std::printf("FAIL: Phase max_thread_updates = %" PRIu64 ", expected 30\n",
                    plan.phases[0].max_thread_updates());
        return false;
    }

    // Test summary doesn't crash
    std::string s = plan.summary();
    if (s.empty()) {
        std::printf("FAIL: Plan summary is empty\n");
        return false;
    }

    std::printf("PASS: Plan IR basic\n");
    return true;
}

//=============================================================================
// D3: StaticPlanner tests
//=============================================================================

static bool test_static_planner_convergence() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 1;
    pcfg.blk = 4;

    StaticPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    // Verify plan covers all n coordinates
    uint64_t coverage = plan.total_updates();
    if (coverage != n) {
        std::printf("FAIL: StaticPlanner coverage = %" PRIu64 ", expected %u\n",
                    coverage, n);
        return false;
    }

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;  // check every epoch
    cfg.num_threads = 1;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    const real_t expected = 1.0 / (1.0 - beta);
    return check_convergence("StaticPlanner ring convergence", result, x.data(), n, expected, eps);
}

//=============================================================================
// D4: ColoredPlanner tests
//=============================================================================

static bool test_colored_planner_convergence() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 2;
    pcfg.blk = 4;
    pcfg.colors = 2;

    ColoredPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    // Verify plan has the right number of phases (= colors)
    if (plan.phases.size() != 2) {
        std::printf("FAIL: ColoredPlanner phases = %zu, expected 2\n",
                    plan.phases.size());
        return false;
    }

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;
    cfg.num_threads = 2;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    const real_t expected = 1.0 / (1.0 - beta);
    return check_convergence("ColoredPlanner ring convergence", result, x.data(), n, expected, eps);
}

static bool test_colored_planner_multithread() {
    constexpr index_t n = 256;
    constexpr real_t beta = 0.95;
    constexpr real_t eps = 1e-5;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 4;
    pcfg.blk = 16;
    pcfg.colors = 4;

    ColoredPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 30.0;
    cfg.monitor_interval_ms = 50;
    cfg.num_threads = 4;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    const real_t expected = 1.0 / (1.0 - beta);
    return check_convergence("ColoredPlanner multithread", result, x.data(), n, expected, eps);
}

//=============================================================================
// D6: PriorityPlanner tests
//=============================================================================

static bool test_priority_planner_convergence() {
    constexpr index_t n = 64;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 1;
    pcfg.blk = 8;
    pcfg.colors = 2;
    pcfg.K = 16;
    pcfg.hot_phase_enabled = true;

    PriorityPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    // Should have hot phases + coverage phases
    if (plan.phases.size() < 2) {
        std::printf("FAIL: PriorityPlanner phases = %zu, expected >= 2\n",
                    plan.phases.size());
        return false;
    }

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;
    cfg.num_threads = 1;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    const real_t expected = 1.0 / (1.0 - beta);
    return check_convergence("PriorityPlanner ring convergence", result, x.data(), n, expected, eps);
}

static bool test_priority_planner_metastable() {
    constexpr index_t n = 128;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-5;

    MDP mdp = build_metastable_mdp(n, beta, 0.95, 0.05, 1.0, 2.0, 42);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    // Start from some non-zero point to get informative residuals
    std::vector<real_t> x(n, 5.0);

    PlannerConfig pcfg;
    pcfg.threads = 2;
    pcfg.blk = 8;
    pcfg.colors = 2;
    pcfg.K = 32;
    pcfg.hot_phase_enabled = true;

    PriorityPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 30.0;
    cfg.monitor_interval_ms = 50;
    cfg.num_threads = 2;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: PriorityPlanner metastable did not converge\n");
        std::printf("  final_residual_inf = %.9e\n", result.final_residual_inf);
        return false;
    }

    // Verify Bellman equation at convergence
    real_t max_bellman_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t fi = op.apply_i(i, x.data());
        real_t err = std::abs(fi - x[i]);
        if (err > max_bellman_err) max_bellman_err = err;
    }

    if (max_bellman_err > eps * 10) {
        std::printf("FAIL: PriorityPlanner metastable Bellman error = %.9e\n", max_bellman_err);
        return false;
    }

    std::printf("PASS: PriorityPlanner metastable convergence\n");
    std::printf("  converged in %.3f sec, %" PRIu64 " updates\n",
                result.wall_time_sec, result.total_updates);
    return true;
}

//=============================================================================
// D7: Profiling counters tests
//=============================================================================

static bool test_profiling_counters() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 1;
    pcfg.blk = 4;

    StaticPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Profiling test did not converge\n");
        return false;
    }

    // Check that profiling counters are populated
    if (result.profiling.per_thread.empty()) {
        std::printf("FAIL: Profiling per_thread empty\n");
        return false;
    }

    if (result.profiling.per_thread[0].updates_completed == 0) {
        std::printf("FAIL: Profiling updates_completed == 0\n");
        return false;
    }

    if (result.profiling.num_residual_scans == 0) {
        std::printf("FAIL: Profiling num_residual_scans == 0\n");
        return false;
    }

    if (result.profiling.total_updates == 0) {
        std::printf("FAIL: Profiling total_updates == 0\n");
        return false;
    }

    std::printf("PASS: Profiling counters\n");
    std::printf("  %s", result.profiling.summary().c_str());
    return true;
}

//=============================================================================
// D8: Cost model tests
//=============================================================================

static bool test_cost_model() {
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;

    MDP mdp = build_ring_mdp(n, beta);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 2;
    pcfg.blk = 4;

    StaticPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    // Populate weights
    populate_task_weights(plan, &mdp);

    // Check weights are non-zero
    bool has_weight = false;
    for (auto& ph : plan.phases) {
        for (auto& wl : ph.worklist) {
            for (auto& t : wl) {
                if (t.weight > 0.0) has_weight = true;
            }
        }
    }

    if (!has_weight) {
        std::printf("FAIL: Cost model - no task weights after populate\n");
        return false;
    }

    // Estimate cost
    CostModelConfig cost_cfg;
    CostEstimate est = estimate_plan_cost(plan, &mdp, cost_cfg);

    if (est.estimated_cost <= 0.0) {
        std::printf("FAIL: Cost model estimated_cost = %.2f\n", est.estimated_cost);
        return false;
    }

    if (est.bottleneck_cost <= 0.0) {
        std::printf("FAIL: Cost model bottleneck_cost = %.2f\n", est.bottleneck_cost);
        return false;
    }

    std::printf("PASS: Cost model\n");
    std::printf("  %s\n", est.summary().c_str());
    return true;
}

//=============================================================================
// D9: Autotune tests
//=============================================================================

static bool test_autotune() {
    constexpr index_t n = 64;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-5;

    MDP mdp = build_ring_mdp(n, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    std::vector<real_t> x(n, 0.0);

    AutotuneConfig atcfg;
    atcfg.blk_candidates = {8, 16};          // small set for test speed
    atcfg.color_multipliers = {1, 2};
    atcfg.K_fractions = {0.02, 0.05};
    atcfg.pilot_seconds = 0.2;
    atcfg.top_M = 2;
    atcfg.runtime_cfg.num_threads = 1;
    atcfg.runtime_cfg.alpha = 1.0;
    atcfg.runtime_cfg.eps = eps;
    atcfg.runtime_cfg.max_seconds = 10.0;
    atcfg.runtime_cfg.monitor_interval_ms = 0;

    AutotuneResult at_result = autotune(op, &mdp, x.data(), atcfg);

    if (at_result.all_candidates.empty()) {
        std::printf("FAIL: Autotune produced no candidates\n");
        return false;
    }

    if (at_result.best_plan.phases.empty()) {
        std::printf("FAIL: Autotune best_plan has no phases\n");
        return false;
    }

    // Now run the autotuned plan to convergence
    std::fill(x.begin(), x.end(), 0.0);

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 10.0;
    cfg.monitor_interval_ms = 0;
    cfg.num_threads = 1;

    Runtime rt;
    RunResult result = rt.run_plan(op, at_result.best_plan, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Autotune plan did not converge\n");
        return false;
    }

    const real_t expected = 1.0 / (1.0 - beta);
    real_t max_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t err = std::abs(x[i] - expected);
        if (err > max_err) max_err = err;
    }

    if (max_err > eps * 10) {
        std::printf("FAIL: Autotune solution error = %.9e\n", max_err);
        return false;
    }

    std::printf("PASS: Autotune\n");
    std::printf("  %s", at_result.summary().c_str());
    std::printf("  converged in %.3f sec, %" PRIu64 " updates\n",
                result.wall_time_sec, result.total_updates);
    return true;
}

//=============================================================================
// Test: run_plan matches scheduler mode
//=============================================================================

static bool test_plan_matches_scheduler() {
    // Verify that run_plan with StaticPlanner reaches same result as
    // Gauss-Seidel mode (single-threaded, same traversal order)
    constexpr index_t n = 16;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-6;

    MDP mdp = build_ring_mdp(n, beta);
    PolicyEvalOp op(&mdp);

    // Run Gauss-Seidel baseline
    std::vector<real_t> x_gs(n, 0.0);
    RuntimeConfig gs_cfg;
    gs_cfg.mode = Mode::GaussSeidel;
    gs_cfg.alpha = 1.0;
    gs_cfg.eps = eps;
    gs_cfg.max_seconds = 10.0;
    gs_cfg.monitor_interval_ms = 0;

    Runtime rt;
    StaticBlocksScheduler sched;
    RunResult gs_result = rt.run(op, sched, x_gs.data(), gs_cfg);

    // Run plan mode with StaticPlanner (single thread)
    std::vector<real_t> x_plan(n, 0.0);
    PlannerConfig pcfg;
    pcfg.threads = 1;
    pcfg.blk = n;  // one big block = sequential

    StaticPlanner planner;
    EpochPlan plan = planner.build(op, x_plan.data(), pcfg);

    RuntimeConfig plan_cfg;
    plan_cfg.alpha = 1.0;
    plan_cfg.eps = eps;
    plan_cfg.max_seconds = 10.0;
    plan_cfg.monitor_interval_ms = 0;

    RunResult plan_result = rt.run_plan(op, plan, x_plan.data(), plan_cfg);

    if (!gs_result.converged || !plan_result.converged) {
        std::printf("FAIL: plan_matches_scheduler - one mode didn't converge\n");
        return false;
    }

    // Both should reach same solution
    real_t max_diff = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t diff = std::abs(x_gs[i] - x_plan[i]);
        if (diff > max_diff) max_diff = diff;
    }

    if (max_diff > eps * 100) {
        std::printf("FAIL: plan vs scheduler max_diff = %.9e\n", max_diff);
        return false;
    }

    std::printf("PASS: Plan matches scheduler mode\n");
    std::printf("  max_diff = %.9e\n", max_diff);
    return true;
}

//=============================================================================
// Test: Grid MDP with plan executor
//=============================================================================

static bool test_plan_grid_mdp() {
    constexpr index_t rows = 8, cols = 8;
    constexpr real_t beta = 0.9;
    constexpr real_t eps = 1e-5;

    MDP mdp = build_grid_mdp(rows, cols, beta);
    mdp.validate(true);
    PolicyEvalOp op(&mdp);

    const index_t n = rows * cols;
    std::vector<real_t> x(n, 0.0);

    PlannerConfig pcfg;
    pcfg.threads = 2;
    pcfg.blk = 8;
    pcfg.colors = 2;

    ColoredPlanner planner;
    EpochPlan plan = planner.build(op, x.data(), pcfg);

    RuntimeConfig cfg;
    cfg.alpha = 1.0;
    cfg.eps = eps;
    cfg.max_seconds = 30.0;
    cfg.monitor_interval_ms = 50;
    cfg.num_threads = 2;

    Runtime rt;
    RunResult result = rt.run_plan(op, plan, x.data(), cfg);

    if (!result.converged) {
        std::printf("FAIL: Plan Grid MDP did not converge\n");
        std::printf("  final_residual = %.9e\n", result.final_residual_inf);
        return false;
    }

    // Verify Bellman equation
    real_t max_err = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t fi = op.apply_i(i, x.data());
        real_t err = std::abs(fi - x[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > eps * 10) {
        std::printf("FAIL: Grid MDP Bellman error = %.9e\n", max_err);
        return false;
    }

    std::printf("PASS: Plan Grid MDP convergence\n");
    std::printf("  converged in %.3f sec, %" PRIu64 " updates\n",
                result.wall_time_sec, result.total_updates);
    return true;
}

//=============================================================================
// Entry point
//=============================================================================

int run_phase3_tests() {
    int failures = 0;

    std::printf("\n=== Phase 3 Tests ===\n\n");

    // D1: Schedule IR
    if (!test_plan_ir_basic()) failures++;

    // D3: StaticPlanner
    if (!test_static_planner_convergence()) failures++;

    // D4: ColoredPlanner
    if (!test_colored_planner_convergence()) failures++;
    if (!test_colored_planner_multithread()) failures++;

    // D5: Plan executor correctness
    if (!test_plan_matches_scheduler()) failures++;
    if (!test_plan_grid_mdp()) failures++;

    // D6: PriorityPlanner
    if (!test_priority_planner_convergence()) failures++;
    if (!test_priority_planner_metastable()) failures++;

    // D7: Profiling
    if (!test_profiling_counters()) failures++;

    // D8: Cost model
    if (!test_cost_model()) failures++;

    // D9: Autotune
    if (!test_autotune()) failures++;

    return failures;
}
