// bench/run_bench.cc — Helios comprehensive benchmark runner
// Outputs CSV files for convergence traces, performance metrics, and thread scaling.
//
// Usage:
//   ./helios_bench                    # run all benchmarks (full mode, ~3 min)
//   ./helios_bench --quick            # smaller subset (~30s)
//   ./helios_bench --outdir /tmp/out  # custom output directory

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

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;

static std::string g_outdir = "bench/results";
static bool g_quick = false;

// ─── Monitoring interval: 2ms for dense convergence traces ──────────────────
static int g_monitor_ms = 2;
static int g_rebuild_ms = 50;
static int g_stride = 1;
static bool g_record_trace = true;

struct BenchEntry {
    std::string mdp_name;
    std::string solver_name;
    index_t n;
    double beta;
    double wall_sec;
    uint64_t total_updates;
    double updates_per_sec;
    double final_residual;
    bool converged;
    size_t threads;
    std::vector<ResidualSample> trace;
};

static FILE* open_csv(const char* fname, const char* hdr) {
    std::string path = g_outdir + "/" + fname;
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); return nullptr; }
    std::fprintf(f, "%s\n", hdr);
    return f;
}

static BenchEntry run_sched(const MDP& mdp, const char* mn, Scheduler& sc, const char* sn,
                             Mode mode, size_t T, real_t eps, double maxs) {
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);
    RuntimeConfig cfg;
    cfg.mode = mode; cfg.num_threads = T; cfg.alpha = 1.0; cfg.eps = eps;
    cfg.max_seconds = maxs; cfg.monitor_interval_ms = g_monitor_ms;
    cfg.rebuild_interval_ms = g_rebuild_ms;
    cfg.residual_scan_stride = g_stride;
    cfg.record_trace = g_record_trace;
    Runtime rt;
    RunResult rr = rt.run(op, sc, x.data(), cfg);
    return {mn, sn, mdp.n, mdp.beta, rr.wall_time_sec, rr.total_updates,
            rr.updates_per_sec, rr.final_residual_inf, rr.converged, T, std::move(rr.trace)};
}

static BenchEntry run_plan(const MDP& mdp, const char* mn, Planner& pl, const PlannerConfig& pc,
                            const char* pn, size_t T, real_t eps, double maxs) {
    PolicyEvalOp op(&mdp);
    std::vector<real_t> x(mdp.n, 0.0);
    EpochPlan plan = pl.build(op, x.data(), pc);
    populate_task_weights(plan, &mdp);
    RuntimeConfig cfg;
    cfg.alpha = 1.0; cfg.eps = eps; cfg.max_seconds = maxs; cfg.num_threads = T;
    cfg.monitor_interval_ms = g_monitor_ms; cfg.residual_scan_stride = g_stride;
    cfg.record_trace = g_record_trace;
    Runtime rt;
    RunResult rr = rt.run_plan(op, plan, x.data(), cfg);
    return {mn, pn, mdp.n, mdp.beta, rr.wall_time_sec, rr.total_updates,
            rr.updates_per_sec, rr.final_residual_inf, rr.converged, T, std::move(rr.trace)};
}

static void write_trace(FILE* f, const BenchEntry& e) {
    for (auto& s : e.trace)
        std::fprintf(f, "%s,%s,%u,%.4f,%zu,%.6f,%.12e\n",
                     e.mdp_name.c_str(), e.solver_name.c_str(), e.n, e.beta, e.threads, s.time_sec, s.residual);
}
static void write_summary(FILE* f, const BenchEntry& e) {
    std::fprintf(f, "%s,%s,%u,%.4f,%zu,%s,%.6f,%" PRIu64 ",%.6e,%.6e\n",
                 e.mdp_name.c_str(), e.solver_name.c_str(), e.n, e.beta, e.threads,
                 e.converged ? "true" : "false", e.wall_sec, e.total_updates, e.updates_per_sec, e.final_residual);
}
static void pr(const BenchEntry& e) {
    std::printf("    %-25s %s  %.3fs  %.2e ups  res=%.2e  updates=%" PRIu64 "\n",
                e.solver_name.c_str(), e.converged?"CONV":"FAIL", e.wall_sec, e.updates_per_sec,
                e.final_residual, e.total_updates);
}

#define EMIT(e) do { pr(e); write_trace(tf,e); write_summary(sf,e); } while(0)

// ─── Bench 1: Convergence comparison ────────────────────────────────────────
// Uses large problems with high beta to produce rich convergence traces.
// Full: n=5K-10K, beta=0.999, eps=1e-8  → solve times 0.5-5s
// Quick: n=2K-4K, beta=0.999, eps=1e-6  → solve times 0.1-1s

static void bench_convergence(FILE* tf, FILE* sf) {
    std::printf("\n====== Bench 1: Convergence Comparison ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 30.0 : 120.0;
    const size_t T = 4;

    struct MC { const char* n; MDP m; };
    std::vector<MC> mdps;

    if (g_quick) {
        mdps.push_back({"Grid_50x50",  build_grid_mdp(50, 50, 0.999, 0.2, 1.0, 0.5)});
        mdps.push_back({"Meta_2K",     build_metastable_mdp(2000, 0.999, 0.95, 0.05, 1.0, 2.0, 42)});
        mdps.push_back({"Star_2K",     build_star_mdp(2000, 0.999, 0.8, 1.0, 0.5)});
        mdps.push_back({"Chain_2K",    build_chain_mdp(2000, 0.999, 0.25, 0.5, 0.25, 1, true)});
        mdps.push_back({"Rand_4K",     build_random_sparse_mdp(4000, 8, 0.999, 1.0, 42)});
    } else {
        mdps.push_back({"Grid_100x100", build_grid_mdp(100, 100, 0.999, 0.2, 1.0, 0.5)});
        mdps.push_back({"Meta_5K",      build_metastable_mdp(5000, 0.999, 0.95, 0.05, 1.0, 2.0, 42)});
        mdps.push_back({"Star_5K",      build_star_mdp(5000, 0.999, 0.8, 1.0, 0.5)});
        mdps.push_back({"Chain_5K",     build_chain_mdp(5000, 0.999, 0.25, 0.5, 0.25, 1, true)});
        mdps.push_back({"Rand_10K",     build_random_sparse_mdp(10000, 8, 0.999, 1.0, 42)});
        mdps.push_back({"MC_5K",        build_multi_cluster_mdp(5000, 5, 0.99, 0.9, {1,2,3,4,5}, 42)});
    }

    for (auto& mc : mdps) {
        std::printf("\n  %s (n=%u, beta=%.4f)\n", mc.n, mc.m.n, mc.m.beta);
        { StaticBlocksScheduler s; auto e=run_sched(mc.m,mc.n,s,"Jacobi",Mode::Jacobi,1,eps,ms); EMIT(e); }
        { StaticBlocksScheduler s; auto e=run_sched(mc.m,mc.n,s,"GaussSeidel",Mode::GaussSeidel,1,eps,ms); EMIT(e); }
        { StaticBlocksScheduler s; auto e=run_sched(mc.m,mc.n,s,"Async_Static",Mode::Async,T,eps,ms); EMIT(e); }
        { ShuffledBlocksScheduler s; auto e=run_sched(mc.m,mc.n,s,"Async_Shuffled",Mode::Async,T,eps,ms); EMIT(e); }
        { TopKGSScheduler::Params p; p.K=std::max(index_t(16),mc.m.n/10); TopKGSScheduler s(p);
          auto e=run_sched(mc.m,mc.n,s,"Async_TopKGS",Mode::Async,T,eps,ms); EMIT(e); }
        { CATopKGSScheduler::Params p; p.K=std::max(index_t(16),mc.m.n/10); p.G=T*2; CATopKGSScheduler s(p);
          auto e=run_sched(mc.m,mc.n,s,"Async_CATopKGS",Mode::Async,T,eps,ms); EMIT(e); }
        { ResidualBucketsScheduler::Params p; ResidualBucketsScheduler s(p);
          auto e=run_sched(mc.m,mc.n,s,"Async_ResBuck",Mode::Async,T,eps,ms); EMIT(e); }
        { StaticPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64;
          auto e=run_plan(mc.m,mc.n,pl,pc,"Plan_Static",T,eps,ms); EMIT(e); }
        { ColoredPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64; pc.colors=T;
          auto e=run_plan(mc.m,mc.n,pl,pc,"Plan_Colored",T,eps,ms); EMIT(e); }
        { PriorityPlanner pl; PlannerConfig pc; pc.threads=T; pc.blk=64; pc.colors=T;
          pc.K=std::max(index_t(16),mc.m.n/5); pc.hot_phase_enabled=true;
          auto e=run_plan(mc.m,mc.n,pl,pc,"Plan_Priority",T,eps,ms); EMIT(e); }
    }
}

// ─── Bench 2: Beta sensitivity ──────────────────────────────────────────────
// Demonstrates exponential growth in solve time as beta → 1.
// Uses Grid MDP with increasing beta values.

static void bench_beta(FILE* tf, FILE* sf) {
    std::printf("\n====== Bench 2: Beta Sensitivity ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 30.0 : 120.0;

    std::vector<double> betas = {0.9, 0.95, 0.99, 0.995};
    if (!g_quick) betas.push_back(0.999);

    // Grid size chosen so highest beta still converges in reasonable time
    index_t side = g_quick ? 30 : 50;

    for (double b : betas) {
        MDP mdp = build_grid_mdp(side, side, b, 0.2, 1.0, 0.5);
        char nm[64]; std::snprintf(nm, sizeof(nm), "Grid_b%.3f", b);
        std::printf("\n  beta=%.3f (n=%u)\n", b, mdp.n);
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"Jacobi",Mode::Jacobi,1,eps,ms); EMIT(e); }
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"GaussSeidel",Mode::GaussSeidel,1,eps,ms); EMIT(e); }
        { TopKGSScheduler::Params p; p.K=std::max(index_t(16),mdp.n/10); TopKGSScheduler s(p);
          auto e=run_sched(mdp,nm,s,"Async_TopKGS",Mode::Async,4,eps,ms); EMIT(e); }
        { PriorityPlanner pl; PlannerConfig pc; pc.threads=4; pc.blk=32; pc.colors=4;
          pc.K=std::max(index_t(16),mdp.n/5); pc.hot_phase_enabled=true;
          auto e=run_plan(mdp,nm,pl,pc,"Plan_Priority",4,eps,ms); EMIT(e); }
    }
}

// ─── Bench 3: Thread scaling ────────────────────────────────────────────────
// Measures speedup from 1→8 threads on large problems.
//
// KEY: The residual scan is O(n) and runs serially on the main thread (Plan)
// or on a competing monitor thread (Async). With frequent monitoring (2ms)
// and stride=1, the scan dominates wall time and kills scaling. For this
// bench we use: monitor_interval=2000ms, stride=16, no trace recording.
// This isolates the parallel compute scaling from monitoring overhead.

static void bench_threads(FILE* sf) {
    std::printf("\n====== Bench 3: Thread Scaling ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 120.0 : 300.0;

    // Save globals — thread scaling needs minimal monitoring overhead
    const int saved_ms = g_monitor_ms;
    const int saved_stride = g_stride;
    const bool saved_trace = g_record_trace;
    g_monitor_ms = 200;       // check every 200ms: good time resolution with low overhead
    g_stride = 16;             // sample 1/16 of states per scan (~2-4ms per scan at n=1M)
    g_record_trace = false;    // no trace vector overhead

    struct SC { const char* n; MDP m; };
    std::vector<SC> cs;

    if (g_quick) {
        cs.push_back({"Rand_500K", build_random_sparse_mdp(500000, 20, 0.99, 1.0, 42)});
        cs.push_back({"Rand_1M",   build_random_sparse_mdp(1000000, 20, 0.99, 1.0, 42)});
    } else {
        cs.push_back({"Rand_1M",   build_random_sparse_mdp(1000000, 20, 0.99, 1.0, 42)});
        cs.push_back({"Rand_2M",   build_random_sparse_mdp(2000000, 10, 0.99, 1.0, 42)});
    }

    FILE* scf = open_csv("thread_scaling.csv",
        "mdp,solver,n,beta,threads,converged,wall_sec,total_updates,updates_per_sec,final_residual");
    if (!scf) return;

    std::vector<size_t> thread_counts = {1, 2, 4, 8};

    for (auto& sc : cs) {
        std::printf("\n  %s (n=%u, beta=%.4f)\n", sc.n, sc.m.n, sc.m.beta);

        for (size_t T : thread_counts) {
            // Plan_Static: barrier-synced workers, best for thread scaling
            { StaticPlanner pl; PlannerConfig pc; pc.threads=T;
              pc.blk=std::max(index_t(512), sc.m.n / (index_t(T) * 4));
              auto e=run_plan(sc.m,sc.n,pl,pc,"Plan_Static",T,eps,ms);
              pr(e); write_summary(scf,e); write_summary(sf,e); }

            // Async_Static: lock-free continuous workers
            { StaticBlocksScheduler s;
              auto e=run_sched(sc.m,sc.n,s,"Async_Static",Mode::Async,T,eps,ms);
              pr(e); write_summary(scf,e); write_summary(sf,e); }
        }
    }

    std::fclose(scf);

    // Restore globals
    g_monitor_ms = saved_ms;
    g_stride = saved_stride;
    g_record_trace = saved_trace;
}

// ─── Bench 4: Difficulty spectrum ───────────────────────────────────────────
// Sweeps metastable bridge probability: as p_bridge → 0, mixing slows
// and convergence becomes harder.

static void bench_difficulty(FILE* tf, FILE* sf) {
    std::printf("\n====== Bench 4: Difficulty Spectrum ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 30.0 : 120.0;
    const index_t meta_n = g_quick ? 2000 : 4000;

    for (double pb : {0.20, 0.10, 0.05, 0.02, 0.01}) {
        MDP mdp = build_metastable_mdp(meta_n, 0.999, 1.0-pb, pb, 1.0, 3.0, 42);
        char nm[64]; std::snprintf(nm, sizeof(nm), "Meta_pb%.3f", pb);
        std::printf("\n  p_bridge=%.3f (n=%u)\n", pb, mdp.n);
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"Jacobi",Mode::Jacobi,1,eps,ms); EMIT(e); }
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"GaussSeidel",Mode::GaussSeidel,1,eps,ms); EMIT(e); }
        { TopKGSScheduler::Params p; p.K=std::max(index_t(16),mdp.n/10); TopKGSScheduler s(p);
          auto e=run_sched(mdp,nm,s,"Async_TopKGS",Mode::Async,4,eps,ms); EMIT(e); }
        { PriorityPlanner pl; PlannerConfig pc; pc.threads=4; pc.blk=32; pc.colors=4;
          pc.K=std::max(index_t(16),mdp.n/5); pc.hot_phase_enabled=true;
          auto e=run_plan(mdp,nm,pl,pc,"Plan_Priority",4,eps,ms); EMIT(e); }
    }
}

// ─── Bench 5: Problem size scaling ──────────────────────────────────────────
// How solve time grows with n. Log-log plot should show near-linear scaling.
// Includes both single-threaded and multi-threaded solvers to show where
// parallelism starts to pay off.

static void bench_size(FILE* sf) {
    std::printf("\n====== Bench 5: Size Scaling ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 30.0 : 120.0;
    const size_t T = 4;

    std::vector<index_t> sizes = g_quick
        ? std::vector<index_t>{1000, 5000, 20000, 100000}
        : std::vector<index_t>{1000, 5000, 20000, 100000, 500000};

    FILE* szf = open_csv("size_scaling.csv",
        "mdp,solver,n,beta,threads,converged,wall_sec,total_updates,updates_per_sec,final_residual");
    if (!szf) return;

    for (index_t n : sizes) {
        MDP mdp = build_random_sparse_mdp(n, 20, 0.99, 1.0, 42);
        char nm[64]; std::snprintf(nm, sizeof(nm), "Rand_n%u", n);
        std::printf("\n  n=%u\n", n);
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"Jacobi",Mode::Jacobi,1,eps,ms);
          pr(e); write_summary(szf,e); write_summary(sf,e); }
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"GaussSeidel",Mode::GaussSeidel,1,eps,ms);
          pr(e); write_summary(szf,e); write_summary(sf,e); }
        { StaticPlanner pl; PlannerConfig pc; pc.threads=T;
          pc.blk=std::max(index_t(256), n/(index_t(T)*4));
          auto e=run_plan(mdp,nm,pl,pc,"Plan_Static_4T",T,eps,ms);
          pr(e); write_summary(szf,e); write_summary(sf,e); }
        { StaticBlocksScheduler s; auto e=run_sched(mdp,nm,s,"Async_Static_4T",Mode::Async,T,eps,ms);
          pr(e); write_summary(szf,e); write_summary(sf,e); }
    }
    std::fclose(szf);
}

// ─── Bench 6: Autotune ─────────────────────────────────────────────────────

static void bench_autotune(FILE* sf) {
    std::printf("\n====== Bench 6: Autotune ======\n");
    const real_t eps = g_quick ? 1e-6 : 1e-8;
    const double ms = g_quick ? 30.0 : 120.0;

    struct AC { const char* n; MDP m; };
    std::vector<AC> cs;
    cs.push_back({"Grid_AT",  build_grid_mdp(50,50,0.999)});
    cs.push_back({"Meta_AT",  build_metastable_mdp(2000,0.999,0.95,0.05,1,2,42)});
    cs.push_back({"Rand_AT",  build_random_sparse_mdp(5000,8,0.999,1,42)});

    FILE* af = open_csv("autotune.csv",
        "mdp,planner,blk,colors,K,est_cost,pilot_ups,rdrop,at_sec,converged,wall_sec,ups");
    if (!af) return;

    for (auto& ac : cs) {
        std::printf("\n  %s (n=%u)\n", ac.n, ac.m.n);
        PolicyEvalOp op(&ac.m);
        std::vector<real_t> x(ac.m.n, 0.0);

        AutotuneConfig atcfg;
        atcfg.blk_candidates = {16,32,64,128};
        atcfg.color_multipliers = {1,2,4};
        atcfg.K_fractions = {0.01,0.05,0.10};
        atcfg.pilot_seconds = g_quick ? 0.3 : 0.5;
        atcfg.top_M = 3;
        atcfg.runtime_cfg.num_threads = 4;
        atcfg.runtime_cfg.alpha = 1.0;
        atcfg.runtime_cfg.eps = eps;
        atcfg.runtime_cfg.max_seconds = ms;
        atcfg.runtime_cfg.monitor_interval_ms = g_monitor_ms;

        AutotuneResult at = autotune(op, &ac.m, x.data(), atcfg);
        std::fill(x.begin(), x.end(), 0.0);

        RuntimeConfig cfg; cfg.alpha=1; cfg.eps=eps; cfg.max_seconds=ms;
        cfg.num_threads=4; cfg.monitor_interval_ms=g_monitor_ms;
        Runtime rt;
        RunResult rr = rt.run_plan(op, at.best_plan, x.data(), cfg);

        std::printf("    => %s blk=%u C=%u K=%u  %s %.3fs\n",
                    at.best.planner_name.c_str(), at.best.planner_cfg.blk,
                    at.best.planner_cfg.colors, at.best.planner_cfg.K,
                    rr.converged?"CONV":"FAIL", rr.wall_time_sec);

        std::fprintf(af, "%s,%s,%u,%u,%u,%.2f,%.2f,%.6f,%.3f,%s,%.6f,%.2e\n",
                     ac.n, at.best.planner_name.c_str(), at.best.planner_cfg.blk,
                     at.best.planner_cfg.colors, at.best.planner_cfg.K,
                     at.best.cost_est.estimated_cost, at.best.pilot_updates_per_sec,
                     at.best.pilot_residual_drop, at.autotune_time_sec,
                     rr.converged?"true":"false", rr.wall_time_sec, rr.updates_per_sec);

        BenchEntry e{ac.n, "AT_"+at.best.planner_name, ac.m.n, ac.m.beta,
                     rr.wall_time_sec, rr.total_updates, rr.updates_per_sec,
                     rr.final_residual_inf, rr.converged, 4, {}};
        write_summary(sf, e);
    }
    std::fclose(af);
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string only_bench;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) g_quick = true;
        else if (std::strcmp(argv[i], "--outdir") == 0 && i+1 < argc) g_outdir = argv[++i];
        else if (std::strcmp(argv[i], "--only") == 0 && i+1 < argc) only_bench = argv[++i];
    }
    fs::create_directories(g_outdir);

    if (g_quick) g_monitor_ms = 5;  // slightly coarser for quick mode

    std::printf("Helios Benchmark Suite  [%s]  monitor=%dms\nOutput: %s/\n\n",
                g_quick?"quick":"full", g_monitor_ms, g_outdir.c_str());

    auto t0 = std::chrono::steady_clock::now();

    FILE* tf = open_csv("convergence_traces.csv", "mdp,solver,n,beta,threads,time_sec,residual");
    FILE* sf = open_csv("summary.csv",
        "mdp,solver,n,beta,threads,converged,wall_sec,total_updates,updates_per_sec,final_residual");
    if (!tf || !sf) return 1;

    const bool all = only_bench.empty();
    if (all || only_bench == "convergence") bench_convergence(tf, sf);
    if (all || only_bench == "beta") bench_beta(tf, sf);
    if (all || only_bench == "threads") bench_threads(sf);
    if (all || only_bench == "difficulty") bench_difficulty(tf, sf);
    if (all || only_bench == "size") bench_size(sf);
    if (all || only_bench == "autotune") bench_autotune(sf);

    std::fclose(tf); std::fclose(sf);
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("\n====== DONE (%.1fs) ======\nPlot: python3 tools/plot.py %s\n", el, g_outdir.c_str());
    return 0;
}
