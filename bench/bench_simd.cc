// bench/bench_simd.cc — Scalar vs NEON kernel performance comparison.
// Usage: ./helios_bench_simd [--quick]

#include "helios/policy_eval_op.h"
#include "helios/policy_eval_op_neon.h"
#include "helios/simd/neon_kernels.h"
#include "helios/mdp_generators.h"
#include "helios/runtime.h"
#include "helios/schedulers/static_blocks.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace helios;
using Clock = std::chrono::steady_clock;

static bool g_quick = false;

// ── Bench A: Sparse dot product throughput (apply_i) ────────────────────
static void bench_sparse_dot() {
    std::printf("\n====== Bench A: Sparse Dot Product (apply_i per row) ======\n");
    std::printf("%-12s %8s %12s %12s %10s\n",
                "n", "nnz/row", "Scalar(ns)", "NEON(ns)", "Speedup");

    std::vector<index_t> nnz_vals = {4, 8, 16, 32};
    if (!g_quick) nnz_vals.push_back(64);

    const index_t n = g_quick ? 10000 : 50000;
    const int reps = g_quick ? 10 : 50;

    for (index_t nnz : nnz_vals) {
        MDP mdp = build_random_sparse_mdp(n, nnz, 0.99, 1.0, 42);
        PolicyEvalOp scalar_op(&mdp);
        PolicyEvalOpNeon neon_op(&mdp);
        std::vector<real_t> x(n, 1.0);

        // Warmup
        volatile real_t sink = 0;
        for (index_t i = 0; i < n; ++i) sink += scalar_op.apply_i(i, x.data());
        for (index_t i = 0; i < n; ++i) sink += neon_op.apply_i(i, x.data());

        // Scalar timing
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            for (index_t i = 0; i < n; ++i) sink += scalar_op.apply_i(i, x.data());
        double scalar_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count()
                           / (static_cast<double>(reps) * n);

        // NEON timing
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            for (index_t i = 0; i < n; ++i) sink += neon_op.apply_i(i, x.data());
        double neon_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count()
                         / (static_cast<double>(reps) * n);

        std::printf("%-12u %8u %12.1f %12.1f %9.2fx\n",
                    n, nnz, scalar_ns, neon_ns, scalar_ns / neon_ns);
    }
}

// ── Bench B: Full Jacobi sweep — scalar per-row vs NEON batch ──────────
static void bench_jacobi_sweep() {
    std::printf("\n====== Bench B: Jacobi Sweep (full epoch) ======\n");
    std::printf("%-12s %8s %12s %12s %10s\n",
                "n", "nnz/row", "Scalar(ms)", "NEON(ms)", "Speedup");

    std::vector<index_t> sizes = g_quick
        ? std::vector<index_t>{1000, 10000, 100000}
        : std::vector<index_t>{1000, 10000, 100000, 500000};

    const index_t nnz_per_row = 20;

    for (index_t n : sizes) {
        MDP mdp = build_random_sparse_mdp(n, nnz_per_row, 0.99, 1.0, 42);
        PolicyEvalOp scalar_op(&mdp);
        std::vector<real_t> x(n, 1.0), x_next(n);

        int reps = g_quick ? 3 : 10;
        if (n >= 500000) reps = 3;

        // Scalar sweep
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            for (index_t i = 0; i < n; ++i) {
                real_t fi = scalar_op.apply_i(i, x.data());
                x_next[i] = fi;
            }
        }
        double scalar_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        // NEON batch sweep
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            neon::neon_jacobi_sweep(mdp, x.data(), x_next.data(), 1.0);
        }
        double neon_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        std::printf("%-12u %8u %12.2f %12.2f %9.2fx\n",
                    n, nnz_per_row, scalar_ms, neon_ms, scalar_ms / neon_ms);
    }
}

// ── Bench C: Residual max — scalar residual_inf vs NEON batch ───────────
static void bench_residual_max() {
    std::printf("\n====== Bench C: Residual Max ======\n");
    std::printf("%-12s %8s %12s %12s %10s\n",
                "n", "nnz/row", "Scalar(ms)", "NEON(ms)", "Speedup");

    std::vector<index_t> sizes = g_quick
        ? std::vector<index_t>{1000, 10000, 100000}
        : std::vector<index_t>{1000, 10000, 100000, 500000};

    const index_t nnz_per_row = 20;

    for (index_t n : sizes) {
        MDP mdp = build_random_sparse_mdp(n, nnz_per_row, 0.99, 1.0, 42);
        PolicyEvalOp scalar_op(&mdp);
        std::vector<real_t> x(n, 1.0);

        int reps = g_quick ? 3 : 10;
        if (n >= 500000) reps = 3;

        volatile real_t sink = 0;

        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            sink = Runtime::residual_inf(scalar_op, x.data(), 1);
        double scalar_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            sink = neon::neon_residual_max(mdp, x.data());
        double neon_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        std::printf("%-12u %8u %12.2f %12.2f %9.2fx\n",
                    n, nnz_per_row, scalar_ms, neon_ms, scalar_ms / neon_ms);
    }
}

// ── Bench D: Full convergence wall time ─────────────────────────────────
static void bench_convergence() {
    std::printf("\n====== Bench D: Full Convergence (Jacobi, eps=1e-6) ======\n");

    const index_t n = g_quick ? 5000 : 50000;
    MDP mdp = build_random_sparse_mdp(n, 20, 0.99, 1.0, 42);

    RuntimeConfig cfg;
    cfg.mode = Mode::Jacobi;
    cfg.alpha = 1.0;
    cfg.eps = 1e-6;
    cfg.max_seconds = g_quick ? 30.0 : 120.0;
    cfg.monitor_interval_ms = 0;
    cfg.record_trace = false;

    std::printf("  MDP: Random n=%u, nnz/row=20, beta=0.99\n\n", n);
    std::printf("%-20s %10s %12s %16s %10s\n",
                "Operator", "Wall(s)", "Converged", "Updates", "UPS");

    // Scalar
    {
        PolicyEvalOp op(&mdp);
        std::vector<real_t> x(mdp.n, 0.0);
        Runtime rt;
        StaticBlocksScheduler sched;
        RunResult rr = rt.run(op, sched, x.data(), cfg);
        std::printf("%-20s %10.3f %12s %16" PRIu64 " %10.2e\n",
                    "Scalar", rr.wall_time_sec, rr.converged ? "yes" : "no",
                    rr.total_updates, rr.updates_per_sec);
    }

    // NEON operator through Runtime
    {
        PolicyEvalOpNeon op(&mdp);
        std::vector<real_t> x(mdp.n, 0.0);
        Runtime rt;
        StaticBlocksScheduler sched;
        RunResult rr = rt.run(op, sched, x.data(), cfg);
        std::printf("%-20s %10.3f %12s %16" PRIu64 " %10.2e\n",
                    "NEON (via Runtime)", rr.wall_time_sec, rr.converged ? "yes" : "no",
                    rr.total_updates, rr.updates_per_sec);
    }
}

// ── Bench E: Gauss-Seidel sweep — scalar vs NEON inner product ──────────
static void bench_gauss_seidel_sweep() {
    std::printf("\n====== Bench E: Gauss-Seidel Sweep ======\n");
    std::printf("%-12s %8s %12s %12s %10s\n",
                "n", "nnz/row", "Scalar(ms)", "NEON(ms)", "Speedup");

    std::vector<index_t> sizes = g_quick
        ? std::vector<index_t>{1000, 10000, 100000}
        : std::vector<index_t>{1000, 10000, 100000, 500000};

    const index_t nnz_per_row = 20;

    for (index_t n : sizes) {
        MDP mdp = build_random_sparse_mdp(n, nnz_per_row, 0.99, 1.0, 42);
        PolicyEvalOp scalar_op(&mdp);
        std::vector<real_t> x_scalar(n, 1.0), x_neon(n, 1.0);

        int reps = g_quick ? 3 : 10;
        if (n >= 500000) reps = 3;

        // Scalar GS sweep
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            for (index_t i = 0; i < n; ++i) {
                real_t fi = scalar_op.apply_i(i, x_scalar.data());
                x_scalar[i] = fi;
            }
        }
        double scalar_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        // NEON GS sweep
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            neon::neon_gauss_seidel_sweep(mdp, x_neon.data(), 1.0);
        }
        double neon_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;

        std::printf("%-12u %8u %12.2f %12.2f %9.2fx\n",
                    n, nnz_per_row, scalar_ms, neon_ms, scalar_ms / neon_ms);
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) g_quick = true;

    std::printf("Helios SIMD Benchmark  [%s mode]\n", g_quick ? "quick" : "full");
    std::printf("========================================\n");

#ifdef __ARM_NEON
    std::printf("ARM NEON: ENABLED\n");
#else
    std::printf("ARM NEON: NOT AVAILABLE (running scalar fallbacks)\n");
#endif

    bench_sparse_dot();
    bench_jacobi_sweep();
    bench_residual_max();
    bench_gauss_seidel_sweep();
    bench_convergence();

    std::printf("\nDone.\n");
    return 0;
}
