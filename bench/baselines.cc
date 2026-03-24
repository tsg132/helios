// bench/baselines.cc — Baseline solver comparison for Helios
//
// Implements four Jacobi variants with zero Helios scheduler overhead:
//   Naive_Jacobi       — single-threaded double-buffer loop
//   RawParallel_Jacobi — std::thread + std::barrier (C++20), contiguous blocks
//   OpenMP_Jacobi      — #pragma omp parallel for (if HELIOS_HAS_OPENMP)
//   Eigen_Jacobi       — Eigen SpMV: x_new = r + beta_P * x (if HELIOS_HAS_EIGEN)
//
// Benchmarks on:
//   - Random sparse MDPs: n = {100K, 500K, 1M}, nnz/row=20, beta=0.99
//   - Metastable MDP:     n = 20K, p_bridge=0.01, beta=0.999 (convergence demo)
//
// Usage:
//   ./helios_baselines [--outdir <dir>] [--threads N] [--quick]
//
// Output: <outdir>/baselines.csv (same schema as summary.csv)

#include "helios/mdp.h"
#include "helios/mdp_generators.h"
#include "helios/types.h"

#ifdef HELIOS_HAS_OPENMP
#  include <omp.h>
#endif

#ifdef HELIOS_HAS_EIGEN
#  ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#    pragma clang diagnostic ignored "-Wshadow"
#  endif
#  include <Eigen/Sparse>
#  ifdef __clang__
#    pragma clang diagnostic pop
#  endif
#endif

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;
namespace chr = std::chrono;

static std::string g_outdir  = "bench/results";
static size_t      g_threads = 4;
static bool        g_quick   = false;

// ─── Result ───────────────────────────────────────────────────────────────────

struct BResult {
    std::string impl;
    index_t n;
    double beta;
    size_t threads;
    bool converged;
    double wall_sec;
    uint64_t total_updates;   // sweeps * n
    double updates_per_sec;
    double final_residual;
};

static void write_bresult(FILE* f, const BResult& r) {
    std::fprintf(f,
        "%s,%u,%.4f,%zu,%s,%.6f,%" PRIu64 ",%.6e,%.6e\n",
        r.impl.c_str(), r.n, r.beta, r.threads,
        r.converged ? "true" : "false",
        r.wall_sec, r.total_updates, r.updates_per_sec, r.final_residual);
}

static void print_bresult(const BResult& r) {
    std::printf("    %-25s T=%-2zu %s  %.3fs  %.2e ups  res=%.2e  upd=%" PRIu64 "\n",
        r.impl.c_str(), r.threads,
        r.converged ? "CONV" : "FAIL",
        r.wall_sec, r.updates_per_sec, r.final_residual, r.total_updates);
}

// ─── Helper: elapsed seconds ─────────────────────────────────────────────────

using TP = chr::time_point<chr::steady_clock>;

static inline double elapsed(TP t0) {
    return chr::duration<double>(chr::steady_clock::now() - t0).count();
}

// ─── 1. Naive Jacobi (single-threaded, double-buffer) ─────────────────────────
// Plain C++ loop — zero framework overhead, pure operator throughput.

static BResult run_naive_jacobi(const MDP& mdp, real_t eps, double max_sec) {
    const index_t n = mdp.n;
    std::vector<real_t> x(n, 0.0), xn(n, 0.0);

    TP t0 = chr::steady_clock::now();
    uint64_t sweeps = 0;
    real_t   resid  = std::numeric_limits<real_t>::max();

    while (true) {
        if (elapsed(t0) >= max_sec) break;

        real_t max_d = 0.0;
        for (index_t i = 0; i < n; ++i) {
            real_t fi = mdp.rewards[i];
            for (index_t k = mdp.row_ptr[i]; k < mdp.row_ptr[i + 1]; ++k)
                fi += mdp.beta * mdp.probs[k] * x[mdp.col_idx[k]];
            xn[i] = fi;
            max_d = std::max(max_d, std::abs(fi - x[i]));
        }
        std::swap(x, xn);
        resid = max_d;
        ++sweeps;
        if (resid <= eps) break;
    }

    double wall = elapsed(t0);
    uint64_t total = sweeps * (uint64_t)n;
    return {"Naive_Jacobi", n, mdp.beta, 1,
            resid <= eps, wall, total, total / wall, resid};
}

// ─── 2. Raw Parallel Jacobi (std::thread + std::barrier, C++20) ────────────────
// T persistent worker threads. Each owns a contiguous block of [0, n).
// Two barriers per epoch: b1 (workers done) → main swaps ptrs → b2 (resume).
// No Helios scheduler — raw parallel Jacobi with minimal overhead.

static BResult run_raw_parallel_jacobi(const MDP& mdp, size_t T, real_t eps, double max_sec) {
    const index_t n = mdp.n;
    std::vector<real_t> buf0(n, 0.0), buf1(n, 0.0);

    // Pointer-swap (not data-swap) each epoch: free after barriers fire.
    real_t* x_rd = buf0.data();   // workers read from here
    real_t* x_wr = buf1.data();   // workers write to here

    // Contiguous block partition
    std::vector<index_t> blk_b(T), blk_e(T);
    for (size_t t = 0; t < T; ++t) {
        blk_b[t] = (index_t)((uint64_t)t       * n / T);
        blk_e[t] = (index_t)((uint64_t)(t + 1) * n / T);
    }

    // Per-thread local max (cache-line padded to prevent false sharing)
    struct alignas(64) Pad { real_t val = 0.0; };
    std::vector<Pad> tmax(T);

    std::atomic<bool>  stop{false};
    // b1: all workers done computing this epoch; b2: main done swapping.
    std::barrier<> b1{(std::ptrdiff_t)(T + 1)};
    std::barrier<> b2{(std::ptrdiff_t)(T + 1)};

    std::vector<std::thread> workers;
    workers.reserve(T);
    for (size_t t = 0; t < T; ++t) {
        workers.emplace_back([&, t]() {
            for (;;) {
                // Snapshot current buffer pointers (main swaps between b1 and b2).
                const real_t* rd = x_rd;
                real_t*       wr = x_wr;

                real_t local_max = 0.0;
                for (index_t i = blk_b[t]; i < blk_e[t]; ++i) {
                    real_t fi = mdp.rewards[i];
                    for (index_t k = mdp.row_ptr[i]; k < mdp.row_ptr[i + 1]; ++k)
                        fi += mdp.beta * mdp.probs[k] * rd[mdp.col_idx[k]];
                    wr[i] = fi;
                    local_max = std::max(local_max, std::abs(fi - rd[i]));
                }
                tmax[t].val = local_max;

                b1.arrive_and_wait();  // signal: epoch compute done
                b2.arrive_and_wait();  // wait for main to swap + decide
                if (stop.load(std::memory_order_acquire)) break;
            }
        });
    }

    TP t0 = chr::steady_clock::now();
    uint64_t sweeps = 0;
    real_t   resid  = std::numeric_limits<real_t>::max();

    for (;;) {
        b1.arrive_and_wait();  // wait for all workers to finish this epoch

        // Aggregate residual across threads
        resid = 0.0;
        for (size_t t = 0; t < T; ++t)
            resid = std::max(resid, tmax[t].val);

        // Swap read/write pointers (workers see new values after b2)
        std::swap(x_rd, x_wr);
        ++sweeps;

        bool done = (resid <= eps || elapsed(t0) >= max_sec);
        if (done) stop.store(true, std::memory_order_release);

        b2.arrive_and_wait();  // release workers (they check stop immediately after)
        if (done) break;
    }

    for (auto& w : workers) w.join();

    double wall = elapsed(t0);
    uint64_t total = sweeps * (uint64_t)n;
    return {"RawParallel_Jacobi", n, mdp.beta, T,
            resid <= eps, wall, total, total / wall, resid};
}

// ─── 3. OpenMP Jacobi ──────────────────────────────────────────────────────────

#ifdef HELIOS_HAS_OPENMP
static BResult run_openmp_jacobi(const MDP& mdp, size_t T, real_t eps, double max_sec) {
    const index_t n   = mdp.n;
    const int64_t n64 = (int64_t)n;  // OpenMP requires signed loop variable
    std::vector<real_t> x(n, 0.0), xn(n, 0.0);

    omp_set_num_threads((int)T);

    TP t0 = chr::steady_clock::now();
    uint64_t sweeps = 0;
    real_t   resid  = std::numeric_limits<real_t>::max();

    while (true) {
        if (elapsed(t0) >= max_sec) break;

        const real_t* xr = x.data();
        real_t*       xw = xn.data();
        double max_d = 0.0;

        #pragma omp parallel for schedule(static) reduction(max: max_d)
        for (int64_t ii = 0; ii < n64; ++ii) {
            const index_t i  = (index_t)ii;
            real_t fi = mdp.rewards[i];
            for (index_t k = mdp.row_ptr[i]; k < mdp.row_ptr[i + 1]; ++k)
                fi += mdp.beta * mdp.probs[k] * xr[mdp.col_idx[k]];
            xw[i] = fi;
            max_d = std::max(max_d, (double)std::abs(fi - xr[i]));
        }

        std::swap(x, xn);
        resid = (real_t)max_d;
        ++sweeps;
        if (resid <= eps) break;
    }

    double wall = elapsed(t0);
    uint64_t total = sweeps * (uint64_t)n;
    return {"OpenMP_Jacobi", n, mdp.beta, T,
            resid <= eps, wall, total, total / wall, resid};
}
#endif  // HELIOS_HAS_OPENMP

// ─── 4. Eigen SpMV Jacobi ──────────────────────────────────────────────────────
// Converts MDP CSR → Eigen RowMajor SparseMatrix then iterates:
//   x_new = r + (beta * P) * x   (one Eigen SpMV call per sweep)

#ifdef HELIOS_HAS_EIGEN
static BResult run_eigen_jacobi(const MDP& mdp, real_t eps, double max_sec) {
    const index_t n = mdp.n;

    // Build row-major sparse matrix scaled by beta
    Eigen::SparseMatrix<double, Eigen::RowMajor> betaP(n, n);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(mdp.probs.size());
        for (index_t i = 0; i < n; ++i)
            for (index_t k = mdp.row_ptr[i]; k < mdp.row_ptr[i + 1]; ++k)
                trips.emplace_back((int)i, (int)mdp.col_idx[k],
                                   mdp.beta * (double)mdp.probs[k]);
        betaP.setFromTriplets(trips.begin(), trips.end());
        betaP.makeCompressed();
    }

    Eigen::VectorXd r = Eigen::Map<const Eigen::VectorXd>(mdp.rewards.data(), n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd xn(n);

    TP t0 = chr::steady_clock::now();
    uint64_t sweeps = 0;
    real_t   resid  = std::numeric_limits<real_t>::max();

    while (true) {
        if (elapsed(t0) >= max_sec) break;

        xn.noalias() = r + betaP * x;         // Eigen optimized SpMV
        resid = (real_t)(xn - x).cwiseAbs().maxCoeff();
        x.swap(xn);                            // O(1) pointer swap
        ++sweeps;
        if (resid <= eps) break;
    }

    double wall = elapsed(t0);
    uint64_t total = sweeps * (uint64_t)n;
    return {"Eigen_Jacobi", n, mdp.beta, 1,
            resid <= eps, wall, total, total / wall, resid};
}
#endif  // HELIOS_HAS_EIGEN

// ─── Benchmark runner ─────────────────────────────────────────────────────────

static void run_all(FILE* f, const MDP& mdp, const char* label,
                    real_t eps, double max_sec, size_t T) {
    std::printf("  %s (n=%u)\n", label, mdp.n);
    auto emit = [&](const BResult& r) { print_bresult(r); write_bresult(f, r); };

    emit(run_naive_jacobi(mdp, eps, max_sec));
    emit(run_raw_parallel_jacobi(mdp, T, eps, max_sec));
#ifdef HELIOS_HAS_OPENMP
    emit(run_openmp_jacobi(mdp, T, eps, max_sec));
#endif
#ifdef HELIOS_HAS_EIGEN
    emit(run_eigen_jacobi(mdp, eps, max_sec));
#endif
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--outdir") == 0 && i + 1 < argc)
            g_outdir = argv[++i];
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            g_threads = (size_t)std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--quick") == 0)
            g_quick = true;
    }

    fs::create_directories(g_outdir);

    std::string csv_path = g_outdir + "/baselines.csv";
    FILE* f = std::fopen(csv_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", csv_path.c_str()); return 1; }
    std::fprintf(f, "impl,n,beta,threads,converged,wall_sec,total_updates,updates_per_sec,final_residual\n");

    const real_t eps     = g_quick ? (real_t)1e-6 : (real_t)1e-8;
    const double max_sec = g_quick ? 60.0 : 300.0;
    const size_t T       = g_threads;

    std::printf("Helios Baseline Comparison\n");
    std::printf("  threads=%zu  eps=%.0e  max_sec=%.0f  quick=%s\n",
                T, (double)eps, max_sec, g_quick ? "yes" : "no");

    // ── Random sparse MDPs: throughput comparison ─────────────────────────────
    std::printf("\n=== Random Sparse MDPs (n varied, nnz/row=20, beta=0.99) ===\n");
    std::vector<index_t> sizes = g_quick
        ? std::vector<index_t>{100000, 500000}
        : std::vector<index_t>{100000, 500000, 1000000};

    for (index_t n : sizes) {
        MDP mdp = build_random_sparse_mdp(n, 20, 0.99, 1.0, 42);
        char label[64]; std::snprintf(label, sizeof(label), "Rand_n%u", n);
        run_all(f, mdp, label, eps, max_sec, T);
    }

    // ── Metastable MDP: convergence difficulty demo ───────────────────────────
    // All Jacobi-based baselines struggle on slow-mixing MDPs. This shows the
    // convergence challenge that Helios's priority scheduling is designed to solve.
    std::printf("\n=== Metastable MDP (p_bridge=0.01, beta=0.999) ===\n");
    {
        const index_t meta_n = g_quick ? 5000 : 20000;
        MDP mdp = build_metastable_mdp(meta_n, 0.999, 1.0 - 0.01, 0.01, 1.0, 3.0, 42);
        char label[64]; std::snprintf(label, sizeof(label), "Meta_n%u_pb0.01", meta_n);
        run_all(f, mdp, label, eps, max_sec, T);
    }

    std::fclose(f);
    std::printf("\nResults written to %s\n", csv_path.c_str());
    return 0;
}
