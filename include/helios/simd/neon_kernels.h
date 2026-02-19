#pragma once

// NEON SIMD kernel functions for Helios hot loops.
// All functions are inline, guarded by __ARM_NEON with scalar fallbacks.
// Primary target: Apple Silicon (ARMv8.4+), real_t = double, 128-bit NEON = 2x f64.

#include "helios/types.h"
#include "helios/mdp.h"

#include <cmath>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace helios {
namespace neon {

// ── Kernel 1: Sparse dot product (single row) ──────────────────────────
//
// Computes: dot = sum_{k=start}^{end-1} probs[k] * x[col_idx[k]]
//
// Strategy: unroll by 4 with two independent NEON FMA accumulators.
// probs[] is contiguous (good for vld1q_f64). x[] values are gathered
// individually via col_idx (manual gather — no NEON gather for f64).
// Two accumulator chains (acc0, acc1) hide the 4-cycle FMA latency on
// Apple M-series by keeping both FMA pipes fed.

inline real_t neon_sparse_dot(const real_t* __restrict__ probs,
                              const index_t* __restrict__ col_idx,
                              const real_t* __restrict__ x,
                              index_t start,
                              index_t end)
{
#ifdef __ARM_NEON

    const index_t len = end - start;
    const real_t* p = probs + start;
    const index_t* ci = col_idx + start;

    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);

    index_t k = 0;
    const index_t len4 = len & ~index_t(3);

    for (; k < len4; k += 4) {
        // Load 2+2 contiguous probabilities
        float64x2_t p0 = vld1q_f64(p + k);
        float64x2_t p1 = vld1q_f64(p + k + 2);

        // Manual gather: load x values via indirect indexing
        float64x2_t x0 = vcombine_f64(vld1_f64(&x[ci[k]]),
                                       vld1_f64(&x[ci[k + 1]]));
        float64x2_t x1 = vcombine_f64(vld1_f64(&x[ci[k + 2]]),
                                       vld1_f64(&x[ci[k + 3]]));

        acc0 = vfmaq_f64(acc0, p0, x0);
        acc1 = vfmaq_f64(acc1, p1, x1);
    }

    // Combine accumulators and horizontal sum
    float64x2_t sum2 = vaddq_f64(acc0, acc1);
    real_t dot = vaddvq_f64(sum2);

    // Scalar tail: remaining 0-3 elements
    for (; k < len; ++k) {
        dot += p[k] * x[ci[k]];
    }

    return dot;

#else  // scalar fallback

    real_t dot = 0.0;
    for (index_t k = start; k < end; ++k) {
        dot += probs[k] * x[col_idx[k]];
    }
    return dot;

#endif
}


// ── Kernel 2: Full Jacobi sweep ────────────────────────────────────────
//
// x_next[i] = (1 - alpha) * x_curr[i] + alpha * (rewards[i] + beta * dot_i)
//
// Processes rows in pairs for vectorized alpha blending.
// Eliminates n virtual calls by computing apply_i inline.

inline void neon_jacobi_sweep(const MDP& mdp,
                              const real_t* __restrict__ x_curr,
                              real_t* __restrict__ x_next,
                              real_t alpha)
{
    const index_t n = mdp.n;
    const real_t beta = mdp.beta;
    const real_t one_minus_alpha = 1.0 - alpha;

#ifdef __ARM_NEON

    const float64x2_t v_alpha = vdupq_n_f64(alpha);
    const float64x2_t v_oma = vdupq_n_f64(one_minus_alpha);
    const float64x2_t v_beta = vdupq_n_f64(beta);

    index_t i = 0;
    const index_t n2 = n & ~index_t(1);

    for (; i < n2; i += 2) {
        // Compute F_i for two consecutive rows
        real_t dot0 = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                      x_curr, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t dot1 = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                      x_curr, mdp.row_ptr[i + 1], mdp.row_ptr[i + 2]);

        // fi = rewards[i] + beta * dot
        float64x2_t v_rewards = vld1q_f64(&mdp.rewards[i]);
        float64x2_t v_dots = vcombine_f64(vdup_n_f64(dot0), vdup_n_f64(dot1));
        float64x2_t v_fi = vfmaq_f64(v_rewards, v_beta, v_dots);

        // x_next = (1-alpha)*x_curr + alpha*fi
        float64x2_t v_xcurr = vld1q_f64(x_curr + i);
        float64x2_t result = vfmaq_f64(vmulq_f64(v_oma, v_xcurr), v_alpha, v_fi);
        vst1q_f64(x_next + i, result);
    }

    // Scalar tail for odd n
    if (i < n) {
        real_t dot = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                     x_curr, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t fi = mdp.rewards[i] + beta * dot;
        x_next[i] = one_minus_alpha * x_curr[i] + alpha * fi;
    }

#else  // scalar fallback

    for (index_t i = 0; i < n; ++i) {
        real_t dot = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                     x_curr, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t fi = mdp.rewards[i] + beta * dot;
        x_next[i] = one_minus_alpha * x_curr[i] + alpha * fi;
    }

#endif
}


// ── Kernel 3: Batch residual max-reduction ─────────────────────────────
//
// Computes max_i |F_i(x) - x[i]| over all rows using NEON max reduction.

inline real_t neon_residual_max(const MDP& mdp, const real_t* __restrict__ x)
{
    const index_t n = mdp.n;
    const real_t beta = mdp.beta;

#ifdef __ARM_NEON

    float64x2_t vmax = vdupq_n_f64(0.0);

    index_t i = 0;
    const index_t n2 = n & ~index_t(1);

    for (; i < n2; i += 2) {
        real_t dot0 = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                      x, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t dot1 = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                      x, mdp.row_ptr[i + 1], mdp.row_ptr[i + 2]);

        float64x2_t v_rewards = vld1q_f64(&mdp.rewards[i]);
        float64x2_t v_dots = vcombine_f64(vdup_n_f64(dot0), vdup_n_f64(dot1));
        float64x2_t v_fi = vfmaq_f64(v_rewards, vdupq_n_f64(beta), v_dots);

        float64x2_t v_x = vld1q_f64(x + i);
        float64x2_t v_diff = vabsq_f64(vsubq_f64(v_fi, v_x));
        vmax = vmaxq_f64(vmax, v_diff);
    }

    real_t max_res = vmaxvq_f64(vmax);

    // Scalar tail
    if (i < n) {
        real_t dot = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                     x, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t fi = mdp.rewards[i] + beta * dot;
        real_t res = std::abs(fi - x[i]);
        if (res > max_res) max_res = res;
    }

    return max_res;

#else  // scalar fallback

    real_t max_res = 0.0;
    for (index_t i = 0; i < n; ++i) {
        real_t dot = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                     x, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t fi = mdp.rewards[i] + beta * dot;
        real_t res = std::abs(fi - x[i]);
        if (res > max_res) max_res = res;
    }
    return max_res;

#endif
}


// ── Kernel 4: Gauss-Seidel in-place sweep ──────────────────────────────
//
// x[i] = (1 - alpha) * x[i] + alpha * (rewards[i] + beta * dot_i)
// In-place: each row reads the latest x (including updates from earlier rows).
// Cannot vectorize across rows due to data dependency, but inner product is NEON.

inline void neon_gauss_seidel_sweep(const MDP& mdp,
                                    real_t* __restrict__ x,
                                    real_t alpha)
{
    const index_t n = mdp.n;
    const real_t beta = mdp.beta;
    const real_t one_minus_alpha = 1.0 - alpha;

    for (index_t i = 0; i < n; ++i) {
        real_t dot = neon_sparse_dot(mdp.probs.data(), mdp.col_idx.data(),
                                     x, mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        real_t fi = mdp.rewards[i] + beta * dot;
        x[i] = one_minus_alpha * x[i] + alpha * fi;
    }
}

}  // namespace neon
}  // namespace helios
