#pragma once

// NEON-optimized PolicyEvalOp — drop-in replacement for PolicyEvalOp.
// Uses NEON SIMD intrinsics for the sparse dot product inner loop.
// Plugs into the existing Runtime via the Operator virtual interface.

#include <atomic>
#include <cmath>
#include <string_view>

#include "helios/operator.h"
#include "helios/mdp.h"
#include "helios/simd/neon_kernels.h"

using namespace std;

namespace helios {

class PolicyEvalOpNeon final : public Operator {

public:

    explicit PolicyEvalOpNeon(const MDP* mdp) : mdp_(mdp) {}

    index_t n() const noexcept override {
        return mdp_->n;
    }

    real_t apply_i(index_t i, const real_t* x) const override {
        const auto& mdp = *mdp_;
        real_t dot = neon::neon_sparse_dot(
            mdp.probs.data(), mdp.col_idx.data(), x,
            mdp.row_ptr[i], mdp.row_ptr[i + 1]);
        return mdp.rewards[i] + mdp.beta * dot;
    }

    real_t residual_i(index_t i, const real_t* x) const override {
        return (real_t)abs(apply_i(i, x) - x[i]);
    }

    // Async variants: scalar fallback because atomic_ref and NEON don't mix safely.
    // The atomic loads in the gather pattern must go through atomic_ref for correctness
    // under concurrent writes from other threads.

    real_t apply_i_async(index_t i, const real_t* x) const override {
        const auto& mdp = *mdp_;
        const index_t start = mdp.row_ptr[i];
        const index_t end = mdp.row_ptr[i + 1];
        real_t dot = 0.0;
        for (index_t k = start; k < end; ++k) {
            const index_t j = mdp.col_idx[k];
            const real_t xj = atomic_ref<real_t>(
                const_cast<real_t&>(x[j])).load(memory_order_relaxed);
            dot += mdp.probs[k] * xj;
        }
        return mdp.rewards[i] + mdp.beta * dot;
    }

    real_t residual_i_async(index_t i, const real_t* x) const override {
        const real_t fi = apply_i_async(i, x);
        const real_t xi = atomic_ref<real_t>(
            const_cast<real_t&>(x[i])).load(memory_order_relaxed);
        return (real_t)abs(fi - xi);
    }

    string_view name() const noexcept override {
        return "policy_eval_neon";
    }

    void check_invariants() const override {
    #ifndef NDEBUG
        if (!mdp_) throw runtime_error("PolicyEvalOpNeon has null MDP pointer");
        mdp_->validate(true);
    #endif
    }

    const MDP* mdp() const noexcept { return mdp_; }

private:

    const MDP* mdp_ = nullptr;

};

}  // namespace helios
