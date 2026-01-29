#include "helios/policy_eval_op.h"

#include <atomic>

#include <cmath>

using namespace std;

namespace helios {

real_t PolicyEvalOp::apply_i(index_t i, const real_t* x) const {

    // F(x) = r + beta P x

    const auto& mdp = *mdp_;

    const index_t start = mdp.row_ptr[i];

    const index_t end = mdp.row_ptr[i + 1];

    real_t acc = mdp.rewards[i];

    real_t dot = 0.0;

    for (index_t idx = start; idx < end; ++idx) {

        dot += mdp.probs[idx] * x[mdp.col_idx[idx]];

    }

    acc += mdp.beta * dot;

    return acc;
}

real_t PolicyEvalOp::residual_i(index_t i, const real_t* x) const {

    const real_t Fx_i = apply_i(i, x);

    return (real_t)abs(Fx_i - x[i]);  
}

real_t PolicyEvalOp::apply_i_async(index_t i, const real_t* x) const {

    const auto& mdp = *mdp_;

    const index_t start = mdp.row_ptr[i];

    const index_t end = mdp.row_ptr[i + 1];

    real_t dot = 0.0;

    for (index_t k = start; k < end; ++k) {

        const index_t j = mdp.col_idx[k];

        const real_t xj = atomic_ref<const real_t>(x[j]).load(memory_order_relaxed);

        dot += mdp.probs[k] * xj;

    }

    return mdp.rewards[i] + mdp.beta * dot;

}




} // namespace helios