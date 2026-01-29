#include "helios/policy_eval_op.h"

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
};

} // namespace helios