#pragma once

#include <string_view>
#include <cmath>

#include "helios/operator.h"
#include "helios/mdp.h"

using namespace std;

namespace helios {

class PolicyEvalOp final : public Operator {

    public:

        explicit PolicyEvalOp(const MDP* mdp) : mdp_(mdp) {}

        index_t n() const noexcept override {
            return mdp_->n;
        }

        real_t apply_i(index_t i, const real_t* x) const override;

        real_t residual_i(index_t i, const real_t* x) const override {
            const real_t Fx_i = apply_i(i, x);
            return (real_t)abs(Fx_i - x[i]);
        }

        string_view name() const noexcept override {
            return "policy_eval";
        }

        void check_invariants() const override {

        #ifndef NDEBUG

            if (!mdp_) throw runtime_error("PolicyEvalOp has null MDP pointer");

            mdp_->validate(true);

        #endif

        }

    private:

    const MDP* mdp_ = nullptr;

};

}  // namespace helios