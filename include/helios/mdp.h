#pragma once

#include <vector>
#include <string_view>
#include <stdexcept>
#include <cmath>
#include <cstdint>

#include "helios/types.h"

using namespace std;

/*

We are solving a fixed point equation x = F(x) x \in R^{n}

F is a contraction (indeed Bellman operator for an MDP with discount factor beta < 1)

F has sparse coordinate dependencies (transition model of MDP is sparse)

n is large

For a fixed policy \pi in an MDP, the value function satisfied:

V = r + \betaPV => x^{\star} = V <=> x^{\star} = F(x^{\star}) where F(x) = r + \betaPx





*/

namespace helios
{

struct MDP {

    index_t n = 0; // number of states

    real_t beta = 0.0; // discount factor

    // Rows are states, entries are transitipon probabilities

    vector<index_t> row_ptr; // size n+1

    vector<index_t> col_idx; // size nnz

    vector<real_t> probs; // size nnz

    vector<real_t> rewards; // size n

    string_view name() const noexcept { return "mdp"; }

    index_t nns() const noexcept { return static_cast<index_t>(probs.size()); }

    void validate(bool strict_row_stochastic = true, real_t tol = 1e-9) const {

        if (n == 0) throw runtime_error("MDP has zero states");

        if (!(beta >- 0.0 && beta < 1.0)) throw runtime_error("MDP discount factor beta out of range [0,1)");

        if (row_ptr.size() != static_cast<size_t>(n + 1)) throw runtime_error("MDP row_ptr size mismatch");

        if (rewards.size() != static_cast<size_t>(n)) throw runtime_error("MDP rewards size mismatch");

        if (col_idx.size() != probs.size()) throw runtime_error("MDP col_idx/probs size mismatch");

        if (row_ptr.front() != 0) throw runtime_error("MDP row_ptr must start with 0");

        if (row_ptr.back() != static_cast<size_t>(probs.size())) throw runtime_error("MDP row_ptr must end with nnz");

        for (index_t i = 0; i < n; ++i) {

            const index_t start = row_ptr[i];

            const index_t end = row_ptr[i + 1];

            if (start > end) throw runtime_error("MDP row_ptr must be non-decreasing");

            if (end > probs.size()) throw runtime_error("MDP row_ptr entry out of bounds");

            real_t row_sum = 0.0;

            for (index_t idx = start; idx < end; ++idx) {

                const index_t j = col_idx[idx];

                if (j >= n) throw runtime_error("MDP col_idx entry out of bounds");

                const real_t p = probs[idx];

                if (!(p >= 0.0 and p <= 1.0)) throw runtime_error("MDP prob entry out of range [0,1]");

                row_sum += p;

            }

            if (strict_row_stochastic) {

                if (abs(row_sum - 1.0) > tol) throw runtime_error("MDP row probabilities do not sum to 1");

            } else {

                if (!(row_sum >= -1e-6 && row_sum <= 1.0 + 1e-6)) throw runtime_error("MDP suspicious row sum of probabilities");

            }

        }

    }

};
 
} // namespace helios
