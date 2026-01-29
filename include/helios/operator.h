/*

the operator must provide:

n(): vector length

apply_i(i, x): Compute F_{i}(x) using the current (possibly stale) state x

residual_i(i, x): Cheap local residual abs(F_{i}(x) - x_{i})

name() for logging purposes

check_invariants() for debugging purposes

*/

#pragma once

#include <cstddef>
#include <string_view>

#include "types.h"

using namespace std;

namespace helios {

    class Operator {
    public:
        virtual ~Operator() = default;

        // Dimension n of the state vector

        virtual index_t n() const noexcept = 0;

        // Copute F_{i}(x) using the current (possibly stale) state x

        // Must be thread-safe for concurrent calls (read-only access to internal state)

        virtual real_t apply_i(index_t i, const real_t* x) const = 0;

        virtual real_t residual_i(index_t i, const real_t* x) const {
            return (real_t)abs(apply_i(i, x) - x[i]);
        }

        virtual string_view name() const noexcept {return "operator";}

        virtual void check_invariants() const {}
    }; 

}  // namespace helios