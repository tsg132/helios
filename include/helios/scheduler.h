/*

init(n, num_thread): set up internal buffers/per-thread ranges

next(tid): return the next coordinate index for thread tid

notify(tid, i, resiudual): scheduler can ifore if not needed

*/

#pragma once

#include <cstddef>
#include <string_view>

#include "types.h"

using namespace std;

namespace helios {

    class Scheduler {
    public:
        virtual ~Scheduler() = default;

        virtual void init(index_t n, size_t num_threads) = 0;

        virtual index_t next(size_t tid) = 0;

        virtual void notify(size_t tid, index_t i, real_t residual) {}

        virtual string_view name() const noexcept {return "scheduler";}
    };

}  // namespace helios