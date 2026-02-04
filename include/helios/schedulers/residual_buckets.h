#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "helios/scheduler.h"
#include "helios/types.h"


using namespace std;

namespace helios {

/*

Residual bucket priority scheduler.

Runtime periodically calls rebuild(residuals)

next(tid) returns indices, prioritizing higher residual buckets

Bucketing policy: bucket = clamp (floor (log2 (res / base)))

where base is a small positive constant

Withing each bucket, indices are filled in increasing i order.

corss thread interleaving is nondeterministic, but

the bucket tructure and per bucket ordering are deterministic.

*/

class ResidualBucketsScheduler final : public Scheduler {

    public:

    struct Params {

        uint32_t num_buckets = 32;

        real_t base = static_cast<real_t>(1e-12);

        bool fallback_round_robin = true;

    };

    ResidualBucketsScheduler() : params_() {}
    explicit ResidualBucketsScheduler(Params p) : params_(p) {}

    void init(index_t n, size_t num_threads) override;

    index_t next(size_t tid) override;

    void rebuild(const std::vector<real_t>& residuals) override;

    bool supports_rebuild() const noexcept override { return true; }

    string_view name() const noexcept override { return "residual_buckets"; }

    private:

        struct Data {

            index_t n = 0;

            uint32_t B = 0;

            vector<index_t> indices;

            vector<uint32_t> offsets;

            vector<atomic<uint32_t>> cursor;

            atomic<index_t> rr_cursor{0};

            bool all_tiny = false;
        };

        static uint32_t bucket_of_(real_t r, real_t base, uint32_t B);

        Params params_;

        index_t n_ = 0;

        size_t num_threads_ = 0;

        std::shared_ptr<Data> data_{nullptr};  // Protected by atomic_load/store

        vector<uint32_t> thread_bucket_hint_;

};

} // namespace helios