#pragma once

#include <cstdint>
#include <cstddef> 
#include <cstdlib>
#include <new>
#include <type_traits>

using namespace std;

namespace helios {

    using index_t = uint32_t;

    using real_t = double;

    constexpr size_t kCacheLine = 64; 


    // Branch prediction hints

    #if defined(__clang__) || defined(__GNUC__)
        #define HELIOS_LIKELY(x)   (__builtin_expect(!!(x), 1))
        #define HELIOS_UNLIKELY(x) (__builtin_expect(!!(x), 0))
    #else
        #define HELIOS_LIKELY(x)   (x)
        #define HELIOS_UNLIKELY(x) (x)
    #endif

    
    // Alignment macros

    inline void* aligned_malloc(size_t alignment, size_t size) {
        if (alignment < alignof(void*)) alignment = alignof(void*);

        #if defined(_MSC_VER)
            return _aligned_malloc(size, alignment);
        #else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
            return ptr;
        #endif

    }

    inline void aligned_free(void* ptr) {
        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    template <class T>

    struct AlignedDeleter {
        void operator()(T* ptr) const {
            aligned_free(static_cast<void*>(ptr));
        }
    };

    template <class T>

    inline T* aligned_new_array(size_t n, size_t alignment = kCacheLine) {

        static_assert(is_trivially_destructible_v<T>,
                      "aligned_new_array can only be used with trivially destructible types");

        void* ptr = aligned_malloc(alignment, n * sizeof(T));
        if (!ptr) throw bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }

    constexpr bool is_power_of_two(size_t x) {
        return (x != 0) && ((x & (x - 1)) == 0);
    }

} // namespace helios