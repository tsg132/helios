# helios/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
# Emit compile_commands.json for clangd/VSCode squiggles to match the real build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(helios
  VERSION 0.1.0
  LANGUAGES CXX
)

# ---------- Options ----------
option(HELIOS_BUILD_TESTS   "Build tests" ON)
option(HELIOS_BUILD_BENCH   "Build benchmarks" ON)
option(HELIOS_ENABLE_LTO    "Enable link-time optimization" ON)
option(HELIOS_ENABLE_ASAN   "Enable AddressSanitizer (Debug only recommended)" OFF)
option(HELIOS_ENABLE_UBSAN  "Enable UndefinedBehaviorSanitizer (Debug only recommended)" OFF)
option(HELIOS_WERROR        "Treat warnings as errors" ON)
option(HELIOS_WITH_OPENMP   "Build OpenMP baseline (requires libomp on macOS)" ON)
option(HELIOS_WITH_EIGEN    "Build Eigen baseline (requires Eigen3)" ON)

# ---------- Global settings ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put outputs in consistent folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------- Include helper CMake modules ----------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Warnings)
include(Sanitizers)
include(LTO)# Emit compile_commands.json for clangd/VSCode squiggles to match the real build.

# ---------- Library target ----------
add_library(helios
  src/metrics.cc
  src/io.cc
  src/verify.cc
  src/runtime.cc
  src/mdp_generators.cc
  src/schedulers/static_blocks.cc
  src/schedulers/shuffled_blocks.cc
  src/schedulers/residual_buckets.cc
  src/schedulers/topk_gs.cc
  src/schedulers/ca_topk_gs.cc
  src/ops/policy_eval_op.cc
  src/ops/policy_eval_op_neon.cc
  src/ops/linear_op.cc
  src/planners.cc
  src/cost_model.cc
  src/autotune.cc
)

target_include_directories(helios
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(helios PUBLIC cxx_std_20)

# helios_set_warnings(helios ${HELIOS_WERROR})
helios_enable_sanitizers(helios ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
helios_enable_lto(helios ${HELIOS_ENABLE_LTO})

# Threads (portable)
find_package(Threads REQUIRED)
target_link_libraries(helios PUBLIC Threads::Threads)

# ---------- Optional: OpenMP (for baselines) ----------
if(HELIOS_WITH_OPENMP)
  if(APPLE)
    # Apple Clang ships without OpenMP; find libomp from Homebrew.
    execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(HOMEBREW_LIBOMP)
      set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_C_LIB_NAMES   "omp")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP}/lib/libomp.dylib")
      include_directories("${HOMEBREW_LIBOMP}/include")
    endif()
  endif()
  find_package(OpenMP)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found — helios_baselines will skip OpenMP_Jacobi")
    set(HELIOS_WITH_OPENMP OFF)
  endif()
endif()

# ---------- Optional: Eigen3 (for baselines) ----------
if(HELIOS_WITH_EIGEN)
  # On macOS with Homebrew, prepend the Eigen prefix so CMake finds the config.
  if(APPLE)
    execute_process(
      COMMAND brew --prefix eigen
      OUTPUT_VARIABLE HOMEBREW_EIGEN_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(HOMEBREW_EIGEN_PREFIX)
      list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_EIGEN_PREFIX}")
    endif()
  endif()
  find_package(Eigen3 CONFIG QUIET)
  if(NOT Eigen3_FOUND)
    find_package(Eigen3 MODULE QUIET)
  endif()
  if(NOT Eigen3_FOUND)
    message(WARNING "Eigen3 not found — helios_baselines will skip Eigen_Jacobi")
    set(HELIOS_WITH_EIGEN OFF)
  endif()
endif()

# ---------- Bench executable ----------
if(HELIOS_BUILD_BENCH)
  add_executable(helios_bench
    bench/run_bench.cc
    bench/gen/gen_grid.cc
    bench/gen/gen_metastable.cc
    bench/gen/gen_random_graph.cc
  )
  target_link_libraries(helios_bench PRIVATE helios)
  # helios_set_warnings(helios_bench ${HELIOS_WERROR})
  helios_enable_sanitizers(helios_bench ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  helios_enable_lto(helios_bench ${HELIOS_ENABLE_LTO})

  add_executable(helios_bench_simd
    bench/bench_simd.cc
  )
  target_link_libraries(helios_bench_simd PRIVATE helios)
  helios_enable_sanitizers(helios_bench_simd ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  helios_enable_lto(helios_bench_simd ${HELIOS_ENABLE_LTO})

  # ── Baseline comparison executable ──────────────────────────────────────────
  add_executable(helios_baselines bench/baselines.cc)
  target_link_libraries(helios_baselines PRIVATE helios)
  helios_enable_sanitizers(helios_baselines ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  helios_enable_lto(helios_baselines ${HELIOS_ENABLE_LTO})

  if(HELIOS_WITH_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(helios_baselines PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(helios_baselines PRIVATE HELIOS_HAS_OPENMP=1)
    message(STATUS "helios_baselines: OpenMP enabled")
  endif()

  if(HELIOS_WITH_EIGEN AND Eigen3_FOUND)
    target_link_libraries(helios_baselines PRIVATE Eigen3::Eigen)
    target_compile_definitions(helios_baselines PRIVATE HELIOS_HAS_EIGEN=1)
    message(STATUS "helios_baselines: Eigen3 enabled")
  endif()
endif()

# ---------- Tests ----------
if(HELIOS_BUILD_TESTS)
  enable_testing()
  add_executable(helios_tests
    tests/test_runtime_smoke.cc
    tests/test_operator_contract.cc
    tests/test_schedulers.cc
    tests/test_complex_mdps.cc
    tests/test_phase3.cc
    tests/test_stress.cc
  )
  target_link_libraries(helios_tests PRIVATE helios)
  # helios_set_warnings(helios_tests ${HELIOS_WERROR})
  helios_enable_sanitizers(helios_tests ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  # Usually keep LTO off for tests/debug, but fine either way:
  # helios_enable_lto(helios_tests ${HELIOS_ENABLE_LTO})
  add_test(NAME helios_tests COMMAND helios_tests)

  add_executable(helios_simd_tests
    tests/test_simd.cc
  )
  target_link_libraries(helios_simd_tests PRIVATE helios)
  helios_enable_sanitizers(helios_simd_tests ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  add_test(NAME helios_simd_tests COMMAND helios_simd_tests)
endif()