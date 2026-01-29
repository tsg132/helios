# helios/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
# Emit compile_commands.json for clangd/VSCode squiggles to match the real build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(helios
  VERSION 0.1.0
  LANGUAGES CXX
)

# ---------- Options ----------
option(HELIOS_BUILD_TESTS "Build tests" ON)
option(HELIOS_BUILD_BENCH  "Build benchmarks" ON)
option(HELIOS_ENABLE_LTO   "Enable link-time optimization" ON)
option(HELIOS_ENABLE_ASAN  "Enable AddressSanitizer (Debug only recommended)" OFF)
option(HELIOS_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (Debug only recommended)" OFF)
option(HELIOS_WERROR       "Treat warnings as errors" ON)

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
  src/schedulers/static_blocks.cc
  src/schedulers/shuffled_blocks.cc
  src/ops/policy_eval_op.cc
  src/ops/linear_op.cc
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
endif()

# ---------- Tests ----------
if(HELIOS_BUILD_TESTS)
  enable_testing()
  add_executable(helios_tests
    tests/test_runtime_smoke.cc
    tests/test_operator_contract.cc
    tests/test_schedulers.cc
  )
  target_link_libraries(helios_tests PRIVATE helios)
  # helios_set_warnings(helios_tests ${HELIOS_WERROR})
  helios_enable_sanitizers(helios_tests ${HELIOS_ENABLE_ASAN} ${HELIOS_ENABLE_UBSAN})
  # Usually keep LTO off for tests/debug, but fine either way:
  # helios_enable_lto(helios_tests ${HELIOS_ENABLE_LTO})
  add_test(NAME helios_tests COMMAND helios_tests)
endif()