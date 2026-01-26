# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Initialized. Singe node, multi-threaded, async fixed point runtime for various operators.

## Getting Started

I am just getting started, I want you to create the following repo structure under the helios repository right now, but don't write anything in them at least for now

:

helios/
  CMakeLists.txt
  cmake/
    Sanitizers.cmake
    LTO.cmake
    Warnings.cmake

  include/helios/
    types.h                 // index_t, real_t, aligned alloc helpers
    operator.h              // Operator interface
    mdp.h                   // MDP / policy-eval data structures (CSR)
    linear.h                // sparse Ax+b operator (CSR)
    runtime.h               // Runtime config + run() API
    scheduler.h             // Scheduler interface
    schedulers/
      static_blocks.h       // S0
      shuffled_blocks.h     // S1
      residual_buckets.h    // S2 (phase 1 stretch)
    metrics.h               // residuals, timers, logging structs
    io.h                    // binary formats for graphs/MDPs
    verify.h                // correctness checks

  src/
    runtime.cc
    metrics.cc
    io.cc
    verify.cc
    schedulers/
      static_blocks.cc
      shuffled_blocks.cc
      residual_buckets.cc
    ops/
      policy_eval_op.cc
      linear_op.cc

  bench/
    CMakeLists.txt
    gen/
      gen_grid.cc
      gen_metastable.cc
      gen_random_graph.cc
    run_bench.cc            // CLI to run configs and dump CSV/JSON

  tools/
    plot.py                 // plots from CSV/JSON (matplotlib)
    summarize.py            // optional: aggregates runs

  docs/
    design_phase1.md        // short design doc (operator+runtime+schedulers)
    benchmarks.md           // definitions + knobs + expected behaviors

  tests/
    CMakeLists.txt
    test_operator_contract.cc   // contraction sanity checks on small instances
    test_runtime_smoke.cc       // runs converge on tiny MDP
    test_schedulers.cc

  README.md
  LICENSE
