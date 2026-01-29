# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Helios is a **production-grade deterministic execution engine** for computing fixed points of contractive operators:

```
x = F(x),  F: R^n -> R^n
```

Primary use case: **Policy evaluation in MDPs** via the Bellman equation `V = r + βPV`.

## Build Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build                    # Build all (library + bench + tests)
cmake --build build --target helios    # Build library only
cmake --build build --target helios_tests
./build/bin/helios_tests               # Run tests
```

## Current Progress (Phase 1)

| Step | Component | Status | Notes |
|------|-----------|--------|-------|
| 0 | Setup (CMake, types.h) | ✅ Done | C++20, LTO, sanitizers configured |
| 1 | Core interfaces | ✅ Done | operator.h, scheduler.h, runtime.h |
| 2 | Policy Evaluation | ✅ Done | mdp.h (CSR), policy_eval_op.cc |
| 3a | Jacobi baseline | ✅ Done | Double-buffered synchronous iteration |
| 3b | Gauss-Seidel | ❌ TODO | In-place sequential updates |
| 4 | Async runtime | ❌ TODO | Multi-threaded + StaticBlocksScheduler |
| 5 | Bench runner | ❌ TODO | CLI + CSV/JSON output |
| 6 | Generators | ❌ TODO | grid, metastable, random_graph MDPs |
| 7 | Scheduler upgrades | ❌ TODO | shuffled_blocks, residual_buckets |

**Smoke test**: `tests/test_runtime_smoke.cc` - Ring MDP with n=16 states, verifies Jacobi converges to analytical solution V=10.0.

## Architecture

```
Operator (n, apply_i, residual_i)     Scheduler (init, next)
              │                              │
              └──────────┬───────────────────┘
                         ▼
                   Runtime::run()
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Jacobi    GaussSeidel   Async
           (done)    (stub)        (stub)
                         │
                         ▼
                    RunResult
```

## Key Files

- `include/helios/operator.h` - Abstract operator interface
- `include/helios/scheduler.h` - Abstract scheduler interface
- `include/helios/runtime.h` - RuntimeConfig, RunResult, Mode enum
- `include/helios/mdp.h` - MDP struct with CSR storage
- `include/helios/policy_eval_op.h` - Bellman operator F_i(x) = r_i + β·Σ P_ij·x_j
- `src/runtime.cc` - Main execution loop (Jacobi implemented, GS/Async stubbed)
- `docs/jacobi_and_smoke_test.md` - Detailed explanation of Jacobi iteration

## Next Steps

1. **Implement Gauss-Seidel** in `run_gauss_seidel_()` - single-threaded in-place updates
2. **Implement StaticBlocksScheduler** - partition [0,n) into contiguous blocks per thread
3. **Implement Async mode** in `run_async_()` - multi-threaded workers calling scheduler
4. **Add tests** for GS and Async modes

## Guidelines for Claude Code

- Don't implement features ahead of the checklist order
- Keep the Operator/Scheduler/Runtime separation clean
- All operators must be thread-safe (read-only access to internal state)
- Use CSR format for sparse matrices
- Convergence criterion: ‖F(x) - x‖_∞ ≤ ε

---

## Reference: Phase 1 Checklist (Original Plan)

Structure I am following, this is my documentation, don't try implementing this on your own Claude Code!!!

Here’s a copyable Phase 1 checklist for Helios (C++20). It’s the same plan, just much cleaner and action-oriented.

⸻

Helios Phase 1: Repo skeleton (copy/paste)

helios/
  CMakeLists.txt
  cmake/
    Warnings.cmake
    Sanitizers.cmake
    LTO.cmake

  include/helios/
    types.h
    operator.h
    scheduler.h
    runtime.h
    metrics.h
    mdp.h
    linear.h
    io.h
    verify.h
    schedulers/
      static_blocks.h
      shuffled_blocks.h
      residual_buckets.h

  src/
    runtime.cc
    metrics.cc
    io.cc
    verify.cc
    ops/
      policy_eval_op.cc
      linear_op.cc
    schedulers/
      static_blocks.cc
      shuffled_blocks.cc
      residual_buckets.cc

  bench/
    run_bench.cc
    gen/
      gen_grid.cc
      gen_metastable.cc
      gen_random_graph.cc

  tests/
    test_runtime_smoke.cc
    test_operator_contract.cc
    test_schedulers.cc

  tools/
    plot.py


⸻

Helios Phase 1: Implementation order (copy/paste)

0) Setup (0.5 day)
	•	CMake builds helios library + helios_bench executable
	•	include/helios/types.h (types + aligned alloc)

    Done.

1) Core interfaces (1 day)
	•	operator.h
	•	n(), apply_i(i,x), residual_i(i,x)
	•	scheduler.h
	•	init(n, num_threads), next(tid)
	•	runtime.h
	•	config struct (threads, alpha, eps, log interval)
	•	run() returns metrics + residual curve

    Done. 

2) Flagship operator: Policy Evaluation (1–2 days)
	•	mdp.h CSR storage:
	•	row_ptr[n+1], col_idx[nnz], prob[nnz], reward[n], beta
	•	policy_eval_op.cc:
	•	F_i(x)=reward[i] + beta * sum_j prob * x[j]
	•	residual |F_i(x)-x[i]|
	•	tests/test_runtime_smoke.cc: tiny MDP converges

3) Baselines (2–3 days)
	•	Jacobi synchronous mode
	•	Gauss–Seidel single-thread mode
	•	Residual computation + stop rule: ||F(x)-x||_inf <= eps

4) Helios runtime: Async multi-thread (2–4 days)
	•	Scheduler S0: static contiguous blocks per thread
	•	Async loop:
	•	pick i
	•	compute fi
	•	update x[i] = (1-alpha)*x[i] + alpha*fi
	•	Monitor thread (or periodic scan) logs ||F(x)-x||_inf every 50–200ms

5) Bench runner (2–3 days)
	•	bench/run_bench.cc CLI:
	•	--op=policy_eval|linear
	•	--sched=static|shuffle|residual
	•	--n --nnz_per_row --beta
	•	--threads --alpha --eps
	•	Writes CSV/JSON:
	•	time-to-eps
	•	updates/sec
	•	residual curve vs time

6) Benchmark generators (3–5 days)
	•	gen_grid.cc (local transitions)
	•	gen_metastable.cc (two clusters + rare bridges) most important
	•	optional gen_random_graph.cc

7) Scheduler upgrades (stretch)
	•	S1: shuffled blocks (reshuffle per epoch)
	•	S2: residual buckets (coarse priority)

⸻

“Stop when Phase 1 is done”

Phase 1 is complete when you can run:
	•	policy_eval on grid and metastable
	•	Jacobi vs GS vs Async(S0) (and ideally Async(S1))
	•	and produce residual vs time plots + time-to-eps table.

⸻

If you want, I’ll now generate the next code skeletons in the same copy/paste style:
	•	operator.h, scheduler.h, runtime.h (headers)
	•	static_blocks.{h,cc} (S0)
	•	mdp.h + policy_eval_op.cc