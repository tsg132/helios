# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Initialized. Singe node, multi-threaded, async fixed point runtime for various operators.

## Getting Started

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