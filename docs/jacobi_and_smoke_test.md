# Jacobi Iteration and Smoke Test Explanation

## 1. The Fixed-Point Problem

We're solving the fixed-point equation:

```
x = F(x),  where F: R^n -> R^n
```

For policy evaluation in MDPs, this becomes the Bellman equation:

```
V = r + β·P·V
```

where:
- `V ∈ R^n` is the value function (one value per state)
- `r ∈ R^n` is the reward vector
- `P` is the transition probability matrix (row-stochastic: rows sum to 1)
- `β ∈ [0,1)` is the discount factor

The operator `F(x) = r + β·P·x` is a **contraction** with factor β when P is row-stochastic. This guarantees a unique fixed point x* and convergence from any starting point.

---

## 2. Jacobi Iteration

### The Algorithm

Jacobi iteration is a **synchronous** method: all coordinates see the same snapshot of x when computing updates.

```
x^{k+1}_i = (1 - α)·x^k_i + α·F_i(x^k)   for all i = 0, ..., n-1
```

where α ∈ (0,1] is the relaxation parameter. With α=1 (no relaxation):

```
x^{k+1}_i = F_i(x^k)
```

### Implementation (Double Buffering)

Since all updates must see the same x^k, we use two buffers:

```
┌─────────────┐          ┌─────────────┐
│   x_curr    │  ──────> │   x_next    │
│  (read)     │  F_i()   │  (write)    │
└─────────────┘          └─────────────┘
       │                        │
       └──── swap after sweep ──┘
```

Each **sweep** iterates over all n coordinates:
1. Read from `x_curr`
2. Compute `F_i(x_curr)` for each i
3. Write to `x_next[i]`
4. Swap buffers

### Convergence

The iteration converges when the **residual** (how far x is from being a fixed point) is small:

```
‖F(x) - x‖_∞ = max_i |F_i(x) - x_i| ≤ ε
```

### Code Location

The Jacobi implementation is in `src/runtime.cc:run_jacobi_()`:

```cpp
for (index_t i = 0; i < n; ++i) {
    const real_t fi = op.apply_i(i, x_curr);
    x_next[i] = (1.0 - alpha) * x_curr[i] + alpha * fi;
}
swap(x_curr, x_next);
```

---

## 3. The Smoke Test

### Purpose

The smoke test validates the entire pipeline end-to-end:
1. CSR matrix construction
2. PolicyEvalOp computes F_i(x) correctly
3. Residual computation works
4. Jacobi iteration converges to the correct fixed point

### Test Setup: Ring MDP

We construct a simple MDP with n=16 states arranged in a ring:

```
State 0 ──> State 1 ──> State 2 ──> ... ──> State 15 ──┐
   ^                                                    │
   └────────────────────────────────────────────────────┘
```

Transition probabilities from state i:
- P(i → i) = 0.5 (stay)
- P(i → (i+1) mod n) = 0.5 (move forward)

Parameters:
- n = 16 states
- β = 0.9 (discount factor)
- r_i = 1.0 for all states (uniform reward)
- ε = 10^-6 (convergence tolerance)

### CSR Representation

The transition matrix P is stored in Compressed Sparse Row (CSR) format:

```
row_ptr = [0, 2, 4, 6, ..., 32]   (n+1 entries)
col_idx = [0, 1, 1, 2, 2, 3, ...]  (2n entries, pairs of [self, next])
probs   = [0.5, 0.5, 0.5, ...]     (2n entries, all 0.5)
```

For state i:
- Non-zeros are at indices [row_ptr[i], row_ptr[i+1]) = [2i, 2i+2)
- col_idx[2i] = i (self-transition)
- col_idx[2i+1] = (i+1) mod n (forward transition)

### Analytical Solution

By symmetry, all states have the same value V. From the Bellman equation:

```
V = r + β·P·V
V = 1 + β·(0.5·V + 0.5·V)    (since P·V = V when all values equal)
V = 1 + β·V
V·(1 - β) = 1
V = 1/(1 - β) = 1/0.1 = 10.0
```

### What the Test Verifies

1. **Residual computation test**:
   - Sets x = [1, 2, 3, 4] for a 4-state ring
   - Manually computes expected F_i(x) and residuals
   - Verifies `residual_i()` and `residual_inf()` match

2. **Jacobi convergence test**:
   - Starts from x = 0
   - Runs Jacobi until ‖F(x) - x‖_∞ ≤ 10^-6
   - Asserts `result.converged == true`
   - Asserts all x[i] ≈ 10.0 (within tolerance)

### Test Output

```
PASS: Residual computation
PASS: Jacobi ring convergence
  n = 16, beta = 0.90, eps = 1.0e-06
  converged in 0.010 sec, 853712 updates (8.53e+07 updates/sec)
  final_residual_inf = 0.000000000e+00
  solution x[0] = 10.000000 (expected 10.000000, max_err = 7.1e-15)
```

The max error of ~7×10^-15 is machine precision (double has ~15-16 significant digits).

---

## 4. What We've Achieved (Phase 1 Progress)

### Completed

| Component | Status | Description |
|-----------|--------|-------------|
| types.h | ✅ | Core types (index_t, real_t), aligned memory |
| operator.h | ✅ | Abstract Operator interface |
| scheduler.h | ✅ | Abstract Scheduler interface |
| runtime.h | ✅ | RuntimeConfig, RunResult, Mode enum |
| mdp.h | ✅ | MDP struct with CSR storage, validation |
| policy_eval_op | ✅ | Bellman operator F_i(x) = r_i + β·Σ P_ij·x_j |
| Jacobi mode | ✅ | Synchronous iteration with residual monitoring |
| Smoke test | ✅ | Ring MDP convergence test |

### Remaining (Phase 1)

| Component | Status | Description |
|-----------|--------|-------------|
| Gauss-Seidel | ❌ | In-place sequential updates |
| Async mode | ❌ | Multi-threaded parallel updates |
| StaticBlocksScheduler | ❌ | Partition coordinates into thread blocks |
| Benchmarks | ❌ | Performance measurement harness |

---

## 5. Architecture Overview

```
                    ┌─────────────────┐
                    │  RuntimeConfig  │
                    │  - mode         │
                    │  - eps, alpha   │
                    │  - max_seconds  │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│   Operator   │      │   Runtime   │      │  Scheduler  │
│  - n()       │◄────►│  - run()    │◄────►│  - next()   │
│  - apply_i() │      │             │      │  - init()   │
│  - residual_i│      └──────┬──────┘      └─────────────┘
└──────────────┘             │
       ▲                     │
       │              ┌──────┴──────┐
       │              │  RunResult  │
┌──────┴───────┐      │  - converged│
│ PolicyEvalOp │      │  - trace    │
│  (uses MDP)  │      │  - metrics  │
└──────────────┘      └─────────────┘
```

The clean separation allows:
- New operators (LinearOp, etc.) without changing runtime
- New schedulers (residual-based priority) without changing operators
- New execution modes sharing the same infrastructure
