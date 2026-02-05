#pragma once

#include "helios/mdp.h"
#include <cstdint>

namespace helios {

//=============================================================================
// MDP Generators for Testing and Benchmarking
//=============================================================================
// Each generator creates an MDP with specific structure to test different
// convergence characteristics and scheduler behaviors.

//-----------------------------------------------------------------------------
// Ring MDP: Simple circular structure (baseline)
//-----------------------------------------------------------------------------
// State i transitions to self (p_self) and (i+1) % n (1-p_self)
// All rewards = reward_val
// Analytical solution: V* = reward_val / (1 - beta) for all states
//
// Use case: Simple baseline test, uniform structure
MDP build_ring_mdp(index_t n, real_t beta, real_t p_self = 0.5, real_t reward_val = 1.0);

//-----------------------------------------------------------------------------
// Grid MDP: 2D grid with local transitions
//-----------------------------------------------------------------------------
// States arranged in rows x cols grid (n = rows * cols)
// From state (r,c), can move to 4 neighbors (or stay) with configurable probs
// Rewards vary by position: reward(r,c) = base_reward + gradient based on position
//
// Parameters:
//   rows, cols: Grid dimensions
//   beta: Discount factor
//   p_stay: Probability of staying in place
//   reward_gradient: If > 0, rewards increase toward top-right corner
//
// Use case: Spatial locality, tests cache-friendly schedulers
MDP build_grid_mdp(index_t rows, index_t cols, real_t beta,
                   real_t p_stay = 0.2, real_t base_reward = 1.0,
                   real_t reward_gradient = 0.0);

//-----------------------------------------------------------------------------
// Metastable MDP: Two clusters with rare inter-cluster transitions
//-----------------------------------------------------------------------------
// n/2 states in cluster A, n/2 in cluster B
// Within each cluster: dense random transitions (high p_intra)
// Between clusters: rare bridge transitions (low p_bridge)
//
// This creates "metastable" dynamics where value propagates quickly within
// clusters but slowly between them - a challenging case for iterative solvers.
//
// Parameters:
//   n: Total number of states (will be rounded to even)
//   beta: Discount factor
//   p_intra: Probability of staying within cluster (per transition)
//   p_bridge: Probability of crossing to other cluster
//   reward_A, reward_B: Different rewards per cluster
//   seed: Random seed for reproducibility
//
// Use case: Tests convergence on hard problems with slow mixing
MDP build_metastable_mdp(index_t n, real_t beta,
                         real_t p_intra = 0.95, real_t p_bridge = 0.05,
                         real_t reward_A = 1.0, real_t reward_B = 2.0,
                         uint64_t seed = 42);

//-----------------------------------------------------------------------------
// Star MDP: Hub-and-spoke structure
//-----------------------------------------------------------------------------
// State 0 is the "hub", states 1..n-1 are "leaves"
// Leaves transition to hub with high probability, to random other leaf with low prob
// Hub transitions uniformly to all leaves
//
// This creates highly skewed residual patterns - the hub has many dependencies
// while leaves have few.
//
// Parameters:
//   n: Total number of states (1 hub + n-1 leaves)
//   beta: Discount factor
//   p_to_hub: Probability of leaf transitioning to hub
//   hub_reward: Reward at hub state
//   leaf_reward: Reward at leaf states
//
// Use case: Tests schedulers with skewed workloads
MDP build_star_mdp(index_t n, real_t beta,
                   real_t p_to_hub = 0.8,
                   real_t hub_reward = 1.0, real_t leaf_reward = 0.5);

//-----------------------------------------------------------------------------
// Chain MDP: Linear chain with biased drift
//-----------------------------------------------------------------------------
// States 0, 1, ..., n-1 arranged in a line
// From state i, can go to i-1 (p_left), stay at i (p_stay), or go to i+1 (p_right)
// Boundary conditions: reflecting (stay at boundary) or absorbing
//
// Parameters:
//   n: Number of states
//   beta: Discount factor
//   p_left, p_stay, p_right: Transition probabilities (must sum to 1)
//   reward_fn: 0 = uniform, 1 = linear (reward = i), 2 = quadratic
//   reflecting: If true, boundary states reflect; if false, they absorb
//
// Use case: Tests convergence with strong directional bias, slow mixing
MDP build_chain_mdp(index_t n, real_t beta,
                    real_t p_left = 0.25, real_t p_stay = 0.5, real_t p_right = 0.25,
                    int reward_fn = 0, bool reflecting = true);

//-----------------------------------------------------------------------------
// Random Sparse MDP: Random graph structure
//-----------------------------------------------------------------------------
// Each state has exactly nnz_per_row random outgoing transitions
// Transition probabilities sampled uniformly then normalized
// Rewards sampled from [0, max_reward]
//
// Parameters:
//   n: Number of states
//   nnz_per_row: Number of non-zero transitions per state
//   beta: Discount factor
//   max_reward: Maximum reward value
//   seed: Random seed for reproducibility
//
// Use case: General sparse problems without special structure
MDP build_random_sparse_mdp(index_t n, index_t nnz_per_row, real_t beta,
                            real_t max_reward = 1.0, uint64_t seed = 42);

//-----------------------------------------------------------------------------
// Multi-cluster MDP: Generalization of metastable with k clusters
//-----------------------------------------------------------------------------
// n states divided into k clusters
// Dense intra-cluster transitions, sparse inter-cluster transitions
//
// Parameters:
//   n: Total states
//   k: Number of clusters
//   beta: Discount factor
//   p_intra: Probability of intra-cluster transition
//   rewards: Vector of per-cluster rewards (size k), or empty for uniform
//   seed: Random seed
//
// Use case: More complex metastability patterns
MDP build_multi_cluster_mdp(index_t n, index_t k, real_t beta,
                            real_t p_intra = 0.9,
                            const std::vector<real_t>& rewards = {},
                            uint64_t seed = 42);

} // namespace helios
