#include "helios/mdp_generators.h"

#include <cassert>
#include <random>
#include <set>

namespace helios {

//-----------------------------------------------------------------------------
// Ring MDP
//-----------------------------------------------------------------------------
MDP build_ring_mdp(index_t n, real_t beta, real_t p_self, real_t reward_val) {
    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    // Each state has exactly 2 transitions (nnz = 2*n)
    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(2 * n);
    mdp.probs.resize(2 * n);
    mdp.rewards.resize(n, reward_val);

    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = 2 * i;

        // Transition to self
        mdp.col_idx[2 * i] = i;
        mdp.probs[2 * i] = p_self;

        // Transition to next state (wrapping)
        mdp.col_idx[2 * i + 1] = (i + 1) % n;
        mdp.probs[2 * i + 1] = 1.0 - p_self;
    }
    mdp.row_ptr[n] = 2 * n;

    return mdp;
}

//-----------------------------------------------------------------------------
// Grid MDP
//-----------------------------------------------------------------------------
MDP build_grid_mdp(index_t rows, index_t cols, real_t beta,
                   real_t p_stay, real_t base_reward, real_t reward_gradient) {
    MDP mdp;
    const index_t n = rows * cols;
    mdp.n = n;
    mdp.beta = beta;

    // Helper to convert (r, c) to state index
    auto to_idx = [cols](index_t r, index_t c) -> index_t {
        return r * cols + c;
    };

    // Count transitions: interior cells have 5 (self + 4 neighbors)
    // Edge cells have fewer neighbors
    std::vector<std::vector<std::pair<index_t, real_t>>> transitions(n);

    for (index_t r = 0; r < rows; ++r) {
        for (index_t c = 0; c < cols; ++c) {
            const index_t i = to_idx(r, c);
            std::vector<index_t> neighbors;

            // Collect valid neighbors
            if (r > 0) neighbors.push_back(to_idx(r - 1, c));        // Up
            if (r < rows - 1) neighbors.push_back(to_idx(r + 1, c)); // Down
            if (c > 0) neighbors.push_back(to_idx(r, c - 1));        // Left
            if (c < cols - 1) neighbors.push_back(to_idx(r, c + 1)); // Right

            // Self-loop
            transitions[i].emplace_back(i, p_stay);

            // Distribute remaining probability uniformly to neighbors
            const real_t p_neighbor = (1.0 - p_stay) / static_cast<real_t>(neighbors.size());
            for (index_t j : neighbors) {
                transitions[i].emplace_back(j, p_neighbor);
            }
        }
    }

    // Build CSR from transitions
    index_t nnz = 0;
    for (index_t i = 0; i < n; ++i) {
        nnz += static_cast<index_t>(transitions[i].size());
    }

    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = ptr;
        for (const auto& [j, p] : transitions[i]) {
            mdp.col_idx[ptr] = j;
            mdp.probs[ptr] = p;
            ++ptr;
        }
    }
    mdp.row_ptr[n] = ptr;

    // Set rewards with optional gradient
    for (index_t r = 0; r < rows; ++r) {
        for (index_t c = 0; c < cols; ++c) {
            const index_t i = to_idx(r, c);
            // Gradient: higher reward toward top-right (r=0, c=cols-1)
            const real_t gradient_component = reward_gradient *
                (static_cast<real_t>(cols - 1 - c + r) / static_cast<real_t>(rows + cols - 2));
            mdp.rewards[i] = base_reward + gradient_component;
        }
    }

    return mdp;
}

//-----------------------------------------------------------------------------
// Metastable MDP (Two clusters)
//-----------------------------------------------------------------------------
MDP build_metastable_mdp(index_t n, real_t beta,
                         real_t p_intra, real_t p_bridge,
                         real_t reward_A, real_t reward_B,
                         uint64_t seed) {
    // Ensure n is even
    n = (n / 2) * 2;
    if (n < 2) n = 2;

    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    std::mt19937_64 rng(seed);

    const index_t cluster_size = n / 2;

    // Build transitions for each state
    // States 0..cluster_size-1 are in cluster A
    // States cluster_size..n-1 are in cluster B
    std::vector<std::vector<std::pair<index_t, real_t>>> transitions(n);

    auto is_cluster_A = [cluster_size](index_t i) { return i < cluster_size; };

    for (index_t i = 0; i < n; ++i) {
        const bool in_A = is_cluster_A(i);
        const index_t my_cluster_start = in_A ? 0 : cluster_size;
        const index_t other_cluster_start = in_A ? cluster_size : 0;

        // Intra-cluster transitions: uniform to all states in same cluster
        const real_t p_each_intra = p_intra / static_cast<real_t>(cluster_size);
        for (index_t j = my_cluster_start; j < my_cluster_start + cluster_size; ++j) {
            transitions[i].emplace_back(j, p_each_intra);
        }

        // Bridge transitions: uniform to all states in other cluster
        const real_t p_each_bridge = p_bridge / static_cast<real_t>(cluster_size);
        for (index_t j = other_cluster_start; j < other_cluster_start + cluster_size; ++j) {
            transitions[i].emplace_back(j, p_each_bridge);
        }
    }

    // Build CSR
    const index_t nnz = n * n;  // Dense within this structure
    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = ptr;
        for (const auto& [j, p] : transitions[i]) {
            mdp.col_idx[ptr] = j;
            mdp.probs[ptr] = p;
            ++ptr;
        }
    }
    mdp.row_ptr[n] = ptr;

    // Set rewards per cluster
    for (index_t i = 0; i < n; ++i) {
        mdp.rewards[i] = is_cluster_A(i) ? reward_A : reward_B;
    }

    return mdp;
}

//-----------------------------------------------------------------------------
// Star MDP
//-----------------------------------------------------------------------------
MDP build_star_mdp(index_t n, real_t beta,
                   real_t p_to_hub,
                   real_t hub_reward, real_t leaf_reward) {
    if (n < 2) n = 2;

    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    const index_t num_leaves = n - 1;

    // Count nnz:
    // Hub (state 0): transitions to all leaves = num_leaves
    // Each leaf: transitions to hub (1) + self (1) + possibly other leaves
    // Simplified: leaf -> hub with p_to_hub, leaf -> self with (1 - p_to_hub)

    // Hub: n-1 transitions (to each leaf uniformly)
    // Each leaf: 2 transitions (hub and self)
    const index_t nnz = num_leaves + 2 * num_leaves;

    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;

    // Hub (state 0): uniform transitions to all leaves
    mdp.row_ptr[0] = ptr;
    const real_t p_per_leaf = 1.0 / static_cast<real_t>(num_leaves);
    for (index_t j = 1; j < n; ++j) {
        mdp.col_idx[ptr] = j;
        mdp.probs[ptr] = p_per_leaf;
        ++ptr;
    }

    // Leaves (states 1..n-1)
    for (index_t i = 1; i < n; ++i) {
        mdp.row_ptr[i] = ptr;

        // Transition to hub
        mdp.col_idx[ptr] = 0;
        mdp.probs[ptr] = p_to_hub;
        ++ptr;

        // Transition to self
        mdp.col_idx[ptr] = i;
        mdp.probs[ptr] = 1.0 - p_to_hub;
        ++ptr;
    }
    mdp.row_ptr[n] = ptr;

    // Set rewards
    mdp.rewards[0] = hub_reward;
    for (index_t i = 1; i < n; ++i) {
        mdp.rewards[i] = leaf_reward;
    }

    return mdp;
}

//-----------------------------------------------------------------------------
// Chain MDP
//-----------------------------------------------------------------------------
MDP build_chain_mdp(index_t n, real_t beta,
                    real_t p_left, real_t p_stay, real_t p_right,
                    int reward_fn, bool reflecting) {
    if (n < 2) n = 2;

    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    // Each interior state has 3 transitions (left, stay, right)
    // Boundary states: 2 transitions if reflecting, 1 if absorbing
    std::vector<std::vector<std::pair<index_t, real_t>>> transitions(n);

    for (index_t i = 0; i < n; ++i) {
        if (i == 0) {
            // Left boundary
            if (reflecting) {
                // Can't go left, redistribute to stay
                transitions[i].emplace_back(0, p_left + p_stay);
                transitions[i].emplace_back(1, p_right);
            } else {
                // Absorbing: stay at 0 with prob 1
                transitions[i].emplace_back(0, 1.0);
            }
        } else if (i == n - 1) {
            // Right boundary
            if (reflecting) {
                transitions[i].emplace_back(i - 1, p_left);
                transitions[i].emplace_back(i, p_stay + p_right);
            } else {
                // Absorbing: stay at n-1 with prob 1
                transitions[i].emplace_back(i, 1.0);
            }
        } else {
            // Interior state
            transitions[i].emplace_back(i - 1, p_left);
            transitions[i].emplace_back(i, p_stay);
            transitions[i].emplace_back(i + 1, p_right);
        }
    }

    // Build CSR
    index_t nnz = 0;
    for (index_t i = 0; i < n; ++i) {
        nnz += static_cast<index_t>(transitions[i].size());
    }

    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = ptr;
        for (const auto& [j, p] : transitions[i]) {
            mdp.col_idx[ptr] = j;
            mdp.probs[ptr] = p;
            ++ptr;
        }
    }
    mdp.row_ptr[n] = ptr;

    // Set rewards based on reward_fn
    for (index_t i = 0; i < n; ++i) {
        switch (reward_fn) {
            case 0:  // Uniform
                mdp.rewards[i] = 1.0;
                break;
            case 1:  // Linear
                mdp.rewards[i] = static_cast<real_t>(i + 1) / static_cast<real_t>(n);
                break;
            case 2:  // Quadratic (peaked in middle)
                {
                    const real_t x = static_cast<real_t>(i) / static_cast<real_t>(n - 1) - 0.5;
                    mdp.rewards[i] = 1.0 - 4.0 * x * x;  // Parabola peaked at center
                }
                break;
            default:
                mdp.rewards[i] = 1.0;
        }
    }

    return mdp;
}

//-----------------------------------------------------------------------------
// Random Sparse MDP
//-----------------------------------------------------------------------------
MDP build_random_sparse_mdp(index_t n, index_t nnz_per_row, real_t beta,
                            real_t max_reward, uint64_t seed) {
    if (nnz_per_row < 1) nnz_per_row = 1;
    if (nnz_per_row > n) nnz_per_row = n;

    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<index_t> col_dist(0, n - 1);
    std::uniform_real_distribution<real_t> prob_dist(0.1, 1.0);
    std::uniform_real_distribution<real_t> reward_dist(0.0, max_reward);

    const index_t nnz = n * nnz_per_row;
    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = ptr;

        // Generate random unique column indices
        std::set<index_t> cols;
        while (cols.size() < static_cast<size_t>(nnz_per_row)) {
            cols.insert(col_dist(rng));
        }

        // Generate random probabilities and normalize
        std::vector<real_t> probs;
        real_t sum = 0.0;
        for (size_t k = 0; k < cols.size(); ++k) {
            real_t p = prob_dist(rng);
            probs.push_back(p);
            sum += p;
        }

        // Store transitions with normalized probabilities
        size_t k = 0;
        for (index_t j : cols) {
            mdp.col_idx[ptr] = j;
            mdp.probs[ptr] = probs[k] / sum;
            ++ptr;
            ++k;
        }

        // Random reward
        mdp.rewards[i] = reward_dist(rng);
    }
    mdp.row_ptr[n] = ptr;

    return mdp;
}

//-----------------------------------------------------------------------------
// Multi-cluster MDP
//-----------------------------------------------------------------------------
MDP build_multi_cluster_mdp(index_t n, index_t k, real_t beta,
                            real_t p_intra,
                            const std::vector<real_t>& rewards,
                            uint64_t seed) {
    if (k < 1) k = 1;
    if (k > n) k = n;

    MDP mdp;
    mdp.n = n;
    mdp.beta = beta;

    std::mt19937_64 rng(seed);

    // Divide states into k clusters (as evenly as possible)
    std::vector<index_t> cluster_start(k + 1);
    const index_t base_size = n / k;
    const index_t remainder = n % k;

    cluster_start[0] = 0;
    for (index_t c = 0; c < k; ++c) {
        const index_t size = base_size + (c < remainder ? 1 : 0);
        cluster_start[c + 1] = cluster_start[c] + size;
    }

    auto get_cluster = [&cluster_start, k](index_t i) -> index_t {
        for (index_t c = 0; c < k; ++c) {
            if (i >= cluster_start[c] && i < cluster_start[c + 1]) {
                return c;
            }
        }
        return k - 1;
    };

    // Build transitions
    std::vector<std::vector<std::pair<index_t, real_t>>> transitions(n);
    const real_t p_inter = 1.0 - p_intra;  // Total inter-cluster probability

    for (index_t i = 0; i < n; ++i) {
        const index_t my_cluster = get_cluster(i);
        const index_t my_cluster_size = cluster_start[my_cluster + 1] - cluster_start[my_cluster];
        const index_t other_states = n - my_cluster_size;

        // Intra-cluster: uniform to all states in my cluster
        const real_t p_each_intra = p_intra / static_cast<real_t>(my_cluster_size);
        for (index_t j = cluster_start[my_cluster]; j < cluster_start[my_cluster + 1]; ++j) {
            transitions[i].emplace_back(j, p_each_intra);
        }

        // Inter-cluster: uniform to all states in other clusters
        if (other_states > 0) {
            const real_t p_each_inter = p_inter / static_cast<real_t>(other_states);
            for (index_t c = 0; c < k; ++c) {
                if (c != my_cluster) {
                    for (index_t j = cluster_start[c]; j < cluster_start[c + 1]; ++j) {
                        transitions[i].emplace_back(j, p_each_inter);
                    }
                }
            }
        } else {
            // Only one cluster, all transitions stay intra
        }
    }

    // Build CSR
    index_t nnz = 0;
    for (index_t i = 0; i < n; ++i) {
        nnz += static_cast<index_t>(transitions[i].size());
    }

    mdp.row_ptr.resize(n + 1);
    mdp.col_idx.resize(nnz);
    mdp.probs.resize(nnz);
    mdp.rewards.resize(n);

    index_t ptr = 0;
    for (index_t i = 0; i < n; ++i) {
        mdp.row_ptr[i] = ptr;
        for (const auto& [j, p] : transitions[i]) {
            mdp.col_idx[ptr] = j;
            mdp.probs[ptr] = p;
            ++ptr;
        }
    }
    mdp.row_ptr[n] = ptr;

    // Set rewards
    for (index_t i = 0; i < n; ++i) {
        const index_t my_cluster = get_cluster(i);
        if (!rewards.empty() && my_cluster < static_cast<index_t>(rewards.size())) {
            mdp.rewards[i] = rewards[my_cluster];
        } else {
            // Default: cluster index + 1
            mdp.rewards[i] = static_cast<real_t>(my_cluster + 1);
        }
    }

    return mdp;
}

} // namespace helios
