import numpy as np


def uniform_policy(n_states, n_actions):
    policy = np.ones((n_states, n_actions)) * (1 / n_actions)
    return policy


def random_policy(n_states, n_actions):
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            # randomly fill vector of length
            # n_actions that sum to 1
            probs = np.random.rand(n_actions)
            probs /= probs.sum()
            policy[s] = probs
    return policy


def random_deterministic_policy(n_states, n_actions):
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        a = np.random.randint(n_actions)
        policy[s, a] = 1
    return policy
