import numpy as np


def value_iteration(T, R, gamma=0.99, epsilon=0.001):
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = max(
                sum(
                    p * (R[s, a] + gamma * V[s_next])
                    for (s_next, p) in enumerate(T[s, a])
                )
                for a in range(n_actions)
            )
            delta = max(delta, np.abs(v - V[s]))
        if delta < epsilon:
            break

    policy = np.zeros((n_states, n_actions))

    for s in range(n_states):
        best_action = max(
            list(range(n_actions)),
            key=lambda a: sum(
                p * (R[s, a] + gamma * V[s_next]) for (s_next, p) in enumerate(T[s, a])
            ),
        )
        policy[s, best_action] = 1

    return V, policy


def policy_evaluation(policy):
    pass


if __name__ == "__main__":
    from gridenv import GridEnv

    env = GridEnv(filename="grids/minigrid.txt")
    V, opt_policy = value_iteration(env.T, env.R)
    # print(env.T[0, 0, :])
    # print(env.T[0, 0, 0])
    for s in range(env.n_states):
        for a in range(env.n_actions):
            print(f"s {s}, a {a}: pi(s) = {opt_policy[s]}")
    print(V)
    print(opt_policy)
