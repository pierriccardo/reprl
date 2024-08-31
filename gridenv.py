import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns
import numpy as np

from itertools import product
from matplotlib.colors import LinearSegmentedColormap


class GridEnv(gym.Env):
    def __init__(self, filename="grids/grid.txt"):
        # parse gridfile and save
        # grid, size, init and goal
        self._parse_grid_file(filename)

        # conversion dicts state and coords
        self.coord2state = {}
        idx = 0
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.grid[r, c]:
                    self.coord2state[(r, c)] = idx
                    idx += 1
        self.state2coord = {idx: state for state, idx in self.coord2state.items()}

        # states setup
        self.n_states = np.sum(self.grid)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.init = self.coord2state[self.init]
        self.goal = self.coord2state[self.goal]
        self.state = self.init

        # actions setup
        self.n_actions = 4
        self.action_map = {
            0: [1, 0],  # down
            1: [0, 1],  # right
            2: [-1, 0],  # up
            3: [0, -1],  # left
        }
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # setup matrices, state graph, laplacian
        self.T = self._create_transition_matrix()
        self.R = self._create_reward_matrix()

        # state graph
        self.G = self._create_state_graph()
        self.A = nx.adjacency_matrix(self.G, nodelist=[*range(self.n_states)]).toarray()
        self.D = np.diag([sum(row) for row in self.A])  # degree matrix
        self.L = self.D - self.A  # graph laplacian
        self.L_norm = self._create_normalized_laplacian(self.A, self.L)

        # unweighted undirected state action graph
        self.G_u_sa = self._create_unweighted_state_action_graph()
        self.A_u_sa = nx.adjacency_matrix(
            self.G_u_sa,
            nodelist=[
                (s, a)
                for (s, a) in product(range(self.n_states), range(self.n_actions))
            ],
        )
        self.D_u_sa = np.diag([sum(row) for row in self.A_u_sa])
        self.L_u_sa = self.D_u_sa - self.A_u_sa
        self.L_u_sa_norm = self._create_normalized_laplacian(self.A_u_sa, self.L_u_sa)

        # weighted directed state action graph
        self.G_w_sa = None
        self.A_w_sa = None

    def _parse_grid_file(self, filename):
        self.grid = []
        self.init = None
        self.goal = None

        with open(filename, "r") as f:
            lines = f.read()
            lines = lines.strip().split("\n")

        x_size, y_size = len(lines), len(lines[0])
        self.size = (x_size, y_size)
        self.grid = np.zeros(self.size, dtype=int)
        for x, line in enumerate(lines):
            for y, char in enumerate(line):
                if char == "0":
                    continue
                elif char in "1SG":
                    self.grid[x, y] = 1
                    if char == "S":
                        self.init = (x, y)
                    elif char == "G":
                        self.goal = (x, y)
                else:
                    raise ValueError(
                        f"Invalid character '{char}' at position ({y}, {x})"
                    )

        if self.init is None or self.goal is None:
            raise ValueError("Invalid file, specify init (S) and goal (G) state")
        return self.grid, (x_size, y_size), self.init, self.goal

    def _create_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.grid[x, y] == 0:  # is a wall
                    continue
                s = self.coord2state[(x, y)]  # current state
                for a, (dx, dy) in self.action_map.items():
                    next_x, next_y = x + dx, y + dy
                    if self.grid[next_x, next_y] != 1:
                        # next state is a wall, then stay in the
                        # current state
                        next_x, next_y = x, y
                    P[s, a, self.coord2state[(next_x, next_y)]] = 1

        # set goal as an absorbing state
        P[self.goal, :, :] = 0  # if in goal each a leads to goal
        P[self.goal, :, self.goal] = 1  # stay in goal for each a
        return P

    def _create_reward_matrix(self):
        R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # reward of 1 is collected when moving to
                # the goal state, if already in the goal state
                # there is no reward
                R[s, a] = 1 if self.T[s, a, self.goal] and s != self.goal else 0
        return R

    def _create_normalized_laplacian(self, A, L):
        inv_sqrt = np.diag([1 / np.sqrt(sum(row)) for row in A])
        return inv_sqrt @ L @ inv_sqrt

    def _create_state_graph(self):
        G = nx.Graph()
        nodes = [*range(self.n_states)]
        edges = []
        for i, j in product(nodes, nodes):
            # if at least an action leads to j from i
            if i != j and sum(self.T[i, :, j]):
                edges.append((i, j))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def _create_unweighted_state_action_graph(self):
        G = nx.Graph()
        nodes = list(product([*range(self.n_states)], [*range(self.n_actions)]))
        edges = []
        for i, j in product(nodes, nodes):
            source_s, source_a = i
            target_s, target_a = j
            # taking action a in s leads to target s
            if i != j and self.T[source_s, source_a, target_s] > 0:
                # and sum(self.T[target_s, target_a, :]) > 0 #(??)
                edges.append((i, j))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def _create_weighted_state_action_graph(self, policy):
        G = nx.DiGraph()
        G.add_nodes_from(self.G_u_sa.nodes)
        for (source_s, source_a), (target_s, target_a) in self.G_u_sa.edges:
            weight = 0
            if policy[source_s, source_a] and policy[target_s, target_a] > 0:
                weight = 1 / policy[source_s, source_a] * 1 / policy[target_s, target_a]
            G.add_edge((source_s, source_a), (target_s, target_a), weight=weight)
        return G

    # ------------------------------
    # Print helpers
    # ------------------------------

    def print_transition_matrix(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for ns in range(self.n_states):
                    print(f"s: {s} a: {a} ns: {ns} P: {self.T[s, a, ns]}")

    def show_reward_matrix(self):
        fig, ax = plt.subplots()
        sns.heatmap(
            self.R,
            cmap="Blues",
            ax=ax,
            cbar=True,
            xticklabels=[*range(self.n_actions)],
            yticklabels=[*range(self.n_states)],
        )
        ax.set_title("Reward")
        ax.set_xlabel("Actions")
        ax.set_ylabel("States")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def show_state_graph(self):
        plt.figure()
        nx.draw(self.G, with_labels=True, node_color="lightblue")
        plt.show()

    def show_state_action_graph(self):
        plt.figure()
        plt.legend(
            handles=[
                mpatches.Patch(color="grey", label="0 - down"),
                mpatches.Patch(color="grey", label="1 - right"),
                mpatches.Patch(color="grey", label="2 - up"),
                mpatches.Patch(color="grey", label="3 - left"),
            ]
        )
        pos = nx.spring_layout(self.G_u_sa)  # unweighted state action graph
        nx.draw(
            self.G_u_sa,  # unweighted state action graph
            pos,
            with_labels=True,
            node_color="lightblue",
            width=0.5,
            node_size=20,
            font_size=10,
            alpha=0.5,
        )
        plt.show()

    def show_grid(self, policy=None):
        colors = {
            "arrows": "#ffb703",
            "walls": "#219ebc",
            "init": "g",
            "goal": "r",
        }
        # define resolution in function of the grid size
        scale = 50  # scale grid size
        grid_width, grid_height = (
            self.size[1] * scale,
            self.size[0] * scale,
        )  # Total grid dimensions in pixels
        fig, ax = plt.subplots(figsize=(grid_width / 100, grid_height / 100))

        ax.imshow(
            1 - self.grid,
            cmap=LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", colors["walls"]]
            ),
        )

        # calculate cell dimensions
        cell_width = grid_width / self.size[0]
        cell_height = grid_height / self.size[1]
        k = 0.1
        fontsize = min(cell_width, cell_height) * k

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.grid[i][j]:
                    s = self.coord2state[(i, j)]
                    if s == self.init:
                        ax.add_patch(
                            mpatches.Rectangle(
                                (j - 0.5, i - 0.51),
                                1.01,
                                1.01,
                                facecolor=colors["init"],
                                alpha=0.5,
                            )
                        )
                    if s == self.goal:
                        ax.add_patch(
                            mpatches.Rectangle(
                                (j - 0.5, i - 0.51),
                                1.01,
                                1.01,
                                facecolor=colors["goal"],
                                alpha=0.5,
                            )
                        )

                    ax.text(
                        j + 0.35,
                        i - 0.35,
                        self.coord2state[(i, j)],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=fontsize,
                    )
                    if policy is not None:
                        action_to_arrow = {
                            # position,
                            0: (0, 0.3),  # Down arrow
                            1: (0.3, 0),  # Right arrow
                            2: (0, -0.3),  # Up arrow
                            3: (-0.3, 0),  # Left arrow
                        }
                        s = self.coord2state[(i, j)]
                        for a in range(self.n_actions):
                            if policy[s, a]:
                                dx, dy = action_to_arrow[a]
                                ax.arrow(
                                    j,
                                    i,  # Starting point (x, y)
                                    dx,
                                    dy,  # Direction (dx, dy)
                                    head_width=0.2,
                                    head_length=0.2,
                                    fc=colors["arrows"],
                                    ec=colors["arrows"],
                                    lw=0.4,
                                )
                                ax.text(
                                    j + dx,
                                    i + dy,
                                    f"{policy[s, a]:.2f}",  # Format the number to 2 decimal places
                                    ha="center",
                                    va="center",
                                    color="black",
                                    fontsize=fontsize / 1.5,
                                )
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

    # ------------------------------
    # Env related functions
    # ------------------------------
    def step(self, action):
        probs = self.T[self.state][action]

        next_state = np.random.choice(self.n_states, p=probs)
        reward = self.R[self.state, action]
        terminated = next_state == self.goal
        truncated = False
        info = {}
        self.state = next_state

        return next_state, reward, terminated, truncated, info

    def reset(self):
        self.state = self.init
        info = {}
        return self.state, info

    def render(self):
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if not self.grid[r][c]:
                    print("#", end="")
                else:
                    print(" ", end="")
            print()
        print()


if __name__ == "__main__":
    env = GridEnv(filename="grids/rectangular.txt")
    env.print_transition_matrix()
    # env.show_grid()
    # env.show_reward_matrix()
    # env.show_state_action_graph()
