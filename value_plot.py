import matplotlib.pyplot as plt
import numpy as np

"""
Input:
    Q_tab : Tabulr Q (numpy matrix |S| by |A|)
    env : an environment object (e.g. env = Maze())
    isMaze : fixed to True
    arrow : True if you want to plot arrows.s
"""


def value_plot(Q_tab, env, isMaze=True, arrow=True):
    direction = {0: (0, -0.4), 1: (0, 0.4), 2: (-0.4, 0), 3: (0.4, 0)}  # (x,y) cooridnate
    V = np.max(Q_tab, axis=1)
    best_action = np.argmax(Q_tab, axis=1)
    if isMaze:
        idx2cell = env.idx2cell
        for i in range(8):
            f, ax = plt.subplots()
            y_mat = np.zeros(env.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = V[8 * j + i]
                if arrow:
                    a = best_action[8 * j + i]
                    ax.arrow(pos[1], pos[0], direction[a][0], direction[a][1],
                             head_width=0.05, head_length=0.1, fc='r', ec='r')
            y_mat[env.goal_pos] = max(V) + 0.1
            ax.imshow(y_mat, cmap='gray')
    else:
        n = int(np.sqrt(len(V)))
        tab = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                if not (r == (n - 1) and c == (n - 1)):
                    tab[r, c] = V[n * c + r]
                    if arrow:
                        d = direction[best_action[n * c + r]]
                        plt.arrow(c, r, d[0], d[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
        tab[env.goal_pos] = max(V[:-1]) + 0.1
        plt.imshow(tab, cmap='gray')
    plt.show()
