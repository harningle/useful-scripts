# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('../default.mplstyle')

np.random.seed(42)
N = 1000


def main():
    # Two i.i.d. uniform r.v. and normalised by their sum
    arr = np.random.uniform(0, 1, (N, 2))
    arr = arr / np.sum(arr, axis=1)[:, np.newaxis]

    # Scatter plot
    fig, ax = plt.subplots()
    ax.plot(arr[:, 0], arr[:, 1], 'o', color='#000080', alpha=0.02)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$', rotation=0)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('figures/uniform_wrong.svg', bbox_inches='tight')

    # 3D case
    arr = np.random.uniform(0, 1, (N * 10, 3))
    arr = arr / np.sum(arr, axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color='#000080', edgecolors='none', alpha=0.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$x_2$')
    ax.view_init(30, 45, 0)
    plt.tight_layout()
    plt.savefig('figures/uniform_wrong_3d.svg', bbox_inches='tight')
    return


if __name__ == '__main__':
    main()
