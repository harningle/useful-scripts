# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

plt.style.use('../default.mplstyle')

np.random.seed(0)
N = 1000


def scatter_2d(arr: np.array, filename: str):
    fig, ax = plt.subplots()
    ax.plot(arr[:, 0], arr[:, 1], 'o', color='#000080', alpha=0.02)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$', rotation=0)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    return


def scatter_3d(arr: np.array, filename: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color='#000080', edgecolors='none', alpha=0.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$x_2$')
    ax.view_init(30, 45, 0)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=Bbox([[1.85, 0.1], [7.3, 4.88]]))
    return


def main():
    # Two i.i.d. uniform r.v. and normalised by their sum
    arr = np.random.uniform(0, 1, (N, 2))
    arr = arr / np.sum(arr, axis=1)[:, np.newaxis]

    # Scatter plot of the resulting points
    scatter_2d(arr, 'figures/uniform_wrong.svg')


    # Theoretical and empirical c.d.f. of the points in 2D case
    arr = np.random.uniform(0, 1, (N, 2))
    arr = arr / np.sum(arr, axis=1)[:, np.newaxis]
    xs = np.linspace(0, 1, 100)
    ys = np.zeros_like(xs)
    ys[xs < 0.5] = xs[xs < 0.5] / (2 - 2 * xs[xs < 0.5])
    ys[xs >= 0.5] = (3 * xs[xs >= 0.5] - 1) / (2 * xs[xs >= 0.5])
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color='#000080', label='c.d.f.')
    ax.ecdf(arr[:, 0], color='orange', label='Empirical c.d.f.')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel(r'$F_{X_0}(X_0\leq x_0)$')
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig('figures/uniform_wrong_cdf.svg', bbox_inches='tight')

    # 3D case
    arr = np.random.uniform(0, 1, (N * 10, 3))
    arr = arr / np.sum(arr, axis=1)[:, np.newaxis]
    scatter_3d(arr, 'figures/uniform_wrong_3d.svg')

    # Sample from Dirichlet distribution for 2D and 3D cases
    arr = np.random.dirichlet([1, 1], N)
    scatter_2d(arr, 'figures/dirichlet.svg')
    arr = np.random.dirichlet([1, 1, 1], N * 10)
    scatter_3d(arr, 'figures/dirichlet_3d.svg')
    return


if __name__ == '__main__':
    main()
