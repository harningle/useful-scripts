# -*- coding: utf-8 -*-
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

plt.style.use('../default.mplstyle')

np.random.seed(42)
N = 1000


def generate_data():
    # Generate data in the latent PCA space (Z)
    z1 = np.random.normal(0, 5, N)    # First component (big, std. dev. = 5)
    z2 = np.random.normal(0, 2, N)    # Second component (smaller)
    z3 = np.random.normal(0, 0.7, N)  # Third very small component
    Z = np.column_stack((z1, z2, z3))

    # Create our desired eigenvectors
    w1 = np.array([1, 1, 0])      # 45-degree line in x-y plane, normalise to unit length
    w1 = w1 / np.linalg.norm(w1)
    w2 = np.array([-1, 1, 0])     # Orthogonal to w1, also in x-y plane
    w2 = w2 / np.linalg.norm(w2)
    w3 = np.array([0, 0, 1])      # Along the z-axis
    W = np.column_stack((w1, w2, w3))

    # Project back into the "original" space
    X = Z @ W.T
    return X


def plot_data(X: np.array) -> matplotlib.axes.Axes:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    ax.scatter(x, y, z, alpha=0.3, s=10, c='#1A237E', linewidths=0)

    # Set 1:1:1 aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    return ax


def plot_multiview(X: np.array):
    plt.close('all')
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1])
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, c='#1A237E', linewidths=0)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal')
    ax2.scatter(X[:, 0], X[:, 2], alpha=0.3, s=10, c='#1A237E', linewidths=0)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_3$')
    ax2.set_aspect('equal')
    ax3.scatter(X[:, 1], X[:, 2], alpha=0.3, s=10, c='#1A237E', linewidths=0)
    ax3.set_xlabel('$x_2$')
    ax3.set_ylabel('$x_3$')
    ax3.set_aspect('equal')
    plt.savefig('figures/multiview.svg')
    return


def main():
    X = generate_data()

    # Rawdata
    os.makedirs('figures', exist_ok=True)
    plot_data(X)
    plt.savefig('figures/rawdata.svg', bbox_inches=Bbox([[2.7, 0], [7.5, 5.1]]))

    # Three views
    plot_multiview(X)

    # Intuition: 45-degree line
    ax = plot_data(X)
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    t_min = max(xlim[0], ylim[0])
    t_max = min(xlim[1], ylim[1])
    t = np.linspace(t_min * 1.05, t_max * 1.05, 10)  # x1 = x2, x3 = 0
    ax.plot(t, t, np.zeros_like(t), color='maroon')
    plt.savefig('figures/intuition.svg', bbox_inches=Bbox([[2.7, 0], [7.5, 5.1]]))
    return


if __name__ == '__main__':
    main()
