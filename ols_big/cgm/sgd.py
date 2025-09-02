# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent for linear regression"""
import os
from typing import Optional

import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.optimize import fsolve

np.random.seed(42)

plt.style.use('../../default.mplstyle')

# Synthetic data
N = 1000
X = np.hstack([np.ones((N, 1)), np.random.rand(N, 1)])
Y = 2 * X[:, 0] -2 * X[:, 1] + np.random.normal(size=N)
sol = np.linalg.inv(X.T @ X) @ X.T @ Y


def plot_3d(X: np.ndarray, Y: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> Axes3D:
    # For every possible \beta_1 and \beta_2, calculate the residuals
    b1, b2 = np.meshgrid(np.linspace(x0, x1, 100), np.linspace(y0, y1, 100))
    b = np.stack([b1.ravel(), b2.ravel()], axis=1)
    ssr = np.sum((Y[:, None] - X @ b.T) ** 2, axis=0).reshape(100, 100)
    best_ssr = np.sum((Y - X @ [sol[0], sol[1]]) ** 2)

    # 3D surface plot of SSR
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter([sol[0]], [sol[1]], [best_ssr], color='#800000')
    ax.plot([sol[0], sol[0]], [y0, y1], [best_ssr, best_ssr], color='#800000', lw=1, ls='--')
    ax.plot([x0, x1], [sol[1], sol[1]], [best_ssr, best_ssr], color='#800000', lw=1, ls='--')
    ax.plot_surface(b1, b2, ssr,
                    edgecolor='royalblue', lw=0.5, rstride=5, cstride=5, alpha=0.2)
    ax.set(xlabel=r'$\beta_0$', ylabel=r'$\beta_1$',
           xlim=(x0, x1), ylim=(y0, y1), zlim=(best_ssr, np.max(ssr)))
    ax.set_zlabel(zlabel='SSR', labelpad=10)  # Have to add gaps. Otherwise overlap with tick label
    plt.tight_layout()
    return ax


def plot_contour(X: np.ndarray, Y: np.ndarray):
    b1, b2 = np.meshgrid(np.linspace(1.5, 2.5, 100), np.linspace(-2.5, -1.5, 100))
    b = np.stack([b1.ravel(), b2.ravel()], axis=1)
    ssr = np.sum((Y[:, None] - X @ b.T) ** 2, axis=0).reshape(100, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    ct = ax.contour(b1, b2, ssr, levels=[980, 990, 1020, 1100, 1200, 1300, 1500],
                    cmap='coolwarm', linewidths=1, alpha=0.9)
    ax.clabel(ct, inline=True)
    ax.axhline(sol[1], color='#800000', lw=1, ls='--')
    ax.axvline(sol[0], color='#800000', lw=1, ls='--')
    ax.scatter([sol[0]], [sol[1]], color='#800000', s=10)
    ax.set_aspect('equal')
    plt.xlabel(r'$\beta_0$')
    plt.ylabel(r'$\beta_1$')
    plt.tight_layout()
    plt.savefig('figures/ssr_contour.svg', bbox_inches='tight')
    return


def jac(X: np.ndarray, Y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the gradient of the loss function for a given beta hat"""
    return -2 * X.T @ (Y - X @ b)


def gd(
        X: np.ndarray, Y: np.ndarray,
        n_iter: int = 49, lr: float = 0.0005, init_b: tuple[float] = (0, 0)
) -> tuple[list[np.ndarray], list[float]]:
    """Gradient descent for linear regression

    :return: Tuple of beta estimates and loss values in each step
    """
    b = np.array(init_b, dtype=float)
    bs = np.zeros((n_iter + 1, 2))
    losses = [None for i in range(n_iter + 1)]
    bs[0] = b
    losses[0] = np.sum((Y - X @ b) ** 2)

    for i in range(n_iter):
        b -= lr * jac(X, Y, b)
        bs[i + 1] = b
        losses[i + 1] = np.sum((Y - X @ b) ** 2)

    return bs, losses


def plot_gd(
        X: np.ndarray,
        Y: np.ndarray,
        bs: list[np.ndarray],
        losses: list[float],
        filename: str | os.PathLike,
        extra_bs: Optional[list[np.ndarray]] = None,
):
    n_iter = len(losses) - 1

    # Plot the trajectory of beta over contours of the loss function
    fig, ax = plt.subplots()
    b1, b2 = np.meshgrid(np.linspace(-0.5, 2.5, 100), np.linspace(-2.5, 0.5, 100))
    b = np.stack([b1.ravel(), b2.ravel()], axis=1)
    ssr = np.sum((Y[:, None] - X @ b.T) ** 2, axis=0).reshape(100, 100)
    ax.contour(b1, b2, ssr, levels=[975, 980, 990, 1020, 1100, 1200, 1500, 2000, 3000, 4000, 6000],
               cmap='coolwarm', linewidths=1, alpha=0.9)
    for i in range(n_iter):
        ax.annotate('', bs[i + 1], bs[i],
                    arrowprops=dict(arrowstyle='->', lw=1, color='#000080', mutation_scale=15))
    ax.scatter(*bs.T, color='#000080', s=5)

    # Plot extra beta estimates if provided
    if extra_bs is not None:
        extra_bs = np.array(extra_bs)
        for i in range(len(extra_bs) - 1):
            ax.annotate('', extra_bs[i + 1], extra_bs[i],
                        arrowprops=dict(arrowstyle='->', lw=1, color='orange', mutation_scale=15))
    ax.plot(*extra_bs.T, color='orange', lw=1.5)
    ax.axhline(sol[1], color='#800000', lw=1, ls='--')
    ax.axvline(sol[0], color='#800000', lw=1, ls='--')
    ax.scatter([sol[0]], [sol[1]], color='#800000', s=10)
    ax.set_aspect('equal')
    plt.xlabel(r'$\beta_0$')
    plt.ylabel(r'$\beta_1$')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}_contour.svg', bbox_inches='tight')

    # Helper class to plot 3D arrows
    class Arrow3D(FancyArrowPatch):

        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)

    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)

    setattr(Axes3D, 'arrow3D', _arrow3D)

    # Plot the trajectory of beta over the loss function in 3D
    ax = plot_3d(X, Y, -0.5, 2.5, -2.5, 0.5)
    for i in range(n_iter):
        ax.arrow3D(bs[i][0], bs[i][1], losses[i], bs[i + 1][0] - bs[i][0], bs[i + 1][1] - bs[i][1],
                   losses[i + 1] - losses[i], mutation_scale=8, arrowstyle='->', color='#000080')
    ax.plot(*bs.T, losses, color='#000080', lw=1.5)

    # Plot extra beta estimates if provided
    if extra_bs is not None:
        for i in range(len(extra_bs) - 1):
            ax.arrow3D(extra_bs[i][0], extra_bs[i][1], np.sum((Y - X @ extra_bs[i]) ** 2),
                       extra_bs[i + 1][0] - extra_bs[i][0],
                       extra_bs[i + 1][1] - extra_bs[i][1],
                       np.sum((Y - X @ extra_bs[i + 1]) ** 2) - np.sum((Y - X @ extra_bs[i]) ** 2),
                       mutation_scale=8,
                       arrowstyle='->',
                       color='orange')
    ax.set_zticks([1000, 3000, 5000, 7000, 9000])
    plt.tight_layout()
    plt.savefig(f'figures/{filename}_3d.svg', bbox_inches=Bbox([[2.4, 0.15], [7.92, 4.85]]))
    return


def sgd(
        X: np.ndarray, Y: np.ndarray,
        batch_size: int = 1, n_iter: int = 49, lr: float = 0.0005, init_b: tuple[float] = (0, 0)
) -> tuple[list[np.ndarray], list[float]]:
    """Batch stochastic gradient descent for linear regression

    :return: Tuple of beta estimates and loss values in each step
    """
    b = np.array(init_b, dtype=float)
    bs = np.zeros((n_iter + 1, 2))
    losses = [None for i in range(n_iter + 1)]
    bs[0] = b
    losses[0] = np.sum((Y - X @ b) ** 2)  # Just for reference. Real SGD not going to do this
    indices = list(range(N))
    rng = np.random.default_rng(seed=42)

    for i in range(n_iter):
        rng.shuffle(indices)
        for j in range(0, N, batch_size):
            X_batch = X[indices[j:j + batch_size]]
            Y_batch = Y[indices[j:j + batch_size]]
            b -= lr * jac(X_batch, Y_batch, b)
        bs[i + 1] = b
        losses[i + 1] = np.sum((Y - X @ b) ** 2)
    return bs, losses


def sd(
        X: np.ndarray, Y: np.ndarray,
        n_iter: int = 49, init_b: tuple[float] = (0, 0)
) -> tuple[list[np.ndarray], list[float]]:
    """Steepest descent for linear regression

    :return: Tuple of beta estimates and loss values in each step
    """
    b = np.array(init_b, dtype=float)
    bs = np.zeros((n_iter + 1, 2))
    losses = [None for i in range(n_iter + 1)]
    bs[0] = b
    losses[0] = np.sum((Y - X @ b) ** 2)

    for i in range(n_iter):
        grad = jac(X, Y, b)
        eta = -grad.T @ grad / (2 * grad.T @ X.T @ X @ grad)
        b += eta * grad
        bs[i + 1] = b
        losses[i + 1] = np.sum((Y - X @ b) ** 2)

    return bs, losses


def main():
    os.makedirs('figures', exist_ok=True)
    plot_contour(X, Y)
    plot_3d(X, Y, 1.5, 2.5, -2.5, -1.5)
    plt.savefig('figures/ssr_3d.svg', bbox_inches=Bbox([[2.55, 0.2], [7.92, 4.85]]))

    # GD
    bs, losses = gd(X, Y)
    plot_gd(X, Y, bs, losses, 'gd_iter')

    # SGD
    bs, losses = sgd(X, Y)
    plot_gd(X, Y, bs, losses, 'sgd_iter')

    # Steepest descent
    bs, losses = sd(X, Y)
    plot_gd(X, Y, bs, losses, 'sd_iter')
    return


if __name__ == '__main__':
    main()
