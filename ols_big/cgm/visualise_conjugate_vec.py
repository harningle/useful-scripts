# -*- coding: utf-8 -*-
"""Illustrate the orthogonality of the conjugate vectors using contours of quadratic form"""
import os

if 'ols_big' not in os.getcwd():
    os.chdir('ols_big')
if 'cgm' not in os.getcwd():
    os.chdir('cgm')

import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.linspace(-50, 50, 50), np.linspace(-50, 50, 50))
points = np.stack([X, Y], axis=-1)
u = [30, 0]

plt.style.use('../../default.mplstyle')
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False


def plot_contour(A: np.ndarray, ax: plt.Axes):
    """
    For a given symmetrical and positive definite matrix A, plot the contour of its quadratic
    form x^T A x, where x is a 2D vector in the plane.
    """
    Z = np.einsum('ijk,kl,ijl->ij', points, A, points)
    ax.contour(X, Y, Z, levels=15, cmap='autumn', linewidths=0.8, alpha=0.5)
    return


def plot_conjugate_vec(A: np.ndarray, ax: plt.Axes):
    """Plot u = (0, 5) and its conjugate vector v"""
    # Solve the conjugate vector v
    """
    u^T A v = 0           --> (u_1 * A_11 + u_2 * A_21) * v_1 + (u_1 * A_12 + u_2 * A_22) * v_2 = 0
    Using symmetry of A:  --> (u_1 * A_11 + u_2 * A_12) * v_1 + (u_1 * A_12 + u_2 * A_11) * v_2 = 0
    Apparently there are infinite solutions to this equation, as if v is a solution, k * v is also
    a solution. So let's pin v_1 = 1, and solve for v_2.
    """
    demon = u[0] * A[0, 1] + u[1] * A[0, 0]
    if np.isclose(demon, 0):
        v = [0, 1]
    else:
        v_2 = -(u[0] * A[0, 0] + u[1] * A[0, 1]) / demon
        v = [1, v_2]
    v = v / np.linalg.norm(v) * np.linalg.norm(u)  # Rescale v to have the same length as u

    # Plot u and v
    ax.quiver(*[0, 0], *u, color='#800000', scale=1, scale_units='xy', angles='xy', width=0.005,
              zorder=2)
    ax.quiver(*[0, 0], *v, color='#000080', scale=1, scale_units='xy', angles='xy', width=0.005,
              zorder=2)
    return


def main():
    os.makedirs('figures', exist_ok=True)
    for i, a in enumerate([-0.3, 0, 0.8]):
        A = np.array([[1, a], [a, 1]])
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_contour(A, ax)
        plot_conjugate_vec(A, ax)
        plt.tight_layout()
        plt.savefig(f'figures/conjugate_vec_{i}.svg', bbox_inches='tight')
    return


if __name__ == '__main__':
    main()
