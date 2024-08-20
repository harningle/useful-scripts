# -*- coding: utf-8 -*-
from functools import partial
import os
import time
from typing import Callable

import cvxpy as cp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

plt.style.use('../default.mplstyle')

raw_df = pd.read_stata('https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta')
raw_df['year'] = raw_df['year'].astype(int)


def pre_process():
    pre = raw_df[raw_df['year'] <= 1988][['state', 'year', 'cigsale']]
    pre = pre \
        .set_index(['state', 'year']) \
        .unstack(level=0) \
        .stack(level=0, future_stack=True)
    X_0 = pre['California'].values
    X_1 = pre.drop(columns='California').values
    return X_0, X_1


def scm_scipy(X_0, X_1, jac: Callable = None):
    def loss(w, X_0, X_1):
        resid = X_0 - X_1 @ w
        return resid.T @ resid

    # Constraint 1: all weights sum up to one
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})

    # Solve for the optimal weights
    n = X_1.shape[1]  # #. of other states, which is 38
    w_scipy = minimize(
        fun=partial(loss, X_0=X_0, X_1=X_1),
        jac=jac,
        x0=[0] * n,  # Initial guess of the solution. Set to zero's for simplicity
        constraints=constraints,
        bounds=[(0, 1)] * n  # Constraint 2: each weight is in [0, 1]
    ).x
    return w_scipy


def scm_cvxpy(X_0, X_1):
    n = X_1.shape[1]
    w = cp.Variable(n)

    # Objective function. Equivalent to `resid.T @ resid` above
    loss = cp.Minimize(cp.sum_squares(X_0 - X_1 @ w))

    # Constraints of the weights
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 1
    ]

    # Solve
    problem = cp.Problem(loss, constraints)
    problem.solve()
    return w.value


def plot_weights(nested: bool = False):
    """Plot the optimal weights from different implementations

    :param nested: Whether to use results from Stata's nested optimisation. Default is no
    """
    # Load the SCM results
    suffix = '_nested' if nested else ''
    stata = pd.read_csv(f'data/stata_synth{suffix}.csv', usecols=['_Co_Number', '_W_Weight'])
    stata.columns = ['state', 'stata']
    python = pd.read_csv('data/python_synth.csv')
    df = pd.merge(stata, python, left_index=True, right_index=True)
    del stata, python

    # Plot the optimal weights from different implementations
    fig, ax = plt.subplots(1, 2, width_ratios=[3, 1])
    df['x'] = range(len(df))
    ax[0].bar(df['x'] - 0.25, df['stata'], width=0.25, color='#ff7f00', label='Stata')
    ax[0].bar(df['x'], df['scipy'], width=0.25, color='#1f78b4', label='SciPy')
    ax[0].bar(df['x'] + 0.25, df['cvxpy'], width=0.25, color='#b2df8a', label='CVXPY')
    ax[0].set_xticks(df['x'], labels=df['state'], rotation=90)
    ax[0].set_ylabel('$w$')
    ax[0].legend(loc='upper left')
    ax[0].margins(x=0.01)
    df = df[df['stata'] > 0]
    df['x'] = range(len(df))
    ax[1].bar(df['x'] - 0.25, df['stata'], width=0.25, color='#ff7f00', label='Stata')
    ax[1].bar(df['x'], df['scipy'], width=0.25, color='#1f78b4', label='SciPy')
    ax[1].bar(df['x'] + 0.25, df['cvxpy'], width=0.25, color='#b2df8a', label='CVXPY')
    ax[1].set_xticks(df['x'], labels=df['state'], rotation=90)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/weights{suffix}.svg', bbox_inches='tight')
    pass


def plot_timing():
    # Violin plot
    df = pd.read_csv('data/timing.csv')
    fig, ax = plt.subplots()
    vp = ax.violinplot(df, showmeans=True, showextrema=False, showmedians=False, widths=0.3)
    colours = ['#ff7f00', '#1f78b4', '#8fbbd9', '#b2df8a']
    for i, p in enumerate(vp['bodies']):
        p.set_facecolor(colours[i])
    vp['cmeans'].set_color(colours)
    vp['cmeans'].set_linewidth(3)
    vp['cmeans'].set_capstyle('round')

    # Add labels
    ax.set_xticks(range(1, 5), labels=['Stata', 'SciPy', 'SciPy (with Jacobian)', 'CVXPY'])
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1000, 10000], labels=['10ms', '100ms', '1s', '10s'])
    ax.set_ylabel('Time')
    plt.tight_layout()
    plt.savefig('figures/timing.svg', bbox_inches='tight')
    pass


def main():
    # Compare if our implementations are consistent with Stata's
    X_0, X_1 = pre_process()
    w_scipy = scm_scipy(X_0, X_1)
    w_cvxpy = scm_cvxpy(X_0, X_1)
    python = pd.DataFrame({'scipy': w_scipy, 'cvxpy': w_cvxpy})
    python.to_csv('data/python_synth.csv', index=False)
    plot_weights(nested=False)
    plot_weights(nested=True)

    # Speed benchmark
    time_scipy = []
    time_scipy_jac = []
    time_cvxpy = []
    for _ in tqdm(range(1000)):
        start = time.time()
        X_0, X_1 = pre_process()
        scm_scipy(X_0, X_1)
        time_scipy.append(time.time() - start)
        start = time.time()
        X_0, X_1 = pre_process()
        scm_scipy(X_0, X_1, jac=lambda w: -2 * X_1.T @ (X_0 - X_1 @ w))
        time_scipy_jac.append(time.time() - start)
        start = time.time()
        X_0, X_1 = pre_process()
        scm_cvxpy(X_0, X_1)
        time_cvxpy.append(time.time() - start)
    time_stata = pd.read_csv('data/stata_timing.csv')

    # Plot speed
    df = pd.DataFrame({'stata': time_stata.iloc[:, 0], 'scipy': time_scipy,
                       'scipy_jac': time_scipy_jac, 'cvxpy': time_cvxpy})
    df *= 1000  # measure time in ms
    df.to_csv('data/timing.csv', index=False)
    plot_timing()
    pass


if __name__ == '__main__':
    main()
