# -*- coding: utf-8 -*-
"""Benchmark of sum and sum of squares"""
import os
import timeit

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
plt.style.use('../default.mplstyle')

COLOURS = {
    'sum': 'black',
    'np.sum': '#4D77CF',
    'np.dot': '#4DABCF',
    'np.linalg.norm': '#013243',
    'cp.sum': '#1E8358',
    'cp.dot': '#8DC14E',
    'cp.linalg.norm': '#0D8080'
}


def perfplot(df: pd.DataFrame, filename: str):
    fig, ax = plt.subplots()
    for col in df:
        if col == 'sum':
            marker = '.'
        elif 'np' in col:
            marker = '+'
        else:
            marker = 'x'
        ax.plot(df.index, df[col], marker=marker, color=COLOURS[col], label=col)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$n$')
    ax.set_ylabel('time (s)')
    ax.legend(prop={'family': 'monospace'})
    plt.margins(x=0.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    return


def benchmark_sum():
    timing = {}
    for n in [2 ** x for x in range(1, 28)]:
        n = int(n)
        arr = np.random.rand(n)
        lst = arr.tolist()
        cp_arr = cp.array(arr)
        ones = np.ones_like(arr)
        cp_ones = cp.ones_like(cp_arr)
        timing[n] = {}

        # Python native sum
        timing[n]['sum'] = timeit.timeit(lambda: sum(lst), number=20)

        # Numpy sum
        timing[n]['np.sum'] = timeit.timeit(lambda: arr.sum(), number=20)

        # Dot product
        timing[n]['np.dot'] = timeit.timeit(lambda: np.dot(arr, ones), number=20)

        # cupy sum
        timing[n]['cp.sum'] = timeit.timeit(lambda: cp_arr.sum(), number=20)

        # cupy dot product
        timing[n]['cp.dot'] = timeit.timeit(lambda: cp.dot(cp_arr, cp_ones), number=20)

    perfplot(pd.DataFrame(timing).T, 'figures/sum.svg')
    return


def benchmark_sum_of_squares():
    timing = {}
    for n in [2 ** x for x in range(1, 28)]:
        n = int(n)
        arr = np.random.rand(n)
        cp_arr = cp.array(arr)
        lst = arr.tolist()
        timing[n] = {}

        # Python native sum of squares
        if n < 2 ** 20:
            timing[n]['sum'] = timeit.timeit(lambda: sum(x ** 2 for x in lst), number=20)

        # Numpy sum of squares
        timing[n]['np.sum'] = timeit.timeit(lambda: (arr ** 2).sum(), number=20)

        # Dot product
        timing[n]['np.dot'] = timeit.timeit(lambda: np.dot(arr, arr), number=20)

        # l2 norm
        timing[n]['np.linalg.norm'] = timeit.timeit(
            lambda: np.linalg.norm(arr, ord=2) ** 2, number=20
        )

        # cupy sum of squares
        timing[n]['cp.sum'] = timeit.timeit(lambda: (cp_arr ** 2).sum(), number=20)

        # cupy dot product
        timing[n]['cp.dot'] = timeit.timeit(lambda: cp.dot(cp_arr, cp_arr), number=20)

        # cupy l2 norm
        timing[n]['cp.linalg.norm'] = timeit.timeit(
            lambda: cp.linalg.norm(cp_arr, ord=2) ** 2, number=20
        )

    perfplot(pd.DataFrame(timing).T, 'figures/sum_of_squares.svg')
    return


def main():
    os.makedirs('figures', exist_ok=True)
    benchmark_sum()
    benchmark_sum_of_squares()


if __name__ == '__main__':
    main()
