# -*- coding: utf-8 -*-
from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as grid_spec
import seaborn as sns
from scipy.stats import gaussian_kde


def kde(hist: dict[float, int], n_samples: int = 100, bw: float = 0.25) \
        -> tuple[np.ndarray, np.ndarray]:
    """Take a histogram and return an estimated numerical Gaussidan kernel density

    :param hist: Histogram as a dictionary where keys are bin centers and values are counts
    :param n_samples: Number of samples to generate for KDE
    :param bw: Bandwidth for KDE. Default is 0.25 (give nice charts; not mathematically good...)
    :return: Tuple of (x values, estimated density values)
    """
    x = list(hist.keys())
    y = list(hist.values())
    kde_estimator = gaussian_kde(x, weights=y, bw_method=bw)
    x_eval = np.linspace(min(x) - 1, max(x) + 1, n_samples)
    y_eval = kde_estimator(x_eval)
    return x_eval, y_eval


def ridge_plot(x: Sequence,
               density: list[tuple[np.ndarray, np.ndarray]],
               means: Optional[Sequence[float]] = None,
               mean_below_label: bool = False) -> None:
    """
    :param x: A collection of variables/categories for each ridge plot
    :param density: Density of each `x` variable， as tuples of (x values, density values)
    :param means: Avg. for each `x` variable to be marked on the plot. If provided, density will be
                  coloured according to the means of `x`
    :param mean_below_label: Whether to put the mean value below the label of each ridge plot
    """
    fig = plt.figure(figsize=(9, 16))
    gs = grid_spec.GridSpec(len(x), 1)
    min_x = min([min(d[0]) for d in density])
    max_x = max([max(d[0]) for d in density])

    if means:
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        norm = plt.Normalize(min(means), max(means))
        colours = [cmap(norm(m)) for m in means]
    else:
        colours = ['tab:blue'] * len(x)

    for i, (x_vals, y_vals) in enumerate(density):
        ax = fig.add_subplot(gs[i])
        ax.plot(x_vals, y_vals, color=colours[i])
        ax.fill_between(x_vals, 0, y_vals, alpha=0.3, color=colours[i])
        ax.patch.set_alpha(0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)
        label = x[i]
        if means and mean_below_label:
            label += f'\n({list(means)[i]:.0f})'
        ax.text(0.95 * min_x - 0.05 * max_x, 0, label, ha='right')

    gs.update(hspace=-0.7)
    return


def main():
    df = pd.read_csv('data.csv')
    data: dict[str, dict[str, float | tuple[np.ndarray, np.ndarray]]] = {}
    for x in df.columns[1:]:
        mean = (df[x] * df.score).sum() / 10
        density = kde(hist={k: v for k, v in zip(df.score, df[x])})
        data[x] = {'mean': mean, 'density': density}
    data = dict(sorted(data.items(), key=lambda item: item[1]['mean']))
    ridge_plot(x=list(data.keys()), density=[v['density'] for v in data.values()],
               means=[v['mean'] for v in data.values()])
    plt.savefig('ridge_plot.svg', bbox_inches='tight')
    return


if __name__ == '__main__':
    main()
