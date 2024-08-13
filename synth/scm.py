# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('../default.mplstyle')


def main():
    # Load the SCM results
    stata = pd.read_csv('data/stata_synth.csv', usecols=['_Co_Number', '_W_Weight'])
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
    temp = df[df['stata'] > 0]
    temp['x'] = range(len(temp))
    ax[1].bar(temp['x'] - 0.25, temp['stata'], width=0.25, color='#ff7f00', label='Stata')
    ax[1].bar(temp['x'], temp['scipy'], width=0.25, color='#1f78b4', label='SciPy')
    ax[1].bar(temp['x'] + 0.25, temp['cvxpy'], width=0.25, color='#b2df8a', label='CVXPY')
    ax[1].set_xticks(temp['x'], labels=temp['state'], rotation=90)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/weights.svg', bbox_inches='tight')
    pass


if __name__ == '__main__':
    main()
