from collections import defaultdict
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

np.random.seed(42)

# N = 10_000_000
# M = 3
# df = pd.DataFrame(np.random.randint(low=0, high=100, size=(N, M)),
#                   columns=['a', 'b', 'c'])
# for col in df.columns:
#     df.loc[np.random.choice(N, int(N * 0.2), replace=False), col] = np.nan

# RUNS = 100

plt.style.use('../default.mplstyle')


def inplace(df):
    df.a.fillna(0, inplace=True)
    df.b.replace(50, 999, inplace=True)
    df.sort_values('c', inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def unchaining(df):
    df.a = df.a.fillna(0)
    df.b = df.b.replace(50, 999)
    df = df.sort_values('c')
    df = df.drop_duplicates()
    return df


def chaining(df):
    return (df
        .fillna({'a': 0})
        .replace({'b': {50: 999}})
        .sort_values('c')
        .drop_duplicates())


def timeit(func):
    start = time.time()
    func()
    return time.time() - start


def plot(df: pd.DataFrame):
    # Normalise inplace operations to unit variance
    chaining = df[['chaining', 'inplace', 'unchaining']]
    ops = ['replace', 'fillna', 'sort_values', 'drop_duplicates', 'reset_index', 'drop', 'dropna',
           'rename']
    for op in ops:
        sd = df[f'{op}_inplace_True'].std()
        for inplace in [True, False]:
            df[f'{op}_inplace_{inplace}'] /= sd

    # Compute 95% confidence intervals
    df = df.agg(['mean', lambda x: x.quantile(0.975), lambda x: x.quantile(0.025)])
    df.index = ['mean', 'uci', 'lci']
    df.loc['uci', :] = df.loc['uci', :] - df.loc['mean', :]
    df.loc['lci', :] = df.loc['mean', :] - df.loc['lci', :]

    # Plot
    fig, ax = plt.subplots()
    x = 0
    for op in ops:
        for inplace in [True, False]:
            ax.errorbar(
                x=x,
                y=df.loc['mean', f'{op}_inplace_{inplace}'],
                yerr=[[df.loc['lci', f'{op}_inplace_{inplace}']],
                      [df.loc['uci', f'{op}_inplace_{inplace}']]],
                fmt='', color='#E95C3B' if inplace else '#A7C9DE', capsize=3, lw=2, capthick=2
            )
            if op == 'replace':
                ax.bar(x, df.loc['mean', f'{op}_inplace_{inplace}'],
                    color='#800000' if inplace else '#000080',
                    label='Inplace' if inplace else 'Copy')
            else:
                ax.bar(x, df.loc['mean', f'{op}_inplace_{inplace}'],
                       color='#800000' if inplace else '#000080')
            x += 1
        x += 1
    
    ax.set_xticks(ticks=np.arange(0.5, len(ops) * 3 + 0.5, 3), labels=['df.' + i for i in ops],
                  rotation=45, fontfamily='monospace')
    ax.set_ylabel('Time (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_simple.svg', bbox_inches='tight')

    # Same plot for method chaining
    chaining = chaining.agg(['mean', lambda x: x.quantile(0.975), lambda x: x.quantile(0.025)]).T
    chaining.columns = ['mean', 'uci', 'lci']
    chaining.uci += chaining['mean']
    chaining.lci = chaining['mean'] - chaining['lci']
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(
        x=0,
        y=chaining['mean'].chaining,
        yerr=[[chaining['lci'].chaining], [chaining['uci'].chaining]],
        fmt='', color='#E95C3B', capsize=3, lw=2, capthick=2
    )
    ax.bar(0, chaining['mean'].chaining, color='#000080')
    ax.errorbar(
        x=1,
        y=chaining['mean'].inplace,
        yerr=[[chaining['lci'].inplace], [chaining['uci'].inplace]],
        fmt='', color='#E95C3B', capsize=3, lw=2, capthick=2
    )
    ax.bar(1, chaining['mean'].inplace, color='#800000')
    ax.errorbar(
        x=2,
        y=chaining['mean'].unchaining,
        yerr=[[chaining['lci'].unchaining], [chaining['uci'].unchaining]],
        fmt='', color='#60AB63', capsize=3, lw=2, capthick=2
    )
    ax.bar(2, chaining['mean'].unchaining, color='#008000')
    ax.set_xticks(ticks=[0, 1, 2], labels=['Method chaining', 'Inplace', 'Method not chaining'])
    ax.set_ylabel('Time (s)')
    plt.tight_layout()
    plt.savefig('benchmark_chaining.svg', bbox_inches='tight')
    return


if __name__ == '__main__':
    # speed = defaultdict(list)
    # for i in range(RUNS):
    #     temp = df.copy()
    #     speed['inplace'].append(timeit(lambda: inplace(temp)))
    #     temp = df.copy()
    #     speed['unchaining'].append(timeit(lambda: unchaining(temp)))
    #     temp = df.copy()
    #     speed['chaining'].append(timeit(lambda: chaining(temp)))

    #     ops = {
    #         'replace': lambda temp, inplace: temp.replace({50: 999}, inplace=inplace),
    #         'fillna': lambda temp, inplace: temp.fillna(0, inplace=inplace),
    #         'sort_values': lambda temp, inplace: temp.sort_values('a', inplace=inplace),
    #         'drop_duplicates': lambda temp, inplace: temp.drop_duplicates(inplace=inplace),
    #         'reset_index': lambda temp, inplace: temp.reset_index(inplace=inplace),
    #         'drop': lambda temp, inplace: temp.drop(columns=['a'], inplace=inplace),
    #         'dropna': lambda temp, inplace: temp.dropna(inplace=inplace),
    #         'rename': lambda temp, inplace: temp.rename(columns={'a': 'A'}, inplace=inplace)
    #     }
    #     for op, func in ops.items():
    #         for inpl in [True, False]:
    #             temp = df[['a']]
    #             speed[f'{op}_inplace_{inpl}'].append(timeit(lambda: func(temp, inpl)))
    
    # speed = pd.DataFrame(speed)
    # speed.to_csv('speed.csv', index=False)

    plot(pd.read_csv('speed.csv'))

