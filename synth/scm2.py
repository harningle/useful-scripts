# -*- coding: utf-8 -*-
from functools import partial
import itertools
import time
from typing import Iterable, Literal
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
from multiprocess import Pool
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

plt.style.use('../default.mplstyle')

raw_df = pd.read_stata('https://github.com/microsoft/SparseSC/raw/master/replication/smoking.dta')
raw_df['year'] = raw_df['year'].astype(int)

X_0: np.ndarray
X_1: np.ndarray
Y_0: np.ndarray
Y_1: np.ndarray
n: int
m: int
stata: dict


def pre_process():
    pre = raw_df[raw_df['year'] <= 1988][['state', 'year', 'cigsale', 'beer', 'lnincome',
                                          'retprice', 'age15to24']]

    # 1984-1988 avg. beer
    beer = pre[pre['year'].between(1984, 1988)].groupby('state')['beer'].mean()

    # Pre-treatment avg. income, cigarette price, and age structure
    temp = pre.groupby('state')[['lnincome', 'retprice', 'age15to24']].mean()
    temp['beer'] = beer
    del beer
    temp = temp[['beer', 'lnincome', 'retprice', 'age15to24']]

    # Cigarette sales in 1988, 1980, and 1975
    cigsale = pre[pre['year'].isin([1988, 1980, 1975])]
    cigsale = cigsale[['state', 'year', 'cigsale']] \
        .set_index(['state', 'year']) \
        .unstack(level=0) \
        .stack(level=0, future_stack=True) \
        .iloc[::-1]  # Sort years in descending order, following Stata's synth...

    # X_0 and X_1
    global X_0, X_1, Y_0, Y_1, n, m
    pre = pd.concat([temp.T, cigsale])
    temp = pre.values
    avg = temp.mean(axis=1, keepdims=True)
    std_dev = temp.std(axis=1, keepdims=True)
    temp = (temp - avg) / std_dev
    pre = pd.DataFrame(temp, index=pre.index, columns=pre.columns)
    del temp, cigsale
    X_0 = pre['California'].values
    X_1 = pre.drop(columns='California').values
    del pre
    n = X_1.shape[1]  # #. of donors
    m = X_0.shape[0]  # #. of matching vars.

    # Y_0 and Y_1
    y = raw_df[(raw_df['year'] <= 1988)][['state', 'year', 'cigsale']] \
        .set_index(['state', 'year']) \
        .unstack(level=0) \
        .stack(level=0, future_stack=True)
    Y_0 = y['California'].values
    Y_1 = y.drop(columns='California').values
    del y
    pass


def solve_w(v):
    """For a given V, solve for w"""
    w = cp.Variable(n)
    V = np.zeros((m, m))
    np.fill_diagonal(V, v)
    similarity = cp.sum(V @ cp.square(X_0 - X_1 @ w))
    constr = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 1
    ]
    problem = cp.Problem(cp.Minimize(similarity), constr)
    problem.solve()
    return w.value


def reg_based_initial_guess():
    X = np.hstack((X_0[:, None], X_1))
    Y = np.hstack((Y_0[:, None], Y_1))
    reg = LinearRegression(fit_intercept=True).fit(X.T, Y.T)
    coef = reg.coef_
    v_0 = np.diag(coef.T @ coef)
    v_0 = v_0 / v_0.sum()
    return v_0


def initial_guess(method: Literal['equal', 'reg', 'random'], level: Literal['single', 'nested']) \
        -> Iterable[float]:
    """Generate initial guesses for the relative importance

    :param method: Equal weights, regression-based weights, or random weights
    :param level: Whether the problem is a single level reformulation or the original nested one.
                  This determines the number/shape of the initial guess
    """
    # Initial guess of the relative importance V
    if method == 'equal':
        initial_V = np.ones(m) / m
    elif method == 'reg':
        initial_V = reg_based_initial_guess()
    elif method == 'random':
        initial_V = np.random.dirichlet(alpha=np.ones(m))
    else:
        raise ValueError(f'Unknown method: {method}')

    if level == 'nested':
        return initial_V  # If nested, then #. of relative importance is #. of matching vars.

    elif level == 'single':  # If single, then [w, V, \lambda, \mu, \nu]
        initial_w = solve_w(initial_V)
        initial_lambda = 1e-3  # TODO: how to choose this?
        initial_mu = np.zeros(n)
        initial_nu = np.zeros(n)
        return np.concatenate([initial_w, initial_V, [initial_lambda], initial_mu, initial_nu])
    else:
        raise ValueError(f'Unknown level: {level}')


def loss(x, level: Literal['single', 'nested'], loss_type: Literal['RSS', 'l2']) -> float:
    """For a given V, solve for w and compute the pre trend loss"""
    if level == 'single':
        w = x[:n]  # If single, then weights are the first n elements
    elif level == 'nested':  # If nested, then the input `x` is the V and we need to solve for w
        try:
            w = solve_w(x)
        except:  # Sometimes optimisers try V that is not positive semi-definite
            return 1e6  # Effectively infinite loss
    else:
        raise ValueError(f'Unknown level: {level}')

    resid = Y_0 - Y_1 @ w
    if loss_type == 'RSS':
        return resid.T @ resid
    elif loss_type == 'l2':
        return np.linalg.norm(resid)
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')


def jac_loss(x, level: Literal['single'], loss_type: Literal['RSS', 'l2']) -> np.ndarray:
    """Jacobian of the loss function. Only available if it's a single level optimisation"""
    assert level == 'single', 'Jacobian is only available for single level optimisation'
    w = x[:n]
    j = np.zeros(len(x))
    if loss_type == 'RSS':
        j[:n] = -2 * Y_1.T @ (Y_0 - Y_1 @ w)
    elif loss_type == 'l2':
        resid = Y_0 - Y_1 @ w
        j[:n] = -(resid.T @ Y_1) / np.linalg.norm(resid)
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')
    return j


# Stationary, i.e. \nabla L = 0
def stationary(x):
    w = x[:n]
    V = np.diag(x[n:n + m])
    lambda_ = x[n + m]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]
    grad = -2 * X_1.T @ V @ (X_0 - X_1 @ w) + lambda_ * np.ones(n) - mu + nu
    return grad


def jac_stationary(x):
    w = x[:n]
    V = np.diag(x[n:n + m])
    j = np.zeros((n, len(x)))

    # w.r.t. w
    j[:, :n] = 2 * X_1.T @ V @ X_1

    # w.r.t. V
    for i in range(m):
        j[:, n + i] = -2 * X_1.T[:, i] * (X_0[i] - X_1[i] @ w)

    # w.r.t. \lambda
    j[:, n + m] = np.ones(n)

    # w.r.t. \mu and \nu
    j[:, n + m + 1:n + m + 1 + n] = -np.eye(n)
    j[:, n + m + 1 + n:] = np.eye(n)
    return j


# Complementary slackness, i.e. \mu * w = 0 and \nu * (w - 1) = 0, element-wise
def complementary_slackness(x):
    w = x[:n]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]
    return np.concatenate([mu * w, nu * (w - 1)])


def jac_complementary_slackness(x):
    w = x[:n]
    mu = x[n + m + 1:n + m + 1 + n]
    nu = x[n + m + 1 + n:]
    j = np.zeros((2 * n, len(x)))

    # Derivatives of mu * w with respect to w and mu
    j[:n, :n] = np.diag(mu)
    j[:n, n + m + 1:n + m + 1 + n] = np.diag(w)

    # Derivatives of nu * (w - 1) with respect to w and nu
    j[n:, :n] = np.diag(nu)
    j[n:, n + m + 1 + n:] = np.diag(w - 1)
    return j


def constraints(level: Literal['single', 'nested'], require_v_sum: bool, eps: float) \
        -> list[LinearConstraint | NonlinearConstraint | None]:
    constr = []

    if level == 'nested':
        if require_v_sum:
            constr.append(LinearConstraint(np.ones(m), 1 - eps, 1 + eps))
    elif level == 'single':
        n_x = 3 * n + m + 1  # Length of the input vector
        if require_v_sum:
            v_sum = np.zeros(n_x)
            v_sum[n:n + m] = 1
            constr.append(LinearConstraint(v_sum, 1 - eps, 1 + eps))

        # 0 <= v_i <= 1
        v_bounds = np.zeros((m, n_x))
        v_bounds[:, n:n + m] = np.eye(m)

        # \sum w_j = 1: eq. (4)
        w_sum = np.zeros(n_x)
        w_sum[:n] = 1

        # 0 <= w_j <= 1
        w_bounds = np.zeros((n, n_x))
        w_bounds[:, :n] = np.eye(n)

        # Dual feasibility
        mu_nu_bounds = np.zeros((n * 2, n_x))
        mu_nu_bounds[:, n + m + 1:] = np.eye(n * 2)

        constr.extend([
            LinearConstraint(v_bounds, 0, 1 + eps),                     # 0 <= v_i <= 1
            NonlinearConstraint(stationary, 0, 0, jac=jac_stationary),  # Stationary
            NonlinearConstraint(complementary_slackness, 0, 0,          # Complementary slackness
                                jac=jac_complementary_slackness),
            LinearConstraint(w_bounds, 0, 1 + eps),                     # 0 <= w_j <= 1
            LinearConstraint(w_sum, 1 - eps, 1 + eps),                  # \sum w_j = 1
            LinearConstraint(mu_nu_bounds, 0, np.inf)                   # Dual feasibility
        ])
        """
        `0, 1 + eps` instead of `0 - eps, 1 + eps` is chosen on purpose, i.e. w should be strictly
        non-negative, though it can be slightly larger than one (`1 + eps` above). This is because
        the inner problem will otherwise become non-convex if we remove the non-negativity
        constraints.
        
        I also do not relax the constraints that come with the single level reformulation, i.e. KKT
        conditions. I'm not sure if mathematically it makes sense to relax them, so I keep them
        strict, i.e. without +/- `eps`.
        """
    else:
        raise ValueError(f'Unknown level: {level}')
    return constr


def solve(
        initial_guess_method: Literal['equal', 'reg', 'random'],
        level: Literal['single', 'nested'],
        require_v_sum: bool,
        optimiser: str,
        loss_type: Literal['RSS', 'l2'],
        constraint_strictness: float
) -> tuple[float, float, np.ndarray]:
    """Find the relative importance V to minimise the loss

    :param initial_guess_method: Equal, regression-based, or random initial guess
    :param level: Single or nested level reformulation
    :param require_v_sum: Whether the relative importance should sum up to one
    :param optimiser: SciPy optimiser to use. See `method` in `scipy.optimize.minimize`
    :param loss_type: RSS or L2 loss
    :param constraint_strictness: How much we can violate the constraints, e.g. 0.1 means we can
                                  violate the constraints by 10%
    :return: (loss, time used, optimal V)
    """
    # If initial guess is random, try ten random guesses. Otherwise, only one initial guess
    tries = 10 if initial_guess_method == 'random' else 1
    start = time.time()
    l = float('inf')
    x = None
    for _ in range(tries):
        x0 = initial_guess(initial_guess_method, level)

        # Constraints
        constr = constraints(level, require_v_sum, constraint_strictness)

        # Solve
        if level == 'nested':  # If nested optimisation, then need to set the bounds for `V`
            v = minimize(
                fun=partial(loss, level=level, loss_type=loss_type),
                x0=x0,
                method=optimiser,
                constraints=constr,
                bounds=[(0 - constraint_strictness, 1 + constraint_strictness)] * len(x0)
            )
            temp = v.x
            temp[temp < 0] = 0  # Sometimes will get negative V bc. bounds are not strictly enforced
            solve_w(temp)  # Also need to solve for `w` using the optimal `V` found above
        elif level == 'single':  # If single level optimisation, then can use Jacobian for loss, and no
            v = minimize(        # bound is needed as it's handled by the constraints
                fun=partial(loss, level=level, loss_type=loss_type),
                x0=x0,
                method=optimiser,
                constraints=constr,
                jac=partial(jac_loss, level=level, loss_type=loss_type),
            )
        else:
            raise ValueError(f'Unknown level: {level}')
        if v.fun < l:
            l = v.fun
            x = v.x
    end = time.time()
    return l, end - start, x


def worker(args):
    np.random.seed(args[-1])  # See https://stackoverflow.com/a/6914470/12867291
    sol = solve(*args[:-1])
    return [*args[:-1], sol[0], sol[1]]


def plot_weights(v: np.ndarray, w: np.ndarray, l: float, out: str):
    """
    Plot the weights and relative importance of Python and Stata. The Stata's loss will be read
    from the global variable `stata`

    :param v: Python implementation's V
    :param w: Python implementation's w
    :param l: Python implementation's loss
    :param out: Output image path
    """
    # Plot v and w in two bar charts
    fig, ax = plt.subplots(1, 2, width_ratios=[1, 3])
    ax[0].bar(np.arange(m) - 0.125, stata['v'], color='#f87530', width=0.25)
    ax[0].bar(np.arange(m) + 0.125, v, color='#004b93', width=0.25)
    ax[0].set_ylabel('$V$')
    ax[0].set_xticks(range(m),
                     labels=['$beer_{1984, \\cdots, 1988}$', '$lnincome$', '$retprice$',
                             '$age15to24$', '$cigsale_{1988}$', '$cigsale_{1980}$',
                             '$cigsale_{1975}$'],
                     rotation=90)
    ax[1].bar(np.arange(n) - 0.125, stata['w'], color='#f87530', width=0.25,
              label=f'Stata (loss={stata["loss"]:.3f})')
    ax[1].bar(np.arange(n) + 0.125, w, color='#004b93', width=0.25,
              label=f'Python (loss={l:.3f})')
    ax[1].set_ylabel('$\\vec{w}$')
    ax[1].set_xticks(range(len(stata['w'])),
                     labels=raw_df['state'].cat.categories.drop('California'),
                     rotation=90,
                     fontsize=9)
    ax[1].legend()
    ax[1].margins(x=0.01)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    pass


def plot_loss():
    losses = pd.read_csv('data/losses_initial_guess.csv')
    fig, ax = plt.subplots()
    ax.hist(losses['loss'], bins=range(50, int(losses['loss'].max()) + 10, 10),
            color='#0e5599', edgecolor='none')
    ax.axvline(stata['loss'], color='#f87530', linestyle='dashed', label="Stata's loss")
    ax.legend(loc='upper right')
    ax.set_xlabel("Python's Loss")
    ax.set_ylabel('Frequency')
    ax.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('figures/initial_guess_loss.svg', bbox_inches='tight')


def plot_loss_time():
    # Bar chart: avg. loss across different settings
    df = pd.read_csv('data/results.csv')
    df.loc[df['loss_type'] == 'l2', 'loss'] = df['loss'] ** 2  # So everything is in RSS now
    df = df[df['loss'] < df['loss'].quantile(0.99)]  # Drop outliers
    fig, ax = plt.subplots(figsize=(9.6, 6))
    i = 0
    colours = ['#87ceeb', '#679dd1', '#426eb7', '#00429d']
    ticks = []
    labels = []
    for j, setting in enumerate(['initial_guess', 'level', 'require_v_sum', 'optimiser',
                                 'loss_type', 'constraint_strictness']):
        avg_loss = df.groupby(setting)['loss'].mean()
        lci = df.groupby(setting)['loss'].quantile(0.025)
        uci = df.groupby(setting)['loss'].quantile(0.975)
        ax.bar(i + np.arange(len(avg_loss)), avg_loss, yerr=[avg_loss - lci, uci - avg_loss],
               capsize=3, color=colours[:len(avg_loss)], error_kw={'elinewidth': 1})
        ticks.extend(i + np.arange(len(avg_loss)))
        labels.extend(pd.Series(avg_loss.index).replace({
            'equal': 'Equal',
            'reg': 'Reg. based',
            'random': '10 random init.',
            'nested': 'Nested',
            'single': 'Single',
            'l2': '$\ell^2$',
            0: '0'
        }))
        i += len(avg_loss) + 1
    ax.annotate('Initial guess', xy=(1, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.annotate('Problem level', xy=(4.5, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.annotate(r'$\sum v_i \questeq 1$', xy=(7.5, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.annotate('Optimiser', xy=(11, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.annotate('Loss type', xy=(14.5, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.annotate('Constr. violation tol.', xy=(18.5, -400), annotation_clip=False,
                verticalalignment='center', horizontalalignment='center')
    ax.margins(x=0.01)
    ax.set_xticks(ticks, labels, rotation=90)
    ax.set_ylabel('Loss')
    plt.tight_layout()
    plt.savefig('figures/loss_across_settings.svg', bbox_inches='tight')

    # Let's only focus on losses that are good enough
    df = df \
        .groupby(['initial_guess', 'level', 'require_v_sum', 'optimiser', 'loss_type',
                  'constraint_strictness']) \
        .agg({
            'loss': ['mean', lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)],
            'time': ['mean', lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)]
        }) \
        .reset_index()
    df.columns = ['initial_guess', 'level', 'require_v_sum', 'optimiser', 'loss_type',
                  'constraint_strictness', 'loss', 'loss_lci', 'loss_rci', 'time', 'time_lci',
                  'time_rci']
    df = df[df['loss'] < df['loss'].quantile(0.1)]
    df.replace({
        'level': {
            'single': 'Single',
            'nested': 'Nested'
        },
        'initial_guess': {
            'random': '10 random guesses',
            'reg': 'Regression based',
            'equal': 'Equal'
        }
    }, inplace=True)

    # Scatter plot in plotly, with each dot as an individual combination of settings
    fig = go.Figure()
    for i in df.itertuples():
        colour = '#000080' if i.initial_guess == 'Regression based' else '#008000'
        fig.add_trace(go.Scatter(
            x=[i.time],
            y=[i.loss],
            customdata=[[i.initial_guess, i.level, i.require_v_sum, i.optimiser, i.loss_type,
                        i.constraint_strictness]],
            mode='markers',
            marker=dict(size=8),
            marker_color=colour,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[i.loss_rci - i.loss],
                arrayminus=[i.loss - i.loss_lci],
                width=0,
                thickness=1
            ),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[i.time_rci - i.time],
                arrayminus=[i.time - i.time_lci],
                width=0,
                thickness=1
            )
        ))
    with open('data/stata_time_nested.csv', 'r') as f:
        stata_time = float(f.read().strip())
    fig.add_vline(x=stata_time, line_dash='dash', line_color='#800000', line_width=2)
    fig.add_hline(y=stata['loss'], line_dash='dash', line_color='#800000', line_width=2)
    fig.update_traces(hovertemplate='<br>'.join([
        'Loss: %{y:.3f}',
        'Time: %{x:.3f}',
        'Initial guess: %{customdata[0]}',
        'Nested or single-level: %{customdata[1]}',
        'Constraint Σv<sub>i</sub> = 1: %{customdata[2]}',
        'Optimiser: %{customdata[3]}',
        'Loss type: %{customdata[4]}',
        'Constraint violation tolerance: %{customdata[5]}'
    ]))
    fig.update_layout(xaxis_title='Time', yaxis_title='Loss', showlegend=False, template='none',
                      font={'size': 16}, margin={'l': 70, 'r': 10, 't': 30, 'b': 50},
                      paper_bgcolor='rgba(255,255,255,0)', plot_bgcolor='rgba(255,255,255,0)')
    fig.update_xaxes(zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.write_html('figures/scatter_loss_time.html')
    pass


def main():
    # Get X_0, X_1, Y_0, Y_1
    pre_process()
    global n, m
    n = X_1.shape[1]
    m = X_0.shape[0]

    # Read the Stata results
    global stata
    stata = pd.read_csv('data/stata_synth_nested.csv')
    stata = {'w': stata['w'].values, 'v': stata['v'][~np.isnan(stata['v'])]}
    stata['loss'] = ((Y_0 - Y_1 @ stata['w']) ** 2).sum()

    # Default SciPy vs Stata
    np.random.seed(42)
    l, t, v = solve(
        initial_guess_method='equal',
        level='nested',
        require_v_sum=True,
        optimiser='SLSQP',
        loss_type='RSS',
        constraint_strictness=0
    )
    w = solve_w(v)
    plot_weights(v, w, l, 'figures/v_w_equal_guess.svg')

    # Losses if we try different initial guesses
    np.random.seed(42)
    losses = []
    for _ in range(100):
        try:
            l, _, _ = solve('random', 'nested', True, 'SLSQP', 'RSS', constraint_strictness=0)
            losses.append(l)
        except:  # Sometimes our initial guess is so bad that CVXPY cannot solve w
            losses.append(np.nan)
    losses = pd.DataFrame({'loss': losses, 'guess': range(100)})
    losses.to_csv('data/losses_initial_guess.csv', index=False)
    plot_loss()

    # Try all combinations
    results = []
    warnings.filterwarnings('ignore', category=UserWarning)  # Suppress singular matrix warning
    warnings.filterwarnings('ignore', category=RuntimeWarning)  # Clipping to bounds warning
    combs = []
    np.random.seed(42)
    for x0, lev, req_v_sum, constr_strictness, l_type in itertools.product(
        ['equal', 'reg', 'random'],
        ['single', 'nested'],
        [True, False],
        [0, 1e-8, 0.01, 0.05],
        ['RSS', 'l2']
    ):
        optimisers = ['SLSQP']
        if (lev == 'nested') and (req_v_sum is False):
            # For nested problem, the only constraints are on V. If no sum up to one constraint,
            # then the only constraint is V being bounded in [0, 1]. Then can use more optimisers
            optimisers.extend(['L-BFGS-B', 'Nelder-Mead'])
        for opt in optimisers:
            combs.append(
                [x0, lev, req_v_sum, opt, l_type, constr_strictness, np.random.randint(1e8)]
            )
    combs = list(itertools.chain.from_iterable(itertools.repeat(x, 100) for x in combs))
    np.random.shuffle(combs)

    # Four cores on the server, use 50%
    with Pool(2) as p:
        with tqdm(total=len(combs)) as pbar:
            for i in p.imap_unordered(worker, combs):
                results.append(i)
                pbar.update()
    results = pd.DataFrame(results,
                           columns=['initial_guess', 'level', 'require_v_sum', 'optimiser',
                                    'loss_type', 'constraint_strictness', 'loss', 'time'])
    results.to_csv('data/results.csv', index=False)

    # Plot loss and time for each combination
    plot_loss_time()
    pass


if __name__ == '__main__':
    main()
