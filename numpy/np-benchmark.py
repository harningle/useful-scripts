import timeit
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# np.show_config()
np.random.seed(42)

n = int(sys.argv[1])
blas = np.show_config(mode='dicts')['Build Dependencies']['blas']['name'].lower()
if 'mkl' in blas:
    lib = 'mkl'
elif 'openblas' in blas:
    lib = 'openblas'
else:
    raise ValueError(f'Unknown BLAS: {blas}')


X = np.random.rand(n, n)
X = X @ X.T  # So it's positive semi-definite
Y = np.random.rand(n, n)
x = np.random.rand(n)

plt.style.use('../default.mplstyle')

"""
def plot():
    mkl = json.load(open('data/mkl.json'))
    openblas = json.load(open('data/openblas.json'))

    # Vec. mul. and l2 norm
    fig, ax = plt.subplots()
    ax.bar([1, 4], [openblas['Vec. norm'], openblas['Mat.-vec. mul.']],
           color='#44A8FC', label='OpenBLAS')
    ax.bar([2, 5], [mkl['Vec. norm'], mkl['Mat.-vec. mul.']],
           color='#00285A', label='MKL')
    ax.set_xticks([1.5, 4.5], ['Vector $l^2$ norm', 'Matrix-vector multiplication'])
    ax.set_yticklabels([f'{i * 1000:.1f}' for i in ax.get_yticks()])
    ax.set_ylabel('Time (ms)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/benchmark_vec.svg', bbox_inches='tight')

    # Mat mul., mat. determinant, mat. inverse, and eigenvalues
    fig, ax = plt.subplots()
    ax.bar([1, 4, 7, 10], [openblas['Mat. mul.'], openblas['Mat. det.'], openblas['Inverse'],
                           openblas['Eigenvalue'] / 10],
           color='#44A8FC', label='OpenBLAS')
    ax.bar([2, 5, 8, 11], [mkl['Mat. mul.'], mkl['Mat. det.'], mkl['Inverse'],
                           mkl['Eigenvalue'] / 10],
           color='#00285A', label='MKL')
    ax.set_xticks([1.5, 4.5, 7.5, 10.5],
                  ['Matrix multiplication', 'Matrix determinant', 'Matrix inverse', 'Eigenvalues'])
    ax.set_ylabel('Time (s)')
    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks(), labels=[f'{i * 10:.1f}' for i in ax.get_yticks()])
    ax2.set_ylabel('Time (s), for eigenvalues')
    ax2.grid(False)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/benchmark_mat.svg', bbox_inches='tight')

    # Decompositions
    fig, ax = plt.subplots()
    ax.bar([1, 4, 7], [openblas['SVD'], openblas['QR'], openblas['Cholesky'] * 20],
           color='#44A8FC', label='OpenBLAS')
    ax.bar([2, 5, 8], [mkl['SVD'], mkl['QR'], mkl['Cholesky'] * 20],
           color='#00285A', label='MKL')
    ax.set_xticks([1.5, 4.5, 7.5], ['SVD', 'QR', 'Cholesky'])
    ax.set_ylabel('Time (s)')
    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks(), labels=[f'{i / 20:.1f}' for i in ax.get_yticks()])
    ax2.set_ylabel('Time (s), for Cholesky decomposition')
    ax2.grid(False)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/benchmark_decomposition.svg', bbox_inches='tight')
    return
"""

def plot():
    mkl = json.load(open('data/mkl.json'))
    openblas = json.load(open('data/openblas.json'))
    for k in mkl:
        mkl[k] = mkl[k] / openblas[k]
    del openblas
    l2_norm = mkl['Vec. norm']  # Outlier
    del mkl['Vec. norm']

    fig, ax = plt.subplots()
    ax.scatter(mkl.values(), mkl.keys(), color='#00285A', label='MKL')
    ax.axvline(1, color='#44A8FC', linestyle='dashed')
    ax.annotate(f'{l2_norm:.1f}',
                xy=(max(mkl.values()) * 0.87, -1), xytext=(max(mkl.values()) * 0.97, -1),
                arrowprops=dict(arrowstyle= '<-', color='#00285A', lw=1.5),
                color='#00285A', va='center')  # Use arrow to mark the outlier
    text = ax.text(1.02, 0.2, 'MKL ', color='#00285A',
                   va='center', ha='center', rotation='vertical')
    text = ax.annotate('has the same speed as', xycoords=text, xy=(0, 1), ha='left', va='bottom',
                       rotation='vertical', color='black')
    text = ax.annotate(' OpenBLAS', xycoords=text, xy=(0, 1), ha='left', va='bottom',
                       rotation='vertical', color='#44A8FC')
    ax.annotate('MKL faster', xy=(0.5, -0.5), xytext=(0.5, -0.5), color='#00285A',
                va='center', ha='center')
    ax.annotate('OpenBLAS faster', xy=(1.3, 5.5), xytext=(1.3, 5.5), color='#44A8FC',
                va='center', ha='center')
    yticks = ax.get_yticks()
    ax.set_yticks(yticks + [-1],
                  labels=['Matrix-vector multiplication', 'Matrix-matrix multiplication',
                          'Matrix determinant', 'Matrix inverse', 'Eigenvalue',
                          'SVD decomposition', 'QR decomposition', 'Cholesky decomposition',
                          'Vector $l^2$ norm'])
    ax.set_ylim(ax.get_ylim()[0] - 0.35, ax.get_ylim()[1])
    ax.set_xlabel('MKL speed up factor')
    plt.tight_layout()
    plt.savefig('figures/benchmark_mkl_openblas.svg', bbox_inches='tight')
    return


def main():
    res = {
        # Matrix-vector multiplication
        'Mat.-vec. mul.': timeit.timeit('np.dot(X, x)', globals=globals(), number=5),

        # Matrix multiplication
        'Mat. mul.': timeit.timeit('np.dot(X, Y)', globals=globals(), number=5),

        # Vec. norm
        'Vec. norm': timeit.timeit('np.linalg.norm(x)', globals=globals(), number=5),

        # Matrix determinant
        'Mat. det.': timeit.timeit('np.linalg.det(X)', globals=globals(), number=5),

        # Matrix inverse
        'Inverse': timeit.timeit('np.linalg.inv(X)', globals=globals(), number=5),

        # Matrix eigenvalues
        'Eigenvalue': timeit.timeit('np.linalg.eigvals(X)', globals=globals(), number=5),

        # Matrix decomposition
        'SVD': timeit.timeit('np.linalg.svd(X)', globals=globals(), number=5),
        'QR': timeit.timeit('np.linalg.qr(X)', globals=globals(), number=5),
        'Cholesky': timeit.timeit('np.linalg.cholesky(X)', globals=globals(), number=5)
    }
    with open(f'data/{lib}.json', 'w') as f:
        json.dump(res, f)
    return
    

if __name__ == '__main__':
    main()
    # plot()
