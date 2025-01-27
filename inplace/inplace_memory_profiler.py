from memory_profiler import profile
import pandas as pd


@profile(precision=3)
def foo(df):
    df = pd.DataFrame(1, index=range(100_000_000), columns=['a'])
    df.a.replace({1: 100}, inplace=True)
    df.sort_values('a', inplace=True)

    # df.a = df.a.replace({1: 100}, inplace=False)
    # df = df.sort_values('a', inplace=False)
    return df


if __name__ == '__main__':
    foo()
