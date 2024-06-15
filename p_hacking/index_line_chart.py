# -*- coding: utf-8 -*-
"""p-hacking by selecting sample time frame to show a desired trend"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

rc = {'figure.figsize': (9.6, 5.4),          # Set figure ratio to be 16:9
      'axes.facecolor': 'white',             # Remove background colour
      'axes.grid': True,                     # Turn on grid
      'axes.grid.which': 'both',             # Show both major and minor ticks
      'axes.edgecolor': '0',                 # Set axes edge color to be black
      'axes.linewidth': '0.8',               # Thicker axis lines
      'grid.color': '0.9',                   # Lighter grey grid lines
      'axes.xmargin': 0.02,                  # Reduce the gap between chart and axis
      'axes.ymargin': 0.02,
      'font.size': 10,
      'axes.titlesize': 'medium',
      'axes.labelsize': 'large',
      'svg.fonttype': 'none'}                # Text as text, rather than paths, in exported .svg
plt.rcdefaults()
plt.rcParams.update(rc)


def buffett():
    """Berkshire Hathaway vs S&P 500, with a time range slider"""
    # Rawdata from https://www.berkshirehathaway.com/letters/2023ltr.pdf
    df = pd.read_csv('rawdata/brk_spx_annual.csv')  # [year, BRK annual ret., S&P 500 annual ret.]
    df['brk'] = df['brk']/100 + 1  # Annual change 78.9% --> price becomes 1.789x of last year
    df['spx'] = df['spx']/100 + 1
    df = pd.concat([pd.DataFrame({'year': 1964, 'brk': 1, 'spx': 1}, index=[0]), df],
                   ignore_index=True)  # 1964 = 1

    # Line chart of two time series, full sample, linear scale
    df['brk_cum'] = df['brk'].cumprod()
    df['spx_cum'] = df['spx'].cumprod()
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(df['year'], df['brk_cum'], label='Berkshire Hathaway', color='#000080')
    ax.plot(df['year'], df['spx_cum'], label='S&P 500', color='#800000')
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    ax.set_xticks(list(range(1965, 2023, 10)) + [2023])
    ax.grid(which='minor', linewidth=0.6)
    ax.set_ylabel('Value when we invested $1 in 1964')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('figure/brk_spx.svg', bbox_inches='tight')

    # Use log scale instead, if want to make S&P 500 "stronger"
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=[1.0]))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig.savefig('figure/brk_spx_log.svg', bbox_inches='tight')

    # Restrict the sample to 2004-2023, linear scale
    # Use monthly data for more granularity. Data from Yahoo Finance
    df = pd.read_csv('rawdata/brk_spx_monthly.csv')
    df['month'] = pd.to_datetime(df['month'])
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(df['month'], df['brk'], label='Berkshire Hathaway', color='#000080')
    ax.plot(df['month'], df['spx'], label='SPDR S&P 500 ETF Trust', color='#800000')
    ax.set_xticks(pd.date_range('2004-01-01', '2024-01-01', freq='2YS', inclusive='both'))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.grid(which='minor', linewidth=0.6)
    ax.set_ylabel('Value when we invested $1 in Jan. 2004')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('figure/brk_spx_2004_2023.svg', bbox_inches='tight')

    # Restrict to year 2023, linear scale. Use daily data from Yahoo Finance
    df = pd.read_csv('rawdata/brk_spx_daily.csv')
    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(df['date'], df['brk'], label='Berkshire Hathaway', color='#000080')
    ax.plot(df['date'], df['spx'], label='SPDR S&P 500 ETF Trust', color='#800000')
    ax.set_xticks([
        pd.Timestamp('2023-01-03'), pd.Timestamp('2023-04-03'), pd.Timestamp('2023-07-03'),
        pd.Timestamp('2023-10-02'), pd.Timestamp('2023-12-29')]
    )
    ax.set_xticks([
        pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-01'),
        pd.Timestamp('2023-06-01'), pd.Timestamp('2023-08-01'), pd.Timestamp('2023-09-01'),
        pd.Timestamp('2023-11-01'), pd.Timestamp('2023-12-01')
    ], minor=True)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.grid(which='minor', linewidth=0.6)
    ax.set_ylabel('Value when we invested $1 in 2023/01/03')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('figure/brk_spx_2023.svg', bbox_inches='tight')
    pass


def adh2013():
    """Replicate Figure 1 in Autor et al. (AER 2013) and redo the same in a longer time frame"""
    # Pure replication of the original figure
    df = pd.read_stata('rawdata/figure1_data.dta')
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    l1 = ax.plot(df['year'], df['impr'], label='China import penetration ratio', color='#000080')
    ax2 = ax.twinx()
    l2 = ax2.plot(df['year'], df['cpsmanufemppop'], label='Manufacturing employment/population',
                  color='#800000', linestyle='dashed')
    ax.set_ylabel('Import penetration')
    ax2.set_ylabel('Manufacturing emp./pop.', rotation=270, labelpad=20)
    ax.set_yticks(np.arange(0, 0.06, 0.01))
    ax2.set_yticks(np.arange(0.08, 0.16, 0.02))
    ax2.grid(False)
    ax.set_xticks(range(1987, 2009, 2))
    ax.legend(l1 + l2, [i.get_label() for i in l1 + l2], loc='upper center')
    plt.tight_layout()
    fig.savefig('figure/adh2013.svg', bbox_inches='tight')

    # Build our own data, from raw sources
    pop = pd.read_csv('rawdata/CLF16OV.csv')  # CLF16OV: civilian labour force (persons >= 16)
    manu = pd.read_csv('rawdata/MANEMP.csv')  # MANEMP: all employees, manufacturing
    manu = pop.merge(manu, on='DATE', how='inner')
    manu['manu_share'] = manu['MANEMP'] / manu['CLF16OV']
    manu['DATE'] = pd.to_datetime(manu['DATE'])
    manu.rename(columns={'DATE': 'date'}, inplace=True)
    del pop
    gdp = pd.read_csv('rawdata/GDP.csv')    # GDP: GDP, seasonally adj.
    imp = pd.read_csv('rawdata/IMPGS.csv')  # IMPGS: imports of goods and services, seasonally adj.
    exp = pd.read_csv('rawdata/EXPGS.csv')  # EXPGS: exports of goods and services, seasonally adj.
    imp_from_china = pd.read_csv('rawdata/IMPCH.csv')  # IMPCH: U.S. imports of goods by customs
                                                       # basis from China, NOT seasonally adj.
    imp_from_china['IMPCH'] = imp_from_china['IMPCH'] \
        .rolling(window=12).mean()  # Smooth out seasonality
    imp_from_china['DATE'] = imp_from_china['DATE'] \
        .str.replace('-02-', '-01-') \
        .str.replace('-03-', '-01-') \
        .str.replace('-05-', '-04-') \
        .str.replace('-06-', '-04-') \
        .str.replace('-08-', '-07-') \
        .str.replace('-09-', '-07-') \
        .str.replace('-11-', '-10-') \
        .str.replace('-12-', '-10-')
    imp_from_china = imp_from_china.groupby('DATE').sum().reset_index()  # To quarterly
    imp_from_china['IMPCH'] /= 1000  # To billion USD
    penetration = gdp \
        .merge(imp, on='DATE', how='inner') \
        .merge(exp, on='DATE', how='inner') \
        .merge(imp_from_china, on='DATE', how='inner')
    del gdp, imp, exp, imp_from_china
    penetration['expenditure'] = penetration['GDP'] + penetration['IMPGS'] - penetration['EXPGS']
    penetration['penetration'] = penetration['IMPCH'] / penetration['expenditure']
    penetration['DATE'] = penetration['DATE'] \
        .str.replace('-01-', '-02-') \
        .str.replace('-04-', '-05-') \
        .str.replace('-07-', '-08-') \
        .str.replace('-10-', '-11-')
    penetration['DATE'] = pd.to_datetime(penetration['DATE'])
    penetration.rename(columns={'DATE': 'date'}, inplace=True)

    # Plot our own data, same time frame
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    temp = penetration[(penetration['date'].dt.year >= 1987)
                       & (penetration['date'].dt.year <= 2007)]
    l1 = ax.plot(temp['date'], temp['penetration'], label='China import penetration ratio',
                 color='#000080')
    ax2 = ax.twinx()
    temp = manu[(manu['date'].dt.year >= 1987) & (manu['date'].dt.year <= 2007)]
    l2 = ax2.plot(temp['date'], temp['manu_share'], label='Manufacturing employment/population',
                  color='#800000', linestyle='dashed')
    ax.set_ylabel('Import penetration')
    ax2.set_ylabel('Manufacturing emp./pop.', rotation=270, labelpad=20)
    ax.set_yticks(np.arange(0, 0.006, 0.001))
    ax2.set_yticks(np.arange(0.09, 0.16, 0.015))
    ax2.grid(False)
    ax.set_xticks(pd.date_range('1987-01-01', '2008-01-01', freq='2YS-JUN', inclusive='left'))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    ax.legend(l1 + l2, [i.get_label() for i in l1 + l2], loc='upper center')
    plt.tight_layout()
    fig.savefig('figure/adh2013_own.svg', bbox_inches='tight')

    # Plot only manufacture employment, full sample
    fig, ax = plt.subplots(figsize=(12.6, 5.4))  # 21:9
    ax.plot(manu['date'], manu['manu_share'], label='Manufacturing employment/population',
            color='#800000', linestyle='dashed')
    ax.axvspan(pd.Timestamp('1987-01-01'), pd.Timestamp('2007-12-31'), label='Original sample',
               color='#008000', alpha=0.1, zorder=0)
    ax.set_ylabel('Manufacturing employment/population')
    ax.set_yticks(np.arange(0.07, 0.27, 0.03))
    ax.set_xticks(pd.date_range('1948-01-01', '2024-01-01', freq='5YS-JUN', inclusive='left'))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    ax.legend(loc=(0.01, 0.2))
    plt.tight_layout()
    fig.savefig('figure/adh2013_own_manu.svg', bbox_inches='tight')

    # Plot everything, full sample
    fig, ax = plt.subplots(figsize=(12.6, 5.4))  # 21:9
    l1 = ax.plot(penetration['date'], penetration['penetration'],
                 label='China import penetration ratio', color='#000080')
    ax2 = ax.twinx()
    l2 = ax2.plot(manu['date'], manu['manu_share'], label='Manufacturing employment/population',
                  color='#800000', linestyle='dashed')
    ax.set_ylabel('Import penetration')
    ax2.set_ylabel('Manufacturing emp./pop.', rotation=270, labelpad=20)
    ax.set_yticks(np.arange(0, 0.0065, 0.001))
    ax2.set_yticks(np.arange(0.07, 0.27, 0.03))
    ax2.grid(False)
    ax.set_xticks(pd.date_range('1948-01-01', '2024-01-01', freq='5YS-JUN', inclusive='left'))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    shade = ax.axvspan(pd.Timestamp('1987-01-01'), pd.Timestamp('2007-12-31'),
                       label='Original sample', color='#008000', alpha=0.1, zorder=0)
    ax.legend(l1 + l2 + [shade], [i.get_label() for i in l1 + l2 + [shade]], loc=(0.01, 0.2))
    plt.tight_layout()
    fig.savefig('figure/adh2013_own_full.svg', bbox_inches='tight')
    pass


def main():
    os.makedirs('figure', exist_ok=True)
    buffett()
    adh2013()
    pass


if __name__ == '__main__':
    main()
