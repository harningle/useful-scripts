# -*- coding: utf-8 -*-
import datetime
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import requests
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from tqdm import tqdm

from ..ridge_plot.ridge_plot import ridge_plot

gdf = gpd.read_file('data/HaltestellenOeV_LV95.gdb',
                    layer='Betriebspunkt',
                    columns=['Nummer', 'Name', 'geometry'])
CANTONS = gpd.read_file('data/'
                        'historisierte-administrative_grenzen_g1_2025-04-06_kanton_2056.geojson'
)[['KTKZ', 'geometry']].rename(columns={'KTKZ': 'canton'})
assert gdf.crs == CANTONS.crs
gdf = gpd.sjoin(gdf, CANTONS, how='inner', predicate='intersects').to_crs(epsg=4326)
CANTONS.to_crs(epsg=4326).to_file('data/switzerland.geojson')
STOPS = (pl.from_pandas(gdf.drop(columns=['geometry', 'index_right']))
         .with_columns(pl.Series('geometry', gdf.geometry.tolist()))
         .rename({'Nummer': 'stop_id',
                  'Name': 'stop_name'})
         .lazy())
del gdf
plt.style.use('../default.mplstyle')


def preprocess_train_data(path: str | os.PathLike) -> pl.DataFrame:
    date = datetime.datetime.fromisoformat(path.split('/')[-1][:10]).date()
    if date.month <= 6:
        df = pl.scan_csv(path, separator=';', try_parse_dates=True,
                         schema_overrides={'AN_PROGNOSE': pl.Datetime})
    else:
        df = (pl.scan_csv(path, separator=';', try_parse_dates=True,
                          schema_overrides={'BPUIC': str, 'AN_PROGNOSE': pl.Datetime})
              .with_columns(pl.col('BPUIC').str.split(':').list.last().cast(int).alias('BPUIC'))
              .with_columns(pl.when(pl.col('BPUIC') < 100_000)
                            .then(pl.col('BPUIC') + 8_500_000)
                            .otherwise(pl.when(pl.col('BPUIC') >= 100_000_000)
                                       .then((pl.col('BPUIC') / 100).floor().cast(int))
                                       .otherwise(pl.col('BPUIC')))
                            .alias('BPUIC')))
    return (df.filter(pl.col('AN_PROGNOSE_STATUS').is_in(['REAL', 'GESCHAETZT', 'VERSPAETET'])
                      & (pl.col('DURCHFAHRT_TF') == False)  # Exclude pass-throughs
                      & (pl.col('PRODUKT_ID') == 'Zug'))    # Only trains
            .join(STOPS, left_on='BPUIC', right_on='stop_id', how='inner')  # Only in Switzerland
            .rename({'BPUIC': 'stop_id',
                     'LINIEN_TEXT': 'line',
                     'BETREIBER_ABK': 'operator',
                     'ANKUNFTSZEIT': 'scheduled_arrival',
                     'AN_PROGNOSE': 'arrival'})
            .with_columns(pl.lit(date).alias('date'))
            .select(['stop_id', 'date', 'operator', 'line', 'scheduled_arrival', 'arrival'])
            .collect())


def preprocess_early_departure_data(path: str | os.PathLike, stop_id: int) -> pl.DataFrame:
    date = datetime.datetime.fromisoformat(path.split('/')[-1][:10]).date()
    if date.month <= 6:
        df = pl.scan_csv(path, separator=';', try_parse_dates=True,
                         schema_overrides={'AB_PROGNOSE': pl.Datetime})
    else:
        df = (pl.scan_csv(path, separator=';', try_parse_dates=True,
                          schema_overrides={'BPUIC': str, 'AB_PROGNOSE': pl.Datetime})
              .with_columns(pl.col('BPUIC').str.split(':').list.last().cast(int).alias('BPUIC'))
              .with_columns(pl.when(pl.col('BPUIC') < 100_000)
                            .then(pl.col('BPUIC') + 8_500_000)
                            .otherwise(pl.when(pl.col('BPUIC') >= 100_000_000)
                                       .then((pl.col('BPUIC') / 100).floor().cast(int))
                                       .otherwise(pl.col('BPUIC')))
                            .alias('BPUIC')))
    return (df.filter((pl.col('BPUIC') == stop_id)        # Only for specified stop
                      & (pl.col('PRODUKT_ID') == 'Bus'))    # Only bus
            .rename({'BPUIC': 'stop_id',
                     'LINIEN_TEXT': 'line',
                     'BETREIBER_ABK': 'operator',
                     'ABFAHRTSZEIT': 'scheduled_departure',
                     'AB_PROGNOSE': 'departure',
                     'AB_PROGNOSE_STATUS': 'departure_status'})
            .with_columns(pl.lit(date).alias('date'))
            .select(['stop_id', 'date', 'operator', 'line', 'scheduled_departure', 'departure'])
            .collect())


def station_delay_stats():
    os.makedirs('temp', exist_ok=True)
    dfs: list[pd.DataFrame] = []
    max_workers = min(4, os.cpu_count() or 1)
    for month in range(1, 13):
        if not os.path.exists(f'temp/{month}.zip'):
            with open(f'temp/{month}.zip', 'wb') as f:
                if month <= 6:
                    url = f'https://archive.opentransportdata.swiss/istdaten/2025/ist-daten-' \
                          f'2025-{month:02d}.zip'
                elif month >= 8:
                    url = f'https://archive.opentransportdata.swiss/istdaten/2025/ist-daten-v2-' \
                          f'2025-{month:02d}.zip'
                r = requests.get(url, stream=True)
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            if month == 7:  # Have both v1 and v2 data for July
                r = requests.get('https://archive.opentransportdata.swiss/istdaten/2025/ist-'
                                 'daten-2025-07.zip')
                with open(f'temp/7_1.zip', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                r = requests.get('https://archive.opentransportdata.swiss/istdaten/2025/ist-'
                                 'daten-v2-2025-07.zip')
                with open(f'temp/7_2.zip', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        # Has deflate64 in some zipped files, so have to use command line `unzip`
        if month == 7:  # Again, handles v1 and v2 data for July separately
            subprocess.run(['unzip', '-o', 'temp/7_1.zip', '-d', 'temp'], check=True)
            subprocess.run(['unzip', '-o', 'temp/7_2.zip', '-d', 'temp'], check=True)
        else:
            subprocess.run(['unzip', '-o', f'temp/{month}.zip', '-d', 'temp'], check=True)
        file_paths = [f'temp/2025-{month:02d}-{day:02d}_istdaten.csv'
                      for day in range(1, 32)
                      if os.path.exists(f'temp/2025-{month:02d}-{day:02d}_istdaten.csv')]
        if file_paths:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(preprocess_train_data, p): p for p in file_paths}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f'Processing month {month}'):
                    dfs.append(fut.result())
                    os.remove(futures[fut])
        os.remove(f'temp/{month}.zip')
    pl.concat(dfs).write_parquet('data/delay.parquet', statistics=False)
    return


def plot_train_delays():
    # Delay by train firm
    df = (pl.scan_parquet('data/delay.parquet')
          .with_columns((pl.col('arrival') - pl.col('scheduled_arrival'))
                         .dt
                         .total_seconds()
                         .alias('delay'),
                        pl.col('operator').replace({'ASM-bti': 'ASM',
                                                    'ASM-rvo': 'ASM',
                                                    'ASM-snb': 'ASM',
                                                    'AVA-bd': 'AVA',
                                                    'AVA-wsb': 'AVA',
                                                    'MGB-bvz': 'MGB',
                                                    'MGB-fo': 'MGB',
                                                    'MVR-cev': 'MVR',
                                                    'MVR-mtgn': 'MVR',
                                                    'TRAVYS': 'TRAVYS',
                                                    'TRAVYS-pbr': 'TRAVYS',
                                                    'TRN-cmn': 'TRN',
                                                    'TRN-rvt': 'TRN',
                                                    'NeTS-ÖBB': 'ÖBB',
                                                    'OeBB': 'ÖBB',
                                                    'SBB': 'SBB',
                                                    'SBB GmbH': 'SBB',
                                                    'AB-ab': 'AB',
                                                    'BLS-bls': 'BLS',
                                                    'BOB': 'BOB',
                                                    'CJ': 'CJ',
                                                    'D': 'D',
                                                    'DB Regio': 'DB',
                                                    'FART': 'FART',
                                                    'FLP': 'FLP',
                                                    'LEB': 'LEB',
                                                    'MBC': 'MBC',
                                                    'MOB': 'MOB',
                                                    'NStCM': 'NStCM',
                                                    'RA': 'RA',
                                                    'RBS': 'RBS',
                                                    'RhB': 'RhB',
                                                    'SOB-sob': 'SOB',
                                                    'SZU': 'SZU',
                                                    'THURBO': 'THURBO',
                                                    'TMR-mc': 'TMR',
                                                    'TN': 'TN',
                                                    'TPC': 'TPC',
                                                    'TPF': 'TPF',
                                                    'TR': 'TR',
                                                    'VDBB': 'VDBB',
                                                    'ZB': 'ZB'}).alias('operator'))
          .with_columns(pl.when(pl.col('delay') < -120)  # Winsorise extreme delays
                        .then(-120)
                        .otherwise(pl.when(pl.col('delay') > 300)
                                   .then(300)
                                   .otherwise(pl.col('delay')))
                        .alias('delay'))
          .select(['operator', 'delay'])
          .collect()
          .to_pandas())
    densities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    medians: dict[str, float] = {}
    for operator, data in df.groupby('operator'):
        if len(data) < 10000:
            continue
        density = gaussian_kde(data.delay)
        x_eval = np.arange(-120, 301, 15)
        y_eval = density(x_eval)
        densities[operator] = (x_eval, y_eval)
        medians[operator] = data.delay.median()
    medians = dict(sorted(medians.items(), key=lambda item: item[1]))
    densities = {k: densities[k] for k in medians.keys()}
    ridge_plot(x=list(densities.keys()), density=list(densities.values()), means=medians.values(),
               mean_below_label=True)
    plt.xticks([-120, 0, 60, 180, 300],
               ['2 min early', 'On time', '1 min late', '3 min late', '5 min late'])
    plt.tight_layout()
    plt.savefig('train_delay_by_operator.svg', bbox_inches='tight')
    return


def plot_bus_early_departures(stop_id: int):
    # Early departures by bus for a specific stop
    os.makedirs('temp', exist_ok=True)
    dfs: list[pd.DataFrame] = []
    max_workers = min(4, os.cpu_count() or 1)
    for month in range(1, 13):
        # Has deflate64 in some zipped files, so have to use command line `unzip`
        if month == 7:  # Again, handles v1 and v2 data for July separately
            subprocess.run(['unzip', '-o', 'temp/7_1.zip', '-d', 'temp'], check=True)
            subprocess.run(['unzip', '-o', 'temp/7_2.zip', '-d', 'temp'], check=True)
        else:
            subprocess.run(['unzip', '-o', f'temp/{month}.zip', '-d', 'temp'], check=True)
        file_paths = [f'temp/2025-{month:02d}-{day:02d}_istdaten.csv'
                      for day in range(1, 32)
                      if os.path.exists(f'temp/2025-{month:02d}-{day:02d}_istdaten.csv')]
        if file_paths:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(preprocess_early_departure_data, p, stop_id): p
                           for p in file_paths}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f'Processing month {month} for stop {stop_id}'):
                    dfs.append(fut.result())
                    os.remove(futures[fut])
    if dfs:
        pl.concat(dfs).write_parquet(f'data/bus_early_departure_{stop_id}.parquet',
                                     statistics=False)

    # Delay density of line 22 and 3
    df = pd.read_parquet(f'data/bus_early_departure_{stop_id}.parquet')
    df['delay'] = (df.departure - df.scheduled_departure).dt.total_seconds()
    fig, ax = plt.subplots()
    temp = df[df.delay.between(-300, 300)]
    temp[temp.line == '22'].delay.plot(kind='kde', ax=ax, label='22')
    temp[temp.line == '3'].delay.plot(kind='kde', ax=ax, label='3')
    ax.set_xlabel('Departure delay (seconds)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'bus_early_departure.svg', bbox_inches='tight')

    # Share of early departures by line and half-hour, only on weekdays
    early_dep = df[(pd.to_datetime(df.date).dt.dayofweek < 5) & (df.line == '22')].copy()
    early_dep['scheduled_departure'] = early_dep.scheduled_departure.dt.time
    scheduled_departures = [(7, 17), (7, 35), (7, 48), (7, 59), (8, 10), (8, 23), (8, 33), (8, 44),
                            (8, 56), (9, 6), (9, 16), (9, 26), (9, 36), (9, 49), (10, 2), (10, 16),
                            (10, 32), (10, 48), (11, 1), (11, 16), (11, 31), (11, 47), (12, 2),
                            (12, 17), (12, 32), (12, 47), (13, 2), (13, 17), (13, 32), (13, 47),
                            (14, 2), (14, 17), (14, 32), (14, 47)]
    early_dep = early_dep[early_dep.scheduled_departure.isin(
        [datetime.time(h, m) for h, m in scheduled_departures]
    )]
    early_dep['is_early'] = (early_dep.delay < -60)
    early_dep = early_dep.groupby('scheduled_departure').is_early.mean().reset_index()
    early_dep['decimal_hour'] = early_dep.scheduled_departure.apply(
        lambda t: t.hour + t.minute / 60 + t.second / 3600
    )
    # Smooth the line
    x = np.arange(early_dep.decimal_hour.min(), early_dep.decimal_hour.max(), 1 / 60)
    y = CubicSpline(early_dep.decimal_hour, early_dep.is_early * 100)(x)
    fig, ax = plt.subplots()
    ax.plot(early_dep.decimal_hour, early_dep.is_early * 100, 'o', x, y, '-', color='#000080')
    ax.set_xticks(range(7, 16, 1), labels=[f'{h}:00' for h in range(7, 16, 1)])
    ax.set_xlabel('Scheduled departure time')
    ax.set_ylabel('Share of departures more than a minute early (%)')
    plt.tight_layout()
    plt.savefig(f'bus_early_departure_share.svg', bbox_inches='tight')
    return


def main():
    station_delay_stats()
    plot_train_delays()
    plot_bus_early_departures(stop_id=8592852)  # My bus stop
    return


if __name__ == '__main__':
    main()
