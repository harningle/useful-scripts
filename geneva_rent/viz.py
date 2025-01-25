"""
The input data is freely available at IPUMS:
https://international.ipums.org/international-action/sample_details/country/ch#ch2011a

The vars. needed for this scripts are:
SAMPLE              IPUMS sample identifier
SERIAL              Household serial number
HHWT                Household weight
CH2011A_ROOMS       Number of rooms in the household
CH2011A_RENT        Net annual rent paid (Swiss Francs)
CH2011A_WKLOCCANT   Canton of workplace location
CH2011A_WKDPCANTON  Canton of the point of departure to workplace
"""
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CANTON = {
    1: 'Zurich',
    2: 'Bern',
    3: 'Luzern',
    4: 'Uri',
    5: 'Schwyz',
    6: 'Obwalden',
    7: 'Nidwalden',
    8: 'Glarus',
    9: 'Zug',
    10: 'Fribourg',
    11: 'Solothurn',
    12: 'Basel-Stadt',
    13: 'Basel-Landschaft',
    14: 'Schaffhausen',
    15: 'Appenzell Ausserrhoden',
    16: 'Appenzell Innerrhoden',
    17: 'St. Gallen',
    18: 'Graubunden',
    19: 'Aargau',
    20: 'Thurgau',
    21: 'Ticino',
    22: 'Vaud',
    23: 'Valais',
    24: 'Neuchatel',
    25: 'Geneva',
    26: 'Jura',
    97: None,
    98: None,
    99: None
}


def hist(rent: Iterable[float], weights: Iterable[float], bw: int, max_rent: int, visible=False):
    density, bins = np.histogram(rent, bins=range(0, max_rent, bw), density=True, weights=weights)
    # TODO: Is HHWT the correct weight?
    density *= bw  # Equal bin width so this is p.m.f.

    # Just in case of no data, still provide a placebolder hist.
    if len(density):
        pmf = [density[0]]
    else:
        pmf = [0]
    for i in density[1:]:
        pmf.append(pmf[-1] + i)
    pmf.append(1)

    return go.Bar(x=bins, y=density, visible=visible,
                  customdata=[(bins[i], bins[i] + bw, pmf[i]) for i in range(len(bins))],
                  marker_color='#000080')


def main():
    df = pd.read_csv('ipums.csv')
    df.rename(columns={'CH2011A_WKLOCCANT': 'work_canton',
                       'CH2011A_WKDPCANTON': 'departure_canton',
                       'CH2011A_RENT': 'rent',
                       'CH2011A_ROOMS': 'n_rooms'}, inplace=True)
    df = df[(df.rent < 99999.98) & (df.n_rooms < 99)
            & df.work_canton.notnull() & df.departure_canton.notnull()]
    
    # Only people working in a canton and the departure location to office is the same canton
    df.replace({'work_canton': CANTON, 'departure_canton': CANTON}, inplace=True)
    df = df[df.work_canton == df.departure_canton]
    del df['work_canton']
    df.rename(columns={'departure_canton': 'canton'}, inplace=True)
    bw = 250
    max_rent = round(np.ceil(df.rent.max() / bw) * bw)
    
    # Go over all cantons and all possible #. of rooms
    fig = go.Figure()
    buttons = []
    choices = [(i, j) for i in sorted(df.canton.unique()) for j in range(df.n_rooms.max() + 1)]
    geneva_id = -1
    for i, (canton, rooms) in enumerate(choices):
        if rooms == 0:  # All apartments together
            temp = df[df.canton == canton]
        else:           # Only apartments of certain room #.
            temp = df[(df.canton == canton) & (df.n_rooms == rooms)]
        
        if temp.empty:  # If no data then generate a placeholder df.
            temp = pd.DataFrame({'rent': [0], 'HHWT': [1]})
            n = 0
            avg = 0
            med = 0
        else:
            n = len(temp)
            avg = temp.rent.mean()
            med = temp.rent.median()

        # Plot hist. of rent. Default is to show Geneva's rent distri.
        if (canton == 'Geneva') and (rooms == 0):
            geneva_id = i
        fig.add_trace(hist(temp.rent, temp.HHWT, bw=bw, max_rent=max_rent,
                      visible=(geneva_id == i)))

        # Add the canton-room option to the dropdown menu
        match rooms:
            case 0:
                label = f'{canton}, all'
            case 1:
                label = f'{canton}, 1 room'
            case _:
                label = f'{canton}, {rooms} rooms'
        button = dict(
            label=label,
            method='update',
            args=[
                {
                    'visible': [(canton, rooms) == j for j in choices]
                },
                {
                    'title': {
                        'xanchor': 'left',
                        'x': 0.8,
                        'yanchor': 'top',
                        'y': 0.8,
                        'text': rf'$N = {n}\\\\\text{{mean}} = {avg:.1f}\\\\' \
                                rf'\text{{median}} = {med:.1f}$'  # Show basic stat.
                    }
                }
            ]
        )
        buttons.append(button)

    # Hover info.
    fig.update_traces(hovertemplate='Price between %{customdata[0]} and %{customdata[1]}' + \
                                    '<br>Density: %{y:.3f}' + \
                                    '<br>Percentile: %{customdata[2]:.3f}<extra></extra>')

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=geneva_id,
                buttons=buttons,
                direction='down',
                xanchor='left',
                yanchor='bottom',
                x=0,
                y=1.03,
                showactive=True
            )
        ]
    )

    # Default is Geneva. Also change a bit graph styling
    temp = df[df.canton == 'Geneva']
    n = len(temp)
    avg = temp.rent.mean()
    med = temp.rent.median()
    fig.update_layout(
        margin=dict(t=0, b=50, l=70, r=0),
        title={'xanchor': 'left', 'x': 0.8, 'yanchor': 'top', 'y': 0.8,
               'text': rf'$N = {n}\\\\\text{{mean}} = {avg:.1f}\\\\\text{{median}} = {med:.1f}$'},
        xaxis_title='Monthly rent in 2011',
        yaxis_title='Density',
        template='none',
        font={'size': 16},
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgba(255,255,255,0)'
    )
    fig.update_xaxes(zeroline=True, showline=False, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(zeroline=True, showline=False, linewidth=1, linecolor='black', mirror=True)

    fig.write_html('rent_distri.html', include_mathjax='cdn')
    return


if __name__ == '__main__':
    main()
