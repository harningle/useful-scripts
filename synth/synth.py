# -*- coding: utf-8 -*-
from functools import partial
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Synth:
    def __init__(self, df: pd.DataFrame, unit_id: str, period: str, outcome: str, treated: str,
                 covariates: list[str]):
        self.df = df.copy()
        self.unit_id = unit_id
        self.period = period
        self.outcome = outcome
        self.treated = treated
        self.covariates = covariates
        self.weights: Optional[dict[object, float]] = None  # unit ID --> weight
        self.att: Optional[pd.DataFrame] = None             # [period, observed, synthetic, att]

        # Check if all vars. exist in the data
        assert unit_id in df.columns, f'unit_id {unit_id} not found in df'
        assert df[unit_id].isnull().sum() == 0, f"unit_id {unit_id} shouldn't have missing"
        assert period in df.columns, f'period {period} not found in df'
        assert df[period].isnull().sum() == 0, f"period {period} shouldn't have missing"
        assert outcome in df.columns, f'outcome {outcome} not found in df'
        assert treated in df.columns, f'treated {treated} not found in df'
        if df[treated].isnull().sum() > 0:
            warnings.warn(f"treatment indicator {treated} has missing's. Assuming them as "
                          f"untreated")
            self.df.fillna({treated: 0}, inplace=True)
        for covariate in covariates:
            assert covariate in df.columns, f'covariate {covariate} not found in df'
        self.df.dropna(subset=[outcome] + covariates, how='any', inplace=True)

        # Make sure the panel structure is correct
        assert df[[unit_id, period]].duplicated().sum() == 0, \
            f'unit_id {unit_id} and period {period} cannot uniquely identify rows'

        # Get the unit being treated and treatment year
        assert (temp := set(df['treated'].unique())) == {0, 1}, \
            f'treated {treated} must be binary. Now {treated} takes the values of {temp}'
        assert (df[treated] == 1).any(), 'no unit is treated at any period'
        assert (temp := df[df[treated] == 1][unit_id].nunique()) == 1, \
            f'one and only unit is allowed to be treated. Now have {temp} units treated'
        self.treated_unit = df[df[treated] == 1][unit_id].iloc[0]
        assert df[df[unit_id] == self.treated_unit][treated].min() == 0, \
            f'treated unit cannot be always treated'
        self.treatment_year = df[df[treated] == 1][period].min()

        # Rename unit ID and period to "unit_id" and "year" to shorten the naming
        self.df.rename(columns={unit_id: 'unit_id', period: 'year', outcome: 'outcome'},
                       inplace=True)
        self.covariates = [i if i != self.outcome else 'outcome' for i in covariates]

    @staticmethod
    def loss(w: Iterable[float], x_donor: np.ndarray, x_treated: np.ndarray) -> float:
        return ((x_donor @ w - x_treated) ** 2).sum()

    def get_unit_weight(self):
        # Pre-treatment data for the treated unit and the donor pool
        pre = self.df[self.df['year'] < self.treatment_year][['unit_id', 'year'] + self.covariates]
        pre = pre \
            .set_index(['unit_id', 'year']) \
            .unstack(level=0) \
            .stack(level=0, future_stack=True)
        pre_treated = pre[self.treated_unit]
        pre_donor = pre.drop(columns=self.treated_unit)
        del pre

        # Define quadratic loss and constraints (eq. 7 in Abadie, JEL 2021)
        constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})

        # Minimise the loss to get the weights for each unit in the donor pool
        weights = minimize(
            fun=partial(self.loss, x_donor=pre_donor.values, x_treated=pre_treated.values),
            x0=[1 / pre_donor.shape[1]] * pre_donor.shape[1],
            constraints=constraints,
            bounds=[(0, 1)] * pre_donor.shape[1]
        ).x
        self.weights = dict(zip(pre_donor.columns, weights))

        # Compute pre-treatment ATT/predictor balance
        pre = self.df[self.df['year'] < self.treatment_year][['unit_id', 'year', 'outcome']]
        pre_treated = pre[pre['unit_id'] == self.treated_unit]
        pre_donors = pre[pre['unit_id'] != self.treated_unit]
        del pre
        pre_donors['w'] = pre_donors['unit_id'].map(self.weights)
        pre_donors['synthetic'] = pre_donors['w'] * pre_donors['outcome']
        pre_donors = pre_donors.groupby('year')['synthetic'].sum().reset_index()
        pre_treated = pre_treated.merge(pre_donors[['year', 'synthetic']], on='year')
        pre_treated['att'] = pre_treated['outcome'] - pre_treated['synthetic']
        pre_treated.rename(columns={'outcome': 'observed'}, inplace=True)
        del pre_treated['unit_id']
        self.att = pre_treated

    def fit(self):
        # Match the covariates before treatment
        self.get_unit_weight()

        # Post-treatment data for the treated unit and the donor pool
        post = self.df[self.df['year'] >= self.treatment_year][['unit_id', 'year', 'outcome']]

        # Observed outcomes for the treated unit
        post_treated = post[post['unit_id'] == self.treated_unit]
        post_donor = post[post['unit_id'] != self.treated_unit]
        del post_treated['unit_id'], post
        post_treated.rename(columns={'outcome': 'observed'}, inplace=True)

        # Weighted average of the outcomes for the donor pool
        post_donor['w'] = post_donor['unit_id'].map(self.weights)
        post_donor['synthetic'] = post_donor['w'] * post_donor['outcome']
        post_donor = post_donor.groupby('year')['synthetic'].sum().reset_index()
        post_treated = post_treated.merge(post_donor[['year', 'synthetic']], on='year')
        post_treated['att'] = post_treated['observed'] - post_treated['synthetic']
        self.att = pd.concat([self.att, post_treated], ignore_index=True)
        self.att.rename(columns={'year': self.period}, inplace=True)  # Back to original naming
