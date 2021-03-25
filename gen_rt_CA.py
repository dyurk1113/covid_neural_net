import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_context('talk')
from matplotlib import pyplot as plt
from rt_live.covid.models.generative import GenerativeModel
from rt_live.covid.data import summarize_inference_data, get_raw_covidtracking_data
import utils

idx = pd.IndexSlice


def run_rt_gen():
    state_df = get_raw_covidtracking_data()
    state_df = state_df.loc[state_df['state'] == 'CA'].dropna(axis=1, thresh=100)
    state_df["date"] = pd.to_datetime(state_df["date"], format="%Y%m%d")

    all_data = [['county', 'date', 'Reff']]

    covan_df = utils.get_covactnow_data()
    covan_df['date'] = pd.to_datetime(covan_df['date'], format="%Y-%m-%d")

    df = utils.get_COVID_df(os.path.join('raw_data_files', 'nyt_us_counties.csv'))
    df = df.loc[df['state'] == 'California']
    counties = df['county'].unique()
    print(counties)
    for county in counties:
        if county == 'Unknown':
            continue
        print(county)
        county_df = df.loc[df['county'] == county]
        covan_county_df = covan_df.loc[covan_df.fips == county_df['fips'].values[0]]
        county_df = county_df.join(covan_county_df.set_index('date'), on='date', rsuffix='_CAN')
        county_df = county_df.join(state_df.set_index('date'), on='date', rsuffix='_STATE')
        county_df['positive'] = np.maximum(0, np.concatenate(
            ([county_df['cases'].values[0]], county_df['cases'].values[1:] - county_df['cases'].values[:-1])))
        daily_negs = np.concatenate(([county_df['actuals.negativeTests'].values[0]],
                                     county_df['actuals.negativeTests'].values[1:] -
                                     county_df['actuals.negativeTests'].values[:-1]))
        state_neg_rat = county_df['negativeIncrease'].values / np.maximum(1, county_df['positiveIncrease'].values)
        neg_fill_in = np.maximum(1, county_df['positive'].values) * state_neg_rat
        daily_negs[np.isnan(daily_negs)] = neg_fill_in[np.isnan(daily_negs)]
        county_df['total'] = county_df['positive'].values + daily_negs
        county_df = county_df[['date', 'county', 'fips', 'positive', 'total']].dropna()
        county_df = county_df.set_index(['date']).sort_index()

        print('Starting Gen Model')
        gm = GenerativeModel(county, county_df)
        gm.sample()
        result = summarize_inference_data(gm.inference_data)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set(title=f"Effective reproduciton number for {county}", ylabel="$R_e(t)$")
        samples = gm.trace["r_t"]
        cmap = plt.get_cmap("Reds")
        percs = np.linspace(51, 99, 40)
        colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
        samples = samples.T
        x = result.index
        result["median"].plot(c="k", ls='-')

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(samples, p, axis=1)
            lower = np.percentile(samples, 100 - p, axis=1)
            color_val = colors[i]
            ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=.8)

        ax.axhline(1.0, c="k", lw=1, linestyle="--")
        sns.despine()
        plt.savefig(os.path.join('CA_rt_plots', '%s_rtlive.png' % county))

        y = np.percentile(samples, 50, axis=1)
        for xx, yy in zip(x, y):
            all_data.append([county, '%s' % xx, '%.4f' % yy])

        with open(os.path.join('data_files', 'all_CA_rt_data.csv'), 'w') as f:
            for row in all_data:
                f.write(','.join(row) + '\n')


if __name__ == '__main__':
    run_rt_gen()
