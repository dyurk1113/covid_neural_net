import numpy as np
import pandas as pd
import os

CovActNowAPI = '9f8b316c252b481e8b0f73073de0102b'


def get_covactnow_data(fips=None):
    if fips is None:
        url = "https://api.covidactnow.org/v2/counties.timeseries.csv?apiKey=%s" % CovActNowAPI
        data = pd.read_csv(url)
    else:
        url = "https://api.covidactnow.org/v2/county/%05d.timeseries.json?apiKey=%s" % (fips, CovActNowAPI)
        print(url)
        data = pd.read_json(url)
    return data


def get_processed_df(file_path, min_date=None):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df['date_processed'] = pd.to_datetime(df['date'].values)
    df_min_date = df['date_processed'].min() if min_date is None else min_date
    # Convert YYYY-MM-DD date format into integer number of days since the first day in the data set
    df['date_processed'] = (df['date_processed'] - df_min_date) / np.timedelta64(1, 'D')
    return df


def get_COVID_df(min_date=None):
    return get_processed_df(os.path.join('raw_data_files', 'nyt_us_counties.csv'), min_date)


def get_CA_policy_df(min_date=None):
    return get_processed_df(os.path.join('data_files', 'all_CA_policies.csv'), min_date)


def get_CA_rt_df(min_date=None):
    return get_processed_df(os.path.join('data_files', 'all_CA_rt_data.csv'), min_date)


def add_all_demo_vars(df):
    fp = os.path.join('raw_data_files', 'county_populations.csv')
    var_names = ['total_pop', '60plus_pop']
    col_names = ['total_pop', '60plus']
    fill_demo_vars(df, fp, var_names, col_names)

    fp = os.path.join('raw_data_files', 'county_land_areas.csv')
    var_names = ['pop_density', 'housing_density', 'county_name']
    col_names = ['2010 Density per square mile of land area - Population',
                 '2010 Density per square mile of land area - Housing units',
                 'County Name']
    fips_col_name = 'County FIPS'
    fill_demo_vars(df, fp, var_names, col_names, fips_col_name, encoding='windows-1252')


def fill_demo_vars(df, file_path, var_names, col_names, fips_col_name='FIPS', encoding='utf-8'):
    if type(var_names) is str:
        var_names = [var_names]
    if type(col_names) is str:
        col_names = [col_names]
    new_df = pd.read_csv(file_path, encoding=encoding)
    # print('Loading data from', file_path)

    for var_name in var_names:
        # first create empty column to fill in
        pop = np.full(len(df), np.nan)
        df[var_name] = pop

    # loop through every fips and fill in with appropriate population
    unique_fips = df['fips'].unique()

    for fip in unique_fips:
        # check to make sure fips match
        if any(new_df[fips_col_name] == fip):
            for col_name, var_name in zip(col_names, var_names):
                # get corresponding population from population df
                fip_pop = new_df[new_df[fips_col_name] == fip][col_name].values[0]

                # fill in df with population
                df.loc[df.fips == fip, var_name] = fip_pop
        else:
            pass

    return df
