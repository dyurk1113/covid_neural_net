import pandas as pd
import numpy as np
import os
import utils


def rotate_policy_file(raw_policy_file='PH_Orders_long.csv'):
    policy_df = pd.read_csv(os.path.join('raw_data_files', raw_policy_file))
    counties = sorted(policy_df['county'].unique())
    dates = sorted(policy_df['dt_PHO'].unique())
    with open('raw_data_files/PH_Orders_long.csv') as f:
        all_data = [line.split(',') for line in f.readlines()[1:]]
    out_data = [
        ['county', 'date', 'order_closure', 'order_gath', 'order_lab', 'order_masks', 'order_quarres', 'order_quarvis',
         'order_shome', 'order_bubble']]
    row_inds = {}
    for county in counties:
        for date in dates:
            out_data.append([county, date] + ['NaN'] * (len(out_data[0]) - 2))
            row_inds[(county, date)] = len(out_data) - 1
    for line in all_data:
        county, date, order_val, order_type = line
        out_data[row_inds[(county, date)]][out_data[0].index(order_type.strip())] = order_val
    with open(os.path.join('data_files', 'all_CA_policies.csv'), 'w') as f:
        for line in out_data:
            f.write(','.join(line))
            f.write('\n')


def build_unified_CA_df():
    df = utils.get_COVID_df().dropna(subset=['fips'])
    df = df.loc[[('%05d' % int(fips))[:2] == '06' for fips in df['fips'].values]]
    min_dp = df['date_processed'].min()
    df['date_processed'] = [x - min_dp for x in df['date_processed'].values]
    min_date = pd.to_datetime(df['date'].min())
    utils.add_all_demo_vars(df)
    df['county'] = [x[:-7] for x in df['county_name']]

    policy_df = utils.get_CA_policy_df(min_date=min_date)
    df = df.join(policy_df.set_index(['date_processed', 'county']), on=['date_processed', 'county'], how='left',
                 rsuffix='_pol')

    counties_rt_df = utils.get_CA_rt_df(min_date=min_date)
    df = df.join(counties_rt_df.set_index(['date_processed', 'county']), on=['date_processed', 'county'], how='left',
                 rsuffix='_rt')
    df.drop(axis='columns', labels=['date_rt', 'date_pol'], inplace=True)

    df.to_csv(os.path.join('data_files', 'CA_unified_df.csv'), index=False)


def build_CA_training_df(Reff_lag=21, min_date='2020-05-01', max_date='2020-12-01', include_all_policies=False):
    df = pd.read_csv(os.path.join('data_files', 'CA_unified_df.csv'))
    if include_all_policies:
        policy_cols = ['order_closure', 'order_shome', 'order_masks', 'order_lab', 'order_quarres',
                       'order_quarvis', 'order_bubble']
    else:
        policy_cols = ['order_closure']
    rt_col = 'Reff'
    header_row = ['county', 'date', 'log_popdens', 'log_case_rate', 'prev_rt', 'rt_trend']
    for pc in policy_cols:
        header_row += [pc, 'del_' + pc]
    header_row += ['del_rt']
    all_data = [header_row]
    for county in df['county'].unique():
        county_df = df.loc[df['county'] == county].dropna(subset=[rt_col])  # + policy_cols)
        if len(county_df) == 0:
            print('Invalid Data for', county)
            continue
        dates = county_df['date'].values
        tot_pop = county_df['total_pop'].values[0]
        log_popdens = np.log10(county_df['pop_density'].values[0])
        cases = county_df['cases'].values
        rt = county_df[rt_col].values
        policies = [county_df[pc].values for pc in policy_cols]
        for i in range(0, len(dates), 7):
            if dates[i] < min_date or dates[i] > max_date or i < Reff_lag + 7:
                continue
            if cases[i] - cases[i - 7] < 10:
                continue
            del_rt = (np.median(rt[i:i + 14]) - np.median(rt[i - 14:i]))
            log_case_rate = np.log10((cases[i] - cases[i - 14]) / tot_pop)
            new_row = [county, dates[i], '%.3f' % log_popdens, '%.5f' % log_case_rate,
                       '%.5f' % rt[i - Reff_lag],
                       '%.5f' % (rt[i - Reff_lag] - rt[i - Reff_lag - 7])]
            for pc in policies:
                new_row += ['%d' % pc[i], '%d' % int(pc[i] - pc[i - 7])]
            new_row += ['%.4f' % del_rt]
            all_data.append(new_row)

    with open(os.path.join('data_files', 'CA_training_df.csv'), 'w') as f:
        for row in all_data:
            f.write(','.join(row) + '\n')
