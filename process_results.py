import pandas as pd
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import os
from scipy.stats import linregress
import shap

import gen_data_files


def plot_fig1():
    fig, ax = plt.subplots()
    file = 'data_files\\all_CA_policies.csv'
    data = pd.read_csv(file)
    dates, pol_vals = data['date'].values, data['order_closure'].values
    monthly_pols = []
    rt_file = 'raw_data_files\\State_Rt_vals.csv'
    rt_data = pd.read_csv(rt_file)
    rt_data = rt_data.loc[rt_data['region'] == 'CA']
    rt_dates, rt_vals = rt_data['date'].values, rt_data['median'].values
    monthly_rt_vals = []
    for month in range(5, 13):
        month_inds = [ind for ind, val in enumerate(dates) if
                      ('2020-%02d-01' % month) <= val <= ('2020-%02d-31' % month)]
        monthly_pols.append([pol_vals[ind] for ind in month_inds])
        month_inds = [ind for ind, val in enumerate(rt_dates) if
                      ('2020-%02d-01' % month) <= val <= ('2020-%02d-31' % month)]
        rt = [rt_vals[ind] for ind in month_inds]
        monthly_rt_vals.append(np.mean(rt))

    month_labs = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.boxplot(monthly_pols, labels=month_labs,
               medianprops={'color': 'black', 'lw': 3})
    ax.set_xlabel('Month')
    ax.set_ylabel('order_closure Values')
    ax.set_title('Distribution of 2020 CA County Closure Orders and Rt Values by Month')

    ax2 = ax.twinx()
    ax2.set_ylabel('State Average Rt Value')
    ax2.scatter([], [], facecolors='white', edgecolors='black', label='Policy Outliers', s=50, marker='o')
    ax2.scatter(['']+month_labs, [-5]+monthly_rt_vals, c='black', marker='D', label='Rt Values')
    ax2.set_ylim(0.85, 1.225)
    ax2.set_xlim(0.5, 8.5)
    ax2.legend(loc=1)

    plt.savefig('analysis_plots\\fig_1_bw.tif', bbox_inches='tight')


def get_avg_shaps(folder, tag, only_pol_change=0):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if ('.csv' in f and tag in f)]
    all_dfs = [pd.read_csv(f) for f in files]
    all_data = []
    for df in all_dfs:
        for row in df.values:
            if only_pol_change == 1 and row[8] == 0:
                continue
            if only_pol_change == -1 and row[8] != 0:
                continue
            if row[-7] == 1.0:
                all_data.append(row[-6:])
    res = np.mean(all_data, axis=0)
    res /= np.sum(res)/100
    print(['%.1f' % x for x in res], (res[-2] + res[-1]) / np.sum(res))


def combine_pred_files(folder, tag):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if ('.csv' in f and tag in f)]
    combo_df = pd.read_csv(files[0]).copy()
    dfs = [pd.read_csv(f) for f in files]
    for del_order in range(-10, 11):
        col_name, sd_name = 'del_order_eff_%d' % del_order, 'del_order_eff_sd_%d' % del_order
        raw_vals = [df[col_name].values for df in dfs]
        combo_df[col_name] = np.mean(raw_vals, axis=0)
        combo_df[sd_name] = np.std(raw_vals, axis=0)
    for shap_ind in range(6):
        col_name = 'shap_val_%d' % shap_ind
        if col_name in list(combo_df.columns):
            raw_vals = [df[col_name].values for df in dfs]
            combo_df[col_name] = np.mean(raw_vals, axis=0)
    file_out = files[0][:-6].split('\\')[-1] + '.csv'
    del combo_df['Unnamed: 0']
    combo_df.to_csv('data_files\\%s' % file_out)


def get_avg_corr(folder, tag, max_date='2020-10-01', use_cases=False, no_mon=False, combine=True):
    gen_data_files.rotate_policy_file()
    gen_data_files.build_unified_CA_df()
    gen_data_files.build_CA_training_df(max_date=max_date, use_cases=use_cases, no_mon_col=no_mon)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if ('.csv' in f and tag in f)]
    train_r2, test_r2, train_rmse, test_rmse = [], [], [], []

    train_df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))
    all_train_pred, train_true, all_test_pred, test_true = [], None, [], None

    # tot, bad, real_bad = 0, 0, 0
    for f in files:
        df = pd.read_csv(f)
        df = df.loc[df['date'] <= max_date]
        train_inds = df['is_test'] == 0
        test_inds = df['is_test'] == 1
        y_train, y_test = train_df.values[train_inds, -1].astype(np.float), train_df.values[test_inds, -1].astype(
            np.float)
        if test_true is None:
            train_true = y_train
            test_true = y_test
        del_pol_col = -3 if no_mon else -2
        real_del_train, real_del_test = train_df.values[train_inds, del_pol_col].astype(np.int), train_df.values[
            test_inds, del_pol_col].astype(np.int)
        all_train_outputs = df.values[train_inds, 10:]
        outputs = np.array([output_row[rd + 10] for output_row, rd in zip(all_train_outputs, real_del_train)])
        all_train_pred.append(outputs)
        if not combine:
            r2 = linregress(y_train, outputs)[2] ** 2
            rmse = np.sqrt(np.mean(np.square(y_train - outputs)))
            train_r2.append(r2)
            train_rmse.append(rmse)

        # if f == files[0]:
        #     styles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1))]
        #     nomon_ind = 0
        #     for row_ind in [22, 162, 202, 582]:
        #         row = df.values[row_ind]
        #         tot += 1
        #         real_oc, real_del = row[7], row[8]
        #         plt.plot(np.arange(-1*real_oc, 10 - real_oc), row[20-real_oc:30-real_oc], c='black', label='%s County\n%s' % (row[1], row[2]), ls=styles[nomon_ind%len(styles)])
        #         nomon_ind += 1
        #     plt.xlabel('Change In order_closure')
        #     plt.ylabel('Projected del_Rt')
        #     plt.title('Sample of Unconstrained Model Outputs')
        #     plt.legend()
        #     plt.savefig('analysis_plots\\fig_s1_bw.tiff')
        #     plt.clf()

        all_test_outputs = df.values[test_inds, 10:]
        outputs = np.array([output_row[rd + 10] for output_row, rd in zip(all_test_outputs, real_del_test)])
        all_test_pred.append(outputs)
        if not combine:
            r2 = linregress(y_test, outputs)[2] ** 2
            rmse = np.sqrt(np.mean(np.square(y_test - outputs)))
            test_r2.append(r2)
            test_rmse.append(rmse)
    # print(bad, tot, real_bad, 100*bad/tot, 100*real_bad/bad, 100*real_bad/tot)
    if combine:
        train_pred = np.median(all_train_pred, axis=0)
        test_pred = np.median(all_test_pred, axis=0)
        ret = (linregress(train_true, train_pred)[2] ** 2, np.sqrt(np.mean(np.square(train_true - train_pred))), \
               linregress(test_true, test_pred)[2] ** 2, np.sqrt(np.mean(np.square(test_true - test_pred))))
        fig, axs = plt.subplots(ncols=2, sharey=True)
        lim = [-0.35, 0.3]
        axs[0].plot(lim, lim, ls='--', c='black', lw=3, label='Ideal Match')
        axs[0].scatter(train_true, train_pred, label='Training Data (r^2 = %.3f)' % ret[0], c='black')
        axs[0].set_title('Training on May-Sep 2020')
        axs[0].set_xlabel('True del_Rt')
        axs[0].set_ylabel('Fitted del_Rt')
        axs[0].legend(loc=2, framealpha=1)
        axs[0].set_xlim(lim)
        axs[0].set_ylim(lim)
        axs[0].set_aspect('equal')
        axs[1].plot(lim, lim, ls='--', c='black', lw=3, label='Ideal Match')
        axs[1].scatter(test_true, test_pred, label='Test Data (r^2 = %.3f)' % ret[2], c='black')
        axs[1].set_title('Testing on Oct-Dec 2020')
        axs[1].set_xlabel('True del_Rt')
        axs[1].legend(loc=2, framealpha=1)
        axs[1].set_xlim(lim)
        axs[1].set_ylim(lim)
        axs[1].set_aspect('equal')
        fig.set_size_inches(10, 5)
        fig.tight_layout()
        plt.savefig('analysis_plots\\fig_2_bw_raw.tif')
        return ret
    return np.mean(train_r2), np.mean(train_rmse), np.mean(test_r2), np.mean(test_rmse)


def categorize_counties():
    df = pd.read_csv('data_files\\CA_order_eff_preds_date.csv')
    df_trim = df.loc[df['date'] < '2020-10-01']
    dat_vals = np.zeros((3, 3))
    for row in df_trim.values:
        real_val = row[9]
        pred_val = row[20 + row[8]]
        dat_row = 0 if real_val < -0.05 else (1 if real_val < 0.05 else 2)
        dat_col = 0 if pred_val < -0.05 else (1 if pred_val < 0.05 else 2)
        dat_vals[dat_row, dat_col] += 1
    badly_wrong = (dat_vals[2, 0] + dat_vals[0, 2]) / np.sum(dat_vals)
    print('Training:', badly_wrong, 'badly wrong')
    for row in dat_vals:
        print(row)

    df_trim = df.loc[df['date'] >= '2020-10-01']
    dat_vals = np.zeros((3, 3))
    for row in df_trim.values:
        real_val = row[9]
        pred_val = row[20 + row[8]]
        dat_row = 0 if real_val < -0.05 else (1 if real_val < 0.05 else 2)
        dat_col = 0 if pred_val < -0.05 else (1 if pred_val < 0.05 else 2)
        dat_vals[dat_row, dat_col] += 1
    badly_wrong = (dat_vals[2, 0] + dat_vals[0, 2]) / np.sum(dat_vals)
    print('Testing:', badly_wrong, 'badly wrong')
    for row in dat_vals:
        print(row)


def county_group_performance(county_list=None, plot_axs=None, ls=None, label=None):
    df = pd.read_csv('data_files\\CA_order_eff_preds_date.csv')
    if county_list is not None:
        df = df[df['county'].isin(county_list)]
    monthly_perfs = []
    for month in range(5, 13):
        perf, rts = np.zeros(3), []
        df_trim = df.values[[('-%02d-' % month) in date for date in df['date'].values]]
        for row in df_trim:
            real_val = row[9]
            pred_val = row[20 + row[8]]
            dat_row = 0 if real_val < -0.05 else (1 if real_val < 0.05 else 2)
            dat_col = 0 if pred_val < -0.05 else (1 if pred_val < 0.05 else 2)
            if dat_col < dat_row:
                perf[0] += 1
            elif dat_col == dat_row:
                perf[1] += 1
            else:
                perf[2] += 1
        perf /= np.sum(perf)
        monthly_perfs.append(perf)
    monthly_perfs = np.array(monthly_perfs)
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if plot_axs is not None:
        plot_axs[0].plot(months, monthly_perfs[:, 1], c='black', ls=ls, label=label)
        plot_axs[1].plot(months, monthly_perfs[:, 0], c='black', ls=ls, label=label)
        plot_axs[2].plot(months, monthly_perfs[:, 2], c='black', ls=ls, label=label)
    return monthly_perfs


def plot_fig3():
    styles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1))]
    fig, axs = plt.subplots(ncols=3, sharey=True)
    county_group_performance(plot_axs=axs, ls=styles[0], label='All Counties')
    all_df = pd.read_csv('data_files\\CA_unified_df.csv')
    counties = all_df['county'].unique()
    county_pops = {county: all_df.loc[all_df['county'] == county]['total_pop'].values[0] for county in counties}
    cutoff1 = np.percentile(list(county_pops.values()), 33)
    cutoff2 = np.percentile(list(county_pops.values()), 66)
    small_counties = [county for county in county_pops if county_pops[county] < cutoff1]
    med_counties = [county for county in county_pops if cutoff1 <= county_pops[county] < cutoff2]
    big_counties = [county for county in county_pops if county_pops[county] >= cutoff2]
    county_group_performance(county_list=small_counties, plot_axs=axs, ls=styles[1], label='Small Counties')
    county_group_performance(county_list=big_counties, plot_axs=axs, ls=styles[2], label='Big Counties')
    county_group_performance(county_list=med_counties, plot_axs=axs, ls=styles[3], label='Medium Counties')

    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Fraction of All Predictions')
    axs[0].set_title('Correct Prediction Rate')
    axs[0].legend(loc=8)
    axs[1].set_title('Underprediction Rate')
    axs[1].legend(loc=9)
    axs[2].set_title('Overprediction Rate')
    axs[2].legend(loc=9)

    fig.tight_layout()
    fig.set_size_inches(10, 4)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig('analysis_plots/fig_3_bw_raw.tiff', bbox_inches='tight')
    plt.clf()


def plot_fig4():
    df = pd.read_csv('data_files\\CA_order_eff_preds_date.csv')
    styles = ['solid', 'dotted', 'dashed']
    counties = ['Fresno', 'Placer', 'Mono']
    oc_vals = [3, 2, 2]
    fig, axs = plt.subplots(nrows=1, ncols=2)
    df_trim = df.loc[df['date'] >= '2020-10-15']
    for county in df_trim['county'].unique():
        df_trim2 = df_trim.loc[df_trim['county'] == county]
        if df_trim2.values[0, 15] < 0:
            print(county)
    for i, county in enumerate(counties):
        df_trim2 = df_trim.loc[df_trim['county'] == county]
        order_lvl = df_trim2.values[0, 7]
        del_orders = np.arange(-1 * order_lvl, 10 - order_lvl)
        inds = del_orders + 10
        order_effs = df_trim2.values[0, 10:31]
        order_sds = df_trim2.values[0, 38:60]
        axs[0].plot(del_orders, order_effs[inds], label=county+(' (order_closure=%d)'%oc_vals[i]), c='black', ls=styles[i])
    axs[0].legend()
    for i, county in enumerate(counties):
        df_trim2 = df_trim.loc[df_trim['county'] == county]
        order_lvl = df_trim2.values[0, 7]
        del_orders = np.arange(-1 * order_lvl, 10 - order_lvl)
        inds = del_orders + 10
        order_effs = df_trim2.values[0, 10:31]
        order_sds = df_trim2.values[0, 38:60]
        axs[0].plot(del_orders, order_effs[inds] - order_sds[inds], c='black', ls=styles[i], lw=0.5)
        axs[0].plot(del_orders, order_effs[inds] + order_sds[inds], c='black', ls=styles[i], lw=0.5)
    axs[0].set_xlabel('Change In order_closure')
    axs[0].set_ylabel('Projected del_Rt')
    axs[0].set_title('Predicted order_closure Policy Effects: Oct 15')
    axs[0].set_xlim(-3, 7)

    rt_df = pd.read_csv(os.path.join('data_files', 'all_CA_rt_data.csv'))
    rt_df = rt_df.loc[rt_df['date'] >= '2020-10-01']
    rt_df = rt_df.loc[rt_df['date'] < '2020-11-01']
    for i, county in enumerate(counties):
        rt_df_trim = rt_df.loc[rt_df['county'] == county]
        axs[1].plot(np.arange(1, 32), rt_df_trim['Reff'].values, label=county, c='black', ls=styles[i])
        axs[1].scatter([15], [rt_df_trim['Reff'].values[14]], c='black')
    axs[1].set_xlabel('Day in October')
    axs[1].set_ylabel('True Rt')
    axs[1].set_title('True Rt Evolution in October')
    axs[1].set_xlim(1, 31)
    fig.set_size_inches(10, 4)
    fig.tight_layout()
    fig.savefig('analysis_plots/fig_4_bw_raw.tiff', bbox_inches='tight')
    plt.clf()


def analyze_pol_eff_preds():
    df = pd.read_csv(os.path.join('data_files', 'CA_order_eff_preds.csv'))
    del df[df.columns[0]]
    x, y = df['prev_rt'].values, df['del_order_eff_0'].values
    df_trim = df.loc[df['del_order_closure'] == 0]
    x2, y2 = df_trim['prev_rt'].values, df_trim['del_rt'].values
    plt.scatter(x2, y2, c='b', label='Real Values', alpha=0.2)
    plt.scatter(x, y, c='r', label='Projected Values', alpha=0.2)
    plt.xlabel('Rt Value 3 Weeks Ago')
    plt.ylabel('Projected Rt Change Under Status Quo')
    plt.title('Status Quo Effect vs. Rt Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_plots/statusquo_v_rt.png', bbox_inches='tight')
    plt.clf()

    x, x2 = df['rt_trend'].values, df_trim['rt_trend'].values
    plt.scatter(x2, y2, c='b', label='Real Values', alpha=0.2)
    plt.scatter(x, y, c='r', label='Projected Values', alpha=0.2)
    plt.xlabel('Rt Trend 3 Weeks Ago')
    plt.ylabel('Projected Rt Change Under Status Quo')
    plt.title('Status Quo Effect vs. Rt Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_plots/statusquo_v_rt_trend.png', bbox_inches='tight')
    plt.clf()

    fig, axs = plt.subplots(nrows=1, ncols=1)
    styles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1))]
    vals = df.values
    for i, ind in enumerate(random.choice(np.arange(len(vals)), 5, replace=False)):
        plt.plot(np.arange(-10, 11), vals[ind, 9:], label='%s, %s' % (vals[ind, 0], vals[ind, 1]), c='black',
                 ls=styles[i])
    axs.set_xlabel('Change In order_closure')
    axs.set_ylabel('Projected del_Rt')
    axs.set_title('County-Specific Policy Impact Projections')
    axs.legend(framealpha=1)
    axs.set_xlim(-10, 10)
    fig.tight_layout()
    fig.savefig('analysis_plots/fig_2_bw.tiff', bbox_inches='tight')
    fig.clf()

    counties = ['Fresno', 'Placer', 'Mono']
    fig, axs = plt.subplots(nrows=1, ncols=2)
    df_trim = df.loc[df['date'] >= '2020-10-15']
    for i, county in enumerate(counties):
        df_trim2 = df_trim.loc[df_trim['county'] == county]
        order_effs = df_trim2.values[0, 9:]
        axs[0].plot(np.arange(-10, 11), order_effs, label=county, c='black', ls=styles[i])
    axs[0].set_xlabel('Change In order_closure')
    axs[0].set_ylabel('Projected del_Rt')

    rt_df = pd.read_csv(os.path.join('data_files', 'all_CA_rt_data.csv'))
    rt_df = rt_df.loc[rt_df['date'] >= '2020-10-01']
    rt_df = rt_df.loc[rt_df['date'] < '2020-11-01']
    for i, county in enumerate(counties):
        rt_df_trim = rt_df.loc[rt_df['county'] == county]
        axs[1].plot(np.arange(1, 32), rt_df_trim['Reff'].values, label=county, c='black', ls=styles[i])
    axs[1].set_xlabel('Day in October')
    axs[1].set_ylabel('True Rt')
    axs[1].set_title('True Rt Evolution in October')
    axs[1].set_xlim(1, 31)
    axs[1].legend()
    fig.set_size_inches(10, 4)
    fig.tight_layout()
    fig.savefig('analysis_plots/fig_3_bw.tiff', bbox_inches='tight')
    plt.clf()

    for col_name in ['log_popdens', 'log_case_rate', 'prev_rt', 'rt_trend', 'order_closure']:
        df_trim = df.loc[df['order_closure'] > 0]
        x, y = df_trim[col_name].values, df_trim['del_order_eff_-1'].values - df_trim['del_order_eff_0'].values
        plt.scatter(x, y, c='#00FFFF', label='-1 Order Decrease')
        m, b, r, p = linregress(x, y)[:4]
        print(col_name)
        print('\tOrder Decrease r2, p = %.3f, %.3f' % (r ** 2, p))
        plt.plot([x.min(), x.max()], [m * x.min() + b, m * x.max() + b], c='b', ls='--', lw=3,
                 label=('Best Fit, r^2=%.3f' % r ** 2))

        df_trim = df.loc[df['order_closure'] < 9]
        x, y = df_trim[col_name].values, df_trim['del_order_eff_1'].values - df_trim['del_order_eff_0'].values
        plt.scatter(x, y, c='#00FF00', label='+1 Order Increase')
        m, b, r, p = linregress(x, y)[:4]
        print('\tOrder Increase r2, p = %.3f, %.3f' % (r ** 2, p))
        plt.plot([x.min(), x.max()], [m * x.min() + b, m * x.max() + b], c='g', ls='--', lw=3,
                 label=('Best Fit, r^2=%.3f' % r ** 2))

        plt.xlabel(col_name)
        plt.ylabel('Projected Policy Impact')
        plt.title('Projected Policy Impact vs. %s' % col_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig('analysis_plots/pol_eff_v_%s.png' % col_name, bbox_inches='tight')
        plt.clf()

def do_full_processing():
    combine_pred_files('trial_data_files', 'date')
    plot_fig1()
    print('Full Model Rt No Mon (date):\n\tTrain: %.3f\tRMSE: %.4f\n\tTest: %.5f\tRMSE: %.4f' % get_avg_corr(
        'trial_data_files_no_mon', 'date', max_date='2020-12-31', combine=True))
    print('Full Model Rt (date):\n\tTrain: %.3f\tRMSE: %.4f\n\tTest: %.5f\tRMSE: %.4f' % get_avg_corr(
        'trial_data_files', 'date', max_date='2020-12-31', combine=True))
    plot_fig3()
    categorize_counties()
    plot_fig4()

    get_avg_shaps('trial_data_files_delrt_no_caserate', 'date')
    get_avg_shaps('trial_data_files_delrt_no_caserate', 'date', only_pol_change=1)
    get_avg_shaps('trial_data_files_delrt_no_caserate', 'date', only_pol_change=-1)


if __name__ == '__main__':
    do_full_processing()