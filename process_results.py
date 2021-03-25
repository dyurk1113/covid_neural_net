import pandas as pd
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import os
from scipy.stats import linregress


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
    fig.set_size_inches(12, 5)
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
