import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
import xgboost as xgb
from sklearn.model_selection import train_test_split
from numpy import random
import os

import torch
import torch.nn as nn
from UMNN.models.UMNN import MonotonicNN

import gen_data_files
import utils


def run_CA_xgboost_fit(pol_excl=[], num_trials=100):
    num_round = 15
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'exact'
    }

    df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))

    policy_cols = ['order_closure', 'order_shome', 'order_masks', 'order_lab', 'order_quarres',
                   'order_quarvis', 'order_bubble']
    for pc in policy_cols:
        print(pc, np.count_nonzero(df['del_' + pc].values != 0),
              np.count_nonzero(df['del_' + pc].values > 0),
              np.count_nonzero(df['del_' + pc].values < 0))

    counties = df['county'].values
    unique_counties = sorted(df['county'].unique())
    del df['county']
    del df['date']
    for pol in pol_excl:
        del df[pol]
        del df['del_' + pol]
    X, y = df.values[:, :-1], df.values[:, -1]

    feature_names = ['%s' % c for c in df.columns][:-1]
    feature_importances = {fn: 0 for fn in feature_names}
    for trial in range(num_trials):
        county_inds = random.choice(np.arange(len(unique_counties)), size=len(unique_counties) // 5, replace=False)
        val_counties = [unique_counties[ind] for ind in county_inds]
        train_inds, test_inds = [i for i, c in enumerate(counties) if c not in val_counties], \
                                [i for i, c in enumerate(counties) if c in val_counties]
        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

        eval_list = [(dtest, 'eval'), (dtrain, 'train')]
        bst = xgb.train(param, dtrain, num_round, eval_list, verbose_eval=False)  # , early_stopping_rounds=3)
        f_scores = bst.get_fscore()
        for fn in feature_names:
            if fn in f_scores:
                feature_importances[fn] += f_scores[fn]
        if trial % 25 == 0:
            print(trial)

    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    print(feature_importances)
    plt.bar(feature_importances.keys(), feature_importances.values())
    plt.tight_layout()
    plt.show()
    return feature_importances


if __name__ == '__main__':
    gen_data_files.build_CA_training_df(include_all_policies=True)
    run_CA_xgboost_fit()
