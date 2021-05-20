import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from numpy import random
import os

import torch
from UMNN.models.UMNN import MonotonicNN
import shap

import utils


def run_CA_UMNN_fit(cutoff_date='2020-10-01', val_method='date', trial_num=None, output_folder=None):
    # random.seed(92696)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)
    df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))
    input_dim, input_dim_nm = len(df.columns) - 3, len(df.columns) - 4
    hidden_size = 5
    model = MonotonicNN(input_dim, [hidden_size] * 3, dev=device).to(device)
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    dates = df['date'].values
    counties = df['county'].values
    unique_counties = sorted(df['county'].unique())
    del df['county']
    del df['date']
    X, y = df.values[:, :-1], df.values[:, -1]
    X_orig = X.copy()
    X = (X - X.mean(axis=0)) / np.maximum(1e-5, X.std(axis=0))

    # Truly random train/test split
    if val_method == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1113)
    else:
        old_inds, new_inds = dates <= cutoff_date, dates > cutoff_date
        # Split old vs new dates
        if val_method == 'date':
            train_inds, test_inds = old_inds, new_inds

        # Split counties randonly into train or test counties
        if val_method == 'county':
            X, y, counties = X[old_inds], y[old_inds], counties[old_inds]
            county_inds = random.choice(np.arange(len(unique_counties)), size=len(unique_counties) // 5, replace=False)
            val_counties = [unique_counties[ind] for ind in county_inds]
            print(val_counties)
            train_inds, test_inds = [i for i, c in enumerate(counties) if c not in val_counties], \
                                    [i for i, c in enumerate(counties) if c in val_counties]
        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds]

    print('%d Train Points, %d Test Points' % (len(y_train), len(y_test)))
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    b_size = 1

    train_errs, val_errs = [], []
    for epoch in range(20):
        # Shuffle
        idx = torch.randperm(len(X_train))
        train_x = X_train[idx]
        train_y = y_train[idx]
        avg_loss_mon = 0.
        for i in range(0, len(X_train) - b_size, b_size):
            # Monotonic
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model(-1 * x[:, input_dim_nm:], x[:, :input_dim_nm])[:, 0]
            loss = ((y_pred - y) ** 2).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss_mon += loss.item()

        print(epoch)
        y_ep = model(-1 * X_train[:, input_dim_nm:], X_train[:, :input_dim_nm])[:, 0]
        y_val = model(-1 * X_test[:, input_dim_nm:], X_test[:, :input_dim_nm])[:, 0]
        train_loss = ((y_ep - y_train) ** 2).sum()
        train_loss = np.sqrt(train_loss.item() / len(y_train))
        train_errs.append(train_loss)
        print("\tTrain Loss: ", train_loss)
        val_loss = ((y_val - y_test) ** 2).sum()
        val_loss = np.sqrt(val_loss.item() / len(y_test))
        val_errs.append(val_loss)
        print("\tVal Loss: ", val_loss)

    df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))
    pred_order_effects = np.zeros((len(df.values), 21))
    null_col = np.sum(X[:, -1]) == 0
    for trial_ind, trial_data in enumerate(df.values):
        del_order_lvls = np.arange(-10, 11)
        input_data = np.zeros((len(del_order_lvls), input_dim))
        for i in range(len(del_order_lvls)):
            input_data[i, :input_dim_nm] = trial_data[2:input_dim_nm + 2]
            if null_col: #We're suppressing monotonicity by adding a dummy column of 0s in the monotonic slot
                input_data[i, input_dim_nm-1] = del_order_lvls[i]
            else:
                input_data[i, input_dim_nm] = del_order_lvls[i]
        input_data = torch.from_numpy((input_data - X_orig.mean(axis=0)) / np.maximum(1e-5, X_orig.std(axis=0))).float()
        y_ep = model(-1 * input_data[:, input_dim_nm:], input_data[:, :input_dim_nm])[:, 0].detach().cpu().numpy()
        pred_order_effects[trial_ind] = y_ep
    for del_order_lvl in range(-10, 11):
        df['del_order_eff_%d' % del_order_lvl] = pred_order_effects[:, del_order_lvl + 10]
    col = np.zeros(df.values.shape[0])
    col[test_inds] = 1
    df['is_test'] = col

    def f(X):
        Xt = torch.from_numpy(X).float().to(device)
        return model(-1 * Xt[:, input_dim_nm:], Xt[:, :input_dim_nm]).detach().cpu().numpy().flatten()
    explainer = shap.Explainer(f, X[train_inds])
    shap_values = explainer(X[test_inds])

    for i in range(input_dim):
        col = np.zeros(df.values.shape[0])
        col[test_inds] = shap_values.abs.values[:, i]
        df['shap_val_%d' % i] = col

    if output_folder is None:
        output_folder = 'data_files'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if trial_num is not None:
        df.to_csv(os.path.join(output_folder, 'CA_order_eff_preds_%s_%d.csv' % (val_method, trial_num)))
    else:
        df.to_csv(os.path.join(output_folder, 'CA_order_eff_preds_%s.csv' % val_method))

    train_preds = model(-1 * X_train[:, input_dim_nm:], X_train[:, :input_dim_nm])[:, 0].detach().cpu().numpy()
    test_preds = model(-1 * X_test[:, input_dim_nm:], X_test[:, :input_dim_nm])[:, 0].detach().cpu().numpy()

    train_r, train_p = linregress(y_train, train_preds)[2:4]
    print('TRAIN R2', train_r ** 2)
    test_r, test_p = linregress(y_test, test_preds)[2:4]
    print('TEST R2', test_r ** 2)

    return train_r ** 2, test_r ** 2
