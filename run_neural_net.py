import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from numpy import random
import os

import torch
from UMNN.models.UMNN import MonotonicNN

import utils


def run_CA_UMNN_fit(cutoff_date='2020-10-01', pol_excl=[], ax=None, val_method='date', ylab=True):
    random.seed(92696)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))
    input_dim, input_dim_nm = len(df.columns) - 3, len(df.columns) - 4
    # 3 Layers of 6:
    # Train Loss: 0.04029203885209558
    # Val Loss: 0.04454779332063541
    # 3 Layers of 3:
    # Train Loss: 0.042967358345156456
    # Val Loss: 0.045554016299142006
    # 5 Layers of 3: worse
    # 5 Layers of 6:
    # Train Loss: 0.04070521083704866
    # Val Loss: 0.045827694671886354
    # 4 Layers of 4: worse
    # Weigth Decay to 1e-8:
    # Train Loss: 0.03926069275824618
    # Val Loss: 0.050774661883905205
    # Weigth Decay to 1e-4:
    # Train Loss: 0.040581790578028305
    # Val Loss: 0.04773834022791609
    model_monotonic = MonotonicNN(input_dim, [6] * 3, dev=device).to(device)
    optim_monotonic = torch.optim.Adam(model_monotonic.parameters(), 1e-3, weight_decay=1e-5)

    dates = df['date'].values
    counties = df['county'].values
    unique_counties = sorted(df['county'].unique())
    del df['county']
    del df['date']
    for pol in pol_excl:
        del df[pol]
        del df['del_' + pol]
    X, y = df.values[:, :-1], df.values[:, -1]
    X_orig = X.copy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Truly random train/test split
    if val_method == 'random':
        plt_title = 'Validating on Randomly Selected Data Points'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1113)

    # Split old vs new dates
    if val_method == 'date':
        plt_title = 'Testing on Data Since %s' % cutoff_date
        old_inds, new_inds = dates <= cutoff_date, dates > cutoff_date
        X_train, X_test, y_train, y_test = X[old_inds], X[new_inds], y[old_inds], y[new_inds]

    # Split counties randonly into train or test counties
    if val_method == 'county':
        plt_title = 'Validating on Randomly Selected Counties'
        county_inds = random.choice(np.arange(len(unique_counties)), size=len(unique_counties) // 5, replace=False)
        val_counties = [unique_counties[ind] for ind in county_inds]
        print(val_counties)
        train_inds, test_inds = [i for i, c in enumerate(counties) if c not in val_counties], \
                                [i for i, c in enumerate(counties) if c in val_counties]
        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds]

    print('%d Train Points, %d Test Points' % (len(y_train), len(y_test)))
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    b_size = 1

    train_errs, val_errs = [], []
    for epoch in range(20):
        # Shuffle
        idx = torch.randperm(len(X_train))
        train_x = X_train[idx].to(device)
        train_y = y_train[idx].to(device)
        avg_loss_mon = 0.
        for i in range(0, len(X_train) - b_size, b_size):
            # Monotonic
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model_monotonic(-1 * x[:, input_dim_nm:], x[:, :input_dim_nm])[:, 0]
            loss = ((y_pred - y) ** 2).sum()
            optim_monotonic.zero_grad()
            loss.backward()
            optim_monotonic.step()
            avg_loss_mon += loss.item()

        print(epoch)
        y_ep = model_monotonic(-1 * X_train[:, input_dim_nm:], X_train[:, :input_dim_nm])[:, 0]
        train_loss = ((y_ep - y_train) ** 2).sum()
        train_loss = np.sqrt(train_loss.item() / len(y_train))
        train_errs.append(train_loss)
        print("\tTrain Loss: ", train_loss)
        y_val = model_monotonic(-1 * X_test[:, input_dim_nm:], X_test[:, :input_dim_nm])[:, 0]
        val_loss = ((y_val - y_test) ** 2).sum()
        val_loss = np.sqrt(val_loss.item() / len(y_test))
        val_errs.append(val_loss)
        print("\tVal Loss: ", val_loss)

    df = pd.read_csv(os.path.join('data_files', 'CA_training_df.csv'))
    pred_order_effects = np.zeros((len(df.values), 21))
    for trial_ind, trial_data in enumerate(df.values):
        del_order_lvls = np.arange(-10, 11)
        input_data = np.zeros((len(del_order_lvls), input_dim))
        for i in range(len(del_order_lvls)):
            input_data[i, :input_dim_nm] = trial_data[2:input_dim_nm + 2]
            input_data[i, input_dim_nm] = del_order_lvls[i]
        input_data = torch.from_numpy((input_data - X_orig.mean(axis=0)) / X_orig.std(axis=0)).float()
        y_ep = model_monotonic(-1 * input_data[:, input_dim_nm:], input_data[:, :input_dim_nm])[:,
               0].detach().cpu().numpy()
        pred_order_effects[trial_ind] = y_ep
    for del_order_lvl in range(-10, 11):
        df['del_order_eff_%d' % del_order_lvl] = pred_order_effects[:, del_order_lvl + 10]
    df.to_csv(os.path.join('data_files', 'CA_order_eff_preds.csv'))

    if ax is None:
        ax = plt.gca()

    train_preds = model_monotonic(-1 * X_train[:, input_dim_nm:], X_train[:, :input_dim_nm])[:,
                  0].detach().cpu().numpy()
    test_preds = model_monotonic(-1 * X_test[:, input_dim_nm:], X_test[:, :input_dim_nm])[:, 0].detach().cpu().numpy()

    train_r, train_p = linregress(y_train, train_preds)[2:4]
    print(train_r ** 2, train_p)
    # ax.scatter(y_train, train_preds, s=40, facecolors='none', edgecolors='black',# c='black', alpha=0.1,
    #             label='Training Data (r^2 = %.3f)' % (train_r ** 2))
    test_r, test_p = linregress(y_test, test_preds)[2:4]
    print(test_r ** 2, test_p)
    ax.set_title(plt_title)
    lim = [-0.35, 0.3]
    ax.plot(lim, lim, ls='--', c='black', lw=3, label='Ideal Match')
    if val_method == 'county':
        ax.scatter(y_test, test_preds, c='black', label='Validation Data (r^2 = %.3f)' % (test_r ** 2))
        # marker='2')
        # ax.legend(loc=2, labels=['Ideal Match', 'Training Data (r^2 = 0.766)', 'Validation Data (r^2 = 0.725)'], framealpha=1)
        ax.legend(loc=2, labels=['Ideal Match', 'Validation Data (r^2 = 0.725)'], framealpha=1)
    else:
        ax.scatter(y_test, test_preds, c='black', label='Testing Data (r^2 = %.3f)' % (test_r ** 2))
        # marker='2')
        # ax.legend(loc=2, labels=['Ideal Match', 'Training Data (r^2 = 0.791)', 'Testing Data (r^2 = 0.757)'], framealpha=1)
        ax.legend(loc=2, labels=['Ideal Match', 'Testing Data (r^2 = 0.757)'], framealpha=1)
    ax.set_xlabel('True del_Rt')
    if ylab:
        ax.set_ylabel('Fitted del_Rt')
    # plt.xlabel('True log(del_deaths)')
    # plt.ylabel('Fitted log(del_deaths)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    # plt.tight_layout()
    # plt.savefig('analysis_plots/date_val.png')
