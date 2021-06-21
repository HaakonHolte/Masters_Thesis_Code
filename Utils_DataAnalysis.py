### THIS FILE CONTAINS SOME HELPER FUNCTIONS FOR THE DATA ANALYSIS ###

import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


# Makes KDE plots of 'variables'
def make_kde_plot(df, variables, n_rows, n_cols, figsize=(9, 9)):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    row = 0
    row = 0
    col = 0
    for i in variables:
        if n_rows * n_cols == 1:
            sns.kdeplot(data=df, x=i, hue='DC2Ind', ax=axs, shade=True,
                        common_norm=False)
        elif n_rows == 1 and n_cols > 1:
            sns.kdeplot(data=df, x=i, hue='DC2Ind', ax=axs[col], shade=True,
                        common_norm=False)
            col += 1
        else:
            sns.kdeplot(data=df, x=i, hue='DC2Ind', ax=axs[row, col], shade=True,
                        common_norm=False)
            if col == n_cols - 1:
                row += 1
                col = 0
            else:
                col += 1
    return fig, axs


# Makes histogram plots of 'variables'
def make_hist_plot(df, variables, n_rows, n_cols, figsize=(9, 9)):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    row = 0
    col = 0
    for i in variables:
        hist = sns.displot(df, x=i, col='DC2Ind', ax=axs[row, col],
                           facet_kws={'sharey': False})
        for axes in hist.axes.flat:
            axes.set_xticklabels(axes.get_xticklabels(), rotation=45,
                                 horizontalalignment='right')
        if col == n_cols - 1:
            row += 1
            col = 0
        else:
            col += 1
        return fig, axs


# Gets some rates for defaulters and non-defaulters
def get_01_ratios(df, variable, histplot=False):
    obss = df[variable]
    defs = obss[df.DC2Ind == 1].value_counts()
    ndefs = obss[df.DC2Ind == 0].value_counts()
    frame = {'Total def rate': (defs / obss.value_counts()).sort_index(),
             'Total surv rate': (ndefs / obss.value_counts()).sort_index(),
             'Def rate in defs': (defs / defs.sum()).sort_index(),
             'Surv rate in survs': (ndefs / ndefs.sum()).sort_index(),
             'Obs count': obss.value_counts().sort_index()}
    if histplot:
        hist = sns.displot(df, x=variable, col='DC2Ind',
                           facet_kws={'sharey': False})
        for axes in hist.axes.flat:
            axes.set_xticklabels(axes.get_xticklabels(), rotation=45,
                                 horizontalalignment='right')
    return pd.DataFrame(data=frame)


# Sets up design and target
def create_design_and_target(df, drop_first=False, to_drop=['RK_ACCOUNT_ID',
            'BK_ACCOUNT_ID', 'AccountCreatedDateId', 'AccountBalanceDateId',
            'prevPeriodId','DC2Ind', 'RemaningLifetime', 'Segment9Name',
            'Segment23Name', 'CashBackStatus']):
    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(
        lambda x: x.astype('category'))
    X = df.drop(to_drop, axis=1)
    X = pd.get_dummies(X, drop_first=drop_first)
    y = df['DC2Ind']
    z = df['RemaningLifetime']
    return X, y, z



if __name__=='__main__':
    pass
