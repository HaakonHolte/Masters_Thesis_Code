### THIS FILE CONTAINS SOME HELPER FUNCTIONS FOR THE RESULT ANALYSIS ###

import torch
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from utils.Utils_Dataset import *
from utils.Utils_Train import *
from utils.Utils_General import *
import importlib
import dill


# Use dill to open files
def open_result_file(filename):
    with open(filename, 'rb') as file:
        data = dill.load(file)
    return data


# Get predicted lifetimes from hazard rate predictions with defined hazard bins and conversions from hazard rates to lifetimes
def predict_lifetime(hazard_preds, hazard_bins, hazard_bin_labels, hazard_to_lifetime):
    preds = torch.zeros(len(hazard_preds))
    hazards_grouped = pd.cut(hazard_preds, bins=hazard_bins, labels=hazard_bin_labels)
    for i in range(0, len(hazards_grouped)):
        preds[i] = hazard_to_lifetime[hazards_grouped[i]]
    return preds


# Plots remainingdays on the x-axis and predicted hazard rate of subjects on the y-axis. Can also be used to plot predicted lifetimes
def plot_hazards_remainingdays(def_hazards, def_remdays, ndef_hazards, ndef_remdays, plot_ndefs = True, x_lower=None, x_upper=None, title="Plot"):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter(def_remdays, def_hazards, c='blue')             #def_hazards[:, -1]
    if plot_ndefs:
        ax1.scatter(ndef_remdays, ndef_hazards, c='red')         # ndef_hazards[:, -1]
    if x_lower is not None and x_upper is not None:
        plt.xlim(x_lower, x_upper)
    plt.title(title)
    plt.show()


# Calculates predicted lifetime from predicted hazard rate using pre-defined bins
def hazard_to_lifetime(hazard_preds, hazard_bins, hazard_bin_labels, hazard_to_lifetime):
    preds = torch.zeros(len(hazard_preds))
    hazards_grouped = pd.cut(hazard_preds, bins=hazard_bins, labels=hazard_bin_labels)
    for i in range(0, len(hazards_grouped)):
        preds[i] = hazard_to_lifetime[hazards_grouped[i]]
    return preds


# Get some statistics for binary predictions about default one year ahead in time
def get_365statistics(def_pred_lifetimes, ndef_pred_lifetimes, def_remdays, ndef_remdays):
    a1 = def_pred_lifetimes[def_remdays < 365]
    c1 = def_pred_lifetimes[def_remdays >= 365]
    b1 = a1[a1 < 365]
    d1 = c1[c1 >= 365]
    def_tp = len(d1)
    def_fp = len(a1) - len(b1)
    def_fn = len(c1) - len(d1)
    def_tn = len(b1)
    # Get the observed lifetimes for non-defaulters that are above and below one year
    ndef_above365 = ndef_remdays[ndef_remdays >= 365]
    ndef_below365 = ndef_remdays[ndef_remdays < 365]
    # Get the predicted lifetimes of the above mentioned subjects
    ndef_above365_preds = ndef_pred_lifetimes[ndef_remdays >= 365]
    ndef_below365_preds = ndef_pred_lifetimes[ndef_remdays < 365]
    # Get the predicted lifetimes that are larger than one year
    ndef_preds_above365 = ndef_above365_preds[ndef_above365_preds >= 365]
    # True positives among non-defaulters will then be length of ndef_preds_above365
    ndef_tp = len(ndef_preds_above365)
    ndef_fn = len(ndef_above365) - len(ndef_preds_above365)
    # For censored observations with observation time < one year, checks if predicted lifetime is larger than observation time
    ndef_preds_is_larger = ndef_below365_preds > ndef_below365
    is_larger_percentage = ndef_preds_is_larger.sum()/len(ndef_below365)
    ndef_fp = 0
    ndef_tn = 0
    # Precision: TP/(TP + FP)
    def_prec = def_tp/(def_tp+def_fp)
    ndef_prec = ndef_tp/(ndef_tp+ndef_fp)
    # Recall: TP/(TP + FN)
    def_rec = def_tp/(def_tp+def_fn)
    ndef_rec = ndef_tp/(ndef_tp+ndef_fn)
    # F1 score: 2*precision*recall/(precision+recall)
    def_f1 = 2*def_prec*def_rec/(def_prec+def_rec)
    ndef_f1 = 2*ndef_prec*ndef_rec/(ndef_prec+ndef_rec)
    # Balanced accuracy: (acc1 + acc2)/2
    bacc = ((def_tp + def_tn)/len(def_pred_lifetimes) + (ndef_tp + ndef_tn)/len(ndef_preds_above365))/2
    # Need binary predictions for Matthews corrcoef
    def_pred_bin = torch.tensor(def_pred_lifetimes >= 365)
    ndef_pred_bin = torch.tensor(ndef_above365_preds >= 365)
    def_true_bin = torch.tensor(def_remdays >= 365)
    ndef_true_bin = torch.tensor(ndef_above365 >= 365)
    all_above365 = torch.cat([def_true_bin, ndef_true_bin])
    all_preds_bin = torch.cat([def_pred_bin, ndef_pred_bin])
    # Matthews correlation coefficient
    mcc = matthews_corrcoef(all_above365, all_preds_bin)
    return (def_prec, def_rec, def_f1), (ndef_prec, ndef_rec, ndef_f1, is_larger_percentage), (bacc, mcc), \
           (len(def_pred_lifetimes), len(ndef_above365), len(ndef_below365))


# Split predictions on train and val set into predictions for defaulters and non-defaulters
def split_def_ndef_hazards(train_preds, val_preds, test_preds, train_is_def, val_is_def, test_is_def):
    #def_train_hazards = train_preds[train_set[:][1][:, 0]==1]
    #ndef_train_hazards = train_preds[train_set[:][1][:, 0]==0]
    #def_val_hazards = val_preds[val_set[:][1][:, 0]==1]
    #ndef_val_hazards = val_preds[val_set[:][1][:, 0]==0]
    def_train_hazards = train_preds[train_is_def == 1]
    ndef_train_hazards = train_preds[train_is_def == 0]
    def_val_hazards = val_preds[val_is_def == 1]
    ndef_val_hazards = val_preds[val_is_def == 0]
    def_test_hazards = test_preds[test_is_def == 1]
    ndef_test_hazards = test_preds[test_is_def == 0]
    return def_train_hazards, ndef_train_hazards, def_val_hazards, ndef_val_hazards, def_test_hazards, ndef_test_hazards


# Split remaining lifetime on train and val set into remaining lifetime for defaulters and non-defaulters
def split_def_ndef_remdays(train_rem_days, val_rem_days, test_rem_days, train_is_def, val_is_def, test_is_def):
    #def_train_remainingdays = array_indexer(train_set[train_set[:][1][:, 0] == 1][2],
                                      #train_idxs[train_set[:][1][:, 0] == 1])
    #def_remainingdays = array_indexer(train_set[train_set[:][1][:, 0]==1][2], train_set[train_set[:][1][:, 0]==1][3])
    #ndef_train_remainingdays = array_indexer(train_set[train_set[:][1][:, 0] == 0][2],
                                       #train_idxs[train_set[:][1][:, 0] == 0])
    #ndef_remainingdays = array_indexer(train_set[train_set[:][1][:, 0]==0][2], train_set[train_set[:][1][:, 0]==0][3])
    #def_val_remainingdays = array_indexer(val_set[val_set[:][1][:, 0] == 1][2], val_idxs[val_set[:][1][:, 0] == 1])
    #ndef_val_remainingdays = array_indexer(val_set[val_set[:][1][:, 0] == 0][2], val_idxs[val_set[:][1][:, 0] == 0])
    def_train_remainingdays = train_rem_days[train_is_def==1]
    ndef_train_remainingdays = train_rem_days[train_is_def==0]
    def_val_remainingdays = val_rem_days[val_is_def==1]
    ndef_val_remainingdays = val_rem_days[val_is_def==0]
    def_test_remainingdays = test_rem_days[test_is_def == 1]
    ndef_test_remainingdays = test_rem_days[test_is_def == 0]
    return def_train_remainingdays, ndef_train_remainingdays, def_val_remainingdays, ndef_val_remainingdays, def_test_remainingdays, ndef_test_remainingdays


# Get "last day" hazard predictions for train and validation sets, with predictions of defaulters and non-defaulters separated
def get_def_ndef_lastdaypreds(dtr_hazards, ndtr_hazards, dval_hazards, ndval_hazards, dte_hazards, ndte_hazards,
                              train_idxs, val_idxs, test_idxs, train_is_def, val_is_def, test_is_def):
    #def_train_preds = array_indexer(def_train_hazards, train_idxs[train_set[:][1][:, 0]==1])
    #ndef_train_preds = array_indexer(ndef_train_hazards, train_idxs[train_set[:][1][:, 0] == 0])
    #def_val_preds = array_indexer(def_val_hazards, val_idxs[val_set[:][1][:, 0]==1])
    #ndef_val_preds = array_indexer(ndef_val_hazards, val_idxs[val_set[:][1][:, 0]==0])
    def_train_preds = array_indexer(dtr_hazards, train_idxs[train_is_def == 1])
    ndef_train_preds = array_indexer(ndtr_hazards, train_idxs[train_is_def == 0])
    def_val_preds = array_indexer(dval_hazards, val_idxs[val_is_def == 1])
    ndef_val_preds = array_indexer(ndval_hazards, val_idxs[val_is_def == 0])
    def_test_preds = array_indexer(dte_hazards, test_idxs[test_is_def == 1])
    ndef_test_preds = array_indexer(ndte_hazards, test_idxs[test_is_def == 0])
    return def_train_preds, ndef_train_preds, def_val_preds, ndef_val_preds, def_test_preds, ndef_test_preds


# Bin remaining lifetime of defaulters in train and val set into provided bins
def bin_remdays(def_train_remdays, def_val_remdays, def_test_remdays, bins, bin_labels):
    remdays_train_grouped = pd.cut(def_train_remdays, bins=bins, labels=bin_labels)
    remdays_val_grouped = pd.cut(def_val_remdays, bins=bins, labels=bin_labels)
    remdays_test_grouped = pd.cut(def_test_remdays, bins=bins, labels=bin_labels)
    return remdays_train_grouped, remdays_val_grouped, remdays_test_grouped


# Get min, max, mean, median and standard deviation within the different bins
def get_hazard_data(remdays_binned, preds):
    hazard_data = torch.zeros(size=(len(remdays_binned.categories), 5))
    for i in range(0, len(remdays_binned.categories)):
        data = torch.zeros(remdays_binned.value_counts()[i])
        counter = 0
        for j in range(0, len(preds)):
            if remdays_binned[j] == remdays_binned.categories[i]:
                data[counter] = preds[j]
                counter += 1
        hazard_data[i][0] = data.min()
        hazard_data[i][1] = data.max()
        hazard_data[i][2] = data.mean()
        hazard_data[i][3] = data.median()
        hazard_data[i][4] = data.std()
    return hazard_data


# Assumes that surv_est does not contain censored observations that have already defaulted or been censored
def IPCWBrier(surv_est, defaulted, cens_est, prev_cens_est):
    score = 0
    for i in surv_est:
        score += i**2*defaulted/prev_cens_est + (1-i)**2*(1-defaulted)/cens_est
    return score/len(surv_est)


# Computes mean relative absolute error
def MRAE(preds, truths):
    return (torch.abs(preds - truths)/truths).sum()/len(truths)

