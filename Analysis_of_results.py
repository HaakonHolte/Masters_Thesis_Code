### THIS FILE CONTAINS CODE USED TO ANALYZE THE PREDICTIONS OBTAINED IN THE THESIS ###

# Import relevant modules
import torch
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from utils.Utils_Dataset import *
from utils.Utils_Train import *
from utils.Utils_General import *
from utils.Utils_ResultAnalysis import *
import importlib
import dill

#%% Fetch data from relevant models
train_name = '0406_NLL70_2_train.pkl'
val_name = '0406_NLL70_2_val.pkl'
test_name = '0406_NLL70_2_test.pkl'

#train_name = '0106_NLL2_train.pkl'
#val_name = '0106_NLL2_val.pkl'
#test_name = '0106_NLL2_test.pkl'

train_data = open_result_file(train_name)
val_data = open_result_file(val_name)
test_data = open_result_file(test_name)

# Needs correct indexing
train_hazards = train_data[:][0]
train_is_def = train_data[:][1]
train_remdays = train_data[:][2]
train_idxs = train_data[:][3]
val_hazards = val_data[:][0]
val_is_def = val_data[:][1]
val_remdays = val_data[:][2]
val_idxs = val_data[:][3]
test_hazards = test_data[:][0]
test_is_def = test_data[:][1]
test_remdays = test_data[:][2]
test_idxs = test_data[:][3]

# Split defaulters and non-defaulters
dtr_hazards, ntr_hazards, dv_hazards, nv_hazards, dte_hazards, nte_hazards = split_def_ndef_hazards(train_hazards, val_hazards, test_hazards, train_is_def, val_is_def, test_is_def)
dtr_remdays, ntr_remainingdays, dv_remdays, nv_remdays, dte_remdays, nte_remdays = split_def_ndef_remdays(train_remdays, val_remdays, test_remdays, train_is_def, val_is_def, test_is_def)
dtr_last_hazards, ntr_last_hazards, dv_last_hazards, nv_last_hazards, dte_last_hazards, nte_last_hazards = get_def_ndef_lastdaypreds(dtr_hazards, ntr_hazards, dv_hazards, nv_hazards, dte_hazards, nte_hazards,
                                                                                                                                     train_idxs, val_idxs, test_idxs, train_is_def, val_is_def, test_is_def)
# Concatenate predictions on defaulters and non-defaulters, with defaulters first
train_last_hazards = torch.cat([dtr_last_hazards, ntr_last_hazards])
val_last_hazards = torch.cat([dv_last_hazards, nv_last_hazards])
test_last_hazards = torch.cat([dte_last_hazards, nte_last_hazards])

#%% Rescaling
# Rescale validation predictions
mean_sfactor_val = train_last_hazards.mean()/val_last_hazards.mean()
median_sfactor_val = train_last_hazards.median()/val_last_hazards.median()
scale_factor_val = (mean_sfactor_val + median_sfactor_val)/2
val_hazards_r = scale_factor_val*val_last_hazards
dv_hazards_r = scale_factor_val*dv_last_hazards
nv_hazards_r = scale_factor_val*nv_last_hazards

# Rescale test predictions
mean_sfactor_test = train_last_hazards.mean()/test_last_hazards.mean()
median_sfactor_test = train_last_hazards.median()/test_last_hazards.median()
scale_factor_test = (mean_sfactor_test + median_sfactor_test)/2
test_hazards_r = scale_factor_test*test_last_hazards
dte_hazards_r = scale_factor_test*dte_last_hazards
nte_hazards_r = scale_factor_test*nte_last_hazards

#%% Set bins for remaining lifetime. Might need adjustment based on results
remdays_bins = [0, 15, 30, 60, 90, 120, 180, 270, 365, 1000]
bin_labels = ['0-15', '15-30', '30-60', '60-90', '90-120', '120-180', '180-270', '270-365', '>365']

# Bin remaining lifetimes
train_remdays_binned, val_remdays_binned, test_remdays_binned = bin_remdays(dtr_remdays, dv_remdays, dte_remdays, remdays_bins, bin_labels)
hazard_data = get_hazard_data(train_remdays_binned, dtr_last_hazards)

#%% Plot mean hazard rates in bins against predicted lifetime within bins
days = [7, 21, 45, 75, 105, 150, 225, 315, 370]
plt.plot(days, hazard_data[:, 2], label='Mean', color='blue')
plt.xlabel('Remaining lifetime')
plt.ylabel('Hazard rate')
plt.legend()
plt.savefig('HAZARD_MEAN_70.pdf', format='pdf')
plt.show()

#%% Plot median hazard rates in bins against predicted lifetime within bins
plt.plot(days[2:], hazard_data[2:, 3], label='Median', color='red')
plt.xlabel('Remaining lifetime')
plt.ylabel('Hazard rate')
plt.legend()
plt.savefig('HAZARD_MEDIAN_70.pdf', format='pdf')
plt.show()

#%% Set bins for hazard rates
# For run 0106_NLL2
#hazard_bins = [0, 3.1e-04, 3.5e-04, 4.1e-04, 4.9e-04, 5.8e-04, 7.0e-04, 8.5e-04, 0.0125, 1]
#hazard_bin_labels = ['0-3.1e-04', '3.1e-04-3.5e-04', '3.5e-04-4.1e-04', '4.1e-04-4.9e-05',
#                     '4.9e-04-5.8e-04', '5.8e-04-7.0e-04', '7.0e-04-8.5e-04', '8.5e-04-0.0125', '>0.0125']

# For run 0406_NLL70_2
hazard_bins = [0, 0.0006, 0.001, 0.0015, 0.0022, 0.003, 0.004, 0.0052, 0.022, 1] #Good values for 70 days
hazard_bin_labels = ['0-6e-04', '6e-04-1e-03', '1e-03-1.5e-03', '1.5e-03-2.2e-03',
                     '2.2e-03-3e-03', '3e-02-4e-03', '4e-03-5.2e-03', '5.2e-03-0.022', '>0.022']

# Set lifetimes to predict, set conversions from hazard rates to lifetimes
lt_to_pred = [370, 315, 225, 150, 105, 75, 45, 21, 7]
hazard_to_lifetime = {hazard_bin_labels[i] : lt_to_pred[i] for i in range(0, len(lt_to_pred))}

#%% Get predicted lifetimes
# Get predicted lifetimes for train set
train_lifetimes = predict_lifetime(train_last_hazards, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
dtr_lifetimes = predict_lifetime(dtr_last_hazards, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
ntr_lifetimes = predict_lifetime(ntr_last_hazards, hazard_bins, hazard_bin_labels, hazard_to_lifetime)

# Get predicted lifetimes for validation set
val_lifetimes = predict_lifetime(val_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
dv_lifetimes = predict_lifetime(dv_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
nv_lifetimes = predict_lifetime(nv_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)

# Get predicted lifetimes for test set
test_lifetimes = predict_lifetime(test_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
dte_lifetimes = predict_lifetime(dte_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)
nte_lifetimes = predict_lifetime(nte_hazards_r, hazard_bins, hazard_bin_labels, hazard_to_lifetime)

#%% Residual plot train set
plt.scatter(dtr_remdays, dtr_remdays-dtr_lifetimes)
plt.xlabel('Remaining lifetime')
plt.ylabel('Difference between true and predicted remaining lifetime')
plt.savefig('RESIDUAL_TRAIN_70.pdf', format='pdf')
plt.show()

#%% Residual plot validation set
plt.scatter(dv_remdays, dv_remdays-dv_lifetimes)
plt.xlabel('Remaining lifetime')
plt.ylabel('Difference between true and predicted remaining lifetime')
plt.savefig('RESIDUAL_VAL_70.pdf', format='pdf')
plt.show()

#%% Residual plot test set
plt.scatter(dte_remdays, dte_remdays-dte_lifetimes)
plt.xlabel('Remaining lifetime')
plt.ylabel('Difference between true and predicted remaining lifetime')
plt.savefig('RESIDUAL_TEST_70.pdf', format='pdf')
plt.show()

#%% Get statistics for binary classification
dtr_stats, ntr_stats, otr_stats, tr_faccs = get_365statistics(dtr_lifetimes, ntr_lifetimes, dtr_remdays, ntr_remainingdays)
dv_stats, nv_stats, ov_stats, v_faccs = get_365statistics(dv_lifetimes, nv_lifetimes, dv_remdays, nv_remdays)
dte_stats, nte_stats, ote_stats, te_faccs = get_365statistics(dte_lifetimes, nte_lifetimes, dte_remdays, nte_remdays)

#%% Get MRAE values
train_mrae = MRAE(dtr_lifetimes, dtr_remdays)
val_mrae = MRAE(dv_lifetimes, dv_remdays)
test_mrae = MRAE(dte_lifetimes, dte_remdays)
