### THIS IS THE FILE THAT PERFORMS THE MODEL TRAINING ###

# Relevant imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.pytorchtools import EarlyStopping
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from time import time
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from utils.Utils_Dataset import *
from utils.Utils_Train import *
from utils.Utils_General import *
from utils.Utils_ResultAnalysis import *
from classes.Class_Dataset import CreditDatasetGenerator
from classes.Class_RNN import RNN
from classes.Class_LSTM import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from procedures.Procedure_Train import train_rnn, train_lstm
import importlib
import dill

#%% Helper function that needs to be in the same file that uses it
def save_results(train_preds, train_idxs, val_preds, val_idxs, test_preds, test_idxs, name, keep_set=False, set_name=None):
    train_is_def = train_set[:][1][:, 0]
    val_is_def = val_set[:][1][:, 0]
    test_is_def = test_set[:][1][:, 0]
    train_rem_days = array_indexer(train_set[:][2], train_idxs)
    val_rem_days = array_indexer(val_set[:][2], val_idxs)
    test_rem_days = array_indexer(test_set[:][2], test_idxs)
    train_info = TensorDataset(train_preds, train_is_def, train_rem_days, train_idxs)
    val_info = TensorDataset(val_preds, val_is_def, val_rem_days, val_idxs)
    test_info = TensorDataset(test_preds, test_is_def, test_rem_days, test_idxs)
    with open(name + '_train.pkl', 'wb') as file:
        dill.dump(train_info, file)
    with open(name + '_val.pkl', 'wb') as file:
        dill.dump(val_info, file)
    with open(name + '_test.pkl', 'wb') as file:
        dill.dump(test_info, file)
    if keep_set:
        with open(set_name + '_train.pkl', 'wb') as file:
            dill.dump(train_set, file)
        with open(set_name + '_val.pkl', 'wb') as file:
            dill.dump(val_set, file)
        with open(set_name + '_test.pkl', 'wb') as file:
            dill.dump(test_set, file)

# Transforms remaining days to be between zero and one. Necessary for using the MSE loss function
def day_transform(array):
    max = array.max()
    min = array.min()
    return (array-min)/(max-min)

#%% Set name of run and set up tensorboard
run_name = 'this_run'
#print(f"Tensorboard version: {tensorboard.__version__}")

# default `log_dir` is "runs"
writer = SummaryWriter('runs/' + run_name)

#%% Setting device
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#%% Setting variables to be included, loading csv and setting up tensor dataset. Also does train-test split and gives BK ids for these
num_vars = ['ApplicationScore',
'CREDIT_LIMIT_AMT',
'CustomerAge',
'MonthsSinceAccountCreated',
'BALANCE_AMT',
'IEL_AMT',
'CASH_BALANCE_AMT',
'OVERDUE_AMT',
'DomCashNum',
'DomCashSum',
'DomPurchaseNum',
'DomPurchaseSum',
'IntCashNum',
'IntCashSum',
'IntPurchaseNum',
'IntPurchaSum',
'Transfernum',
'Transfersum',
'FeeNum',
'FeeSum',
# 'SumAirlineL12M',
# 'SumELECTRIC_APPLIANCEL12M',
# 'SumFOOD_STORES_WAREHOUSEL12M',
# 'SumHOTEL_MOTELL12M',
# 'SumHARDWAREL12M',
# 'SumINTERIOR_FURNISHINGSL12M',
# 'SumOTHER_RETAILL12M',
# 'SumOTHER_SERVICESL12M',
# 'SumOTHER_TRANSPORTL12M',
# 'SumRECREATIONL12M',
# 'SumRESTAURANTS_BARSL12M',
# 'SumSPORTING_TOY_STORESL12M',
# 'SumTRAVEL_AGENCIESL12M',
# 'SumVEHICLESL12M',
# 'SumQuasiCashL12M',
'SumAirlineL3M',
'SumELECTRIC_APPLIANCEL3M',
'SumFOOD_STORES_WAREHOUSEL3M',
'SumHOTEL_MOTELL3M',
'SumHARDWAREL3M',
'SumINTERIOR_FURNISHINGSL3M',
'SumOTHER_RETAILL3M',
'SumOTHER_SERVICESL3M',
'SumOTHER_TRANSPORTL3M',
'SumRECREATIONL3M',
'SumRESTAURANTS_BARSL3M',
'SumSPORTING_TOY_STORESL3M',
'SumTRAVEL_AGENCIESL3M',
'SumVEHICLESL3M',
'SumQuasiCashL3M',
'UtilizationL12',
'UtilizationL3',
'AvgRevBalL3onL12',
]
cat_vars = ['PRODUCT_NAME',
'HAS_DIRECT_DEBIT_AGREEMENT_IND',
'HAS_ESTATEMENT_AGREEMENT_IND',
'GENDER_NAME',
'DISTRIBUTOR_NAME'
]
drop_vars = [
'BK_ACCOUNT_ID',
'GEN_BK_ACCOUNT_STATUS_CD',
'SumAirlineL12M',
'SumELECTRIC_APPLIANCEL12M',
'SumFOOD_STORES_WAREHOUSEL12M',
'SumHOTEL_MOTELL12M',
'SumHARDWAREL12M',
'SumINTERIOR_FURNISHINGSL12M',
'SumOTHER_RETAILL12M',
'SumOTHER_SERVICESL12M',
'SumOTHER_TRANSPORTL12M',
'SumRECREATIONL12M',
'SumRESTAURANTS_BARSL12M',
'SumSPORTING_TOY_STORESL12M',
'SumTRAVEL_AGENCIESL12M',
'SumVEHICLESL12M',
'SumQuasiCashL12M',
'PostalCodeFirst2'
]
credit_dataset_gen = CreditDatasetGenerator('C:/Skolearbeid/Vår 2021/Master/Datasett/lifetimes_augm.csv',
                                        num_vars, drop_vars)
credit_dataset_gen.train_val_test_split(test_size=0.2, seed=88)
credit_dataset_gen.prepare_data()
credit_dataset_gen.setup_tensor_dataset()
credit_dataset_gen.normal_transform()
credit_dataset_gen.assemble_tensor_dataset()
credit_dataset = credit_dataset_gen.get_tensor_dataset() # Get the tensor dataset
idx_list = credit_dataset_gen.get_idx_list() # Get list of index for last observed day for customers
colnames = credit_dataset_gen.get_colnames() # Get names of columns included in final dataframe
train_ids, val_ids, test_ids = credit_dataset_gen.get_train_val_test_ids() # Get train, validation and test BK ids
acc_ind = credit_dataset_gen.get_acc_ind() # Get acc_ind

colname_to_idx = {key: i for i, key in enumerate(colnames)}
bka_to_idx = {key: i for i, key in enumerate(acc_ind.keys())}

train_inds = dict_indexer(bka_to_idx, train_ids)
val_inds = dict_indexer(bka_to_idx, val_ids)
test_inds = dict_indexer(bka_to_idx, test_ids)

train_last_idx_full = idx_list[train_inds]
val_last_idx_full = idx_list[val_inds]
test_last_idx_full = idx_list[test_inds]

#%% Set parameters
# Net params
input_size = 66
hidden_size = 120
output_size = 1
n_layers = 4
activation = 'tanh'
dropout_prob = 0.5
grad_clip = 5

# Dataloader params
batch_size = 256
days = 100
n_customers = 6774

# Optimization params
lr = 0.001
momentum = 0.5

# Loss params
reg = 0.1
loss_weight = 0.6
bce_weights = torch.tensor((0.8, 0.2), requires_grad=False)

#%% Create dataloader for training set
selected_customers = torch.tensor(np.random.choice(train_inds, n_customers, replace=False)) #Select random customers for train set
#train_last_idx = idx_list[selected_customers] # Fetch the index for the last observation for the train customers
val_length = int(0.2*n_customers) # Set size of validation set
val_selected = torch.tensor(np.random.choice(val_inds, val_length, replace=False))
val_selected = val_selected[credit_dataset[val_selected][2][:, 0] > days]
test_selected = torch.tensor(np.random.choice(test_inds, val_length, replace=False))
test_selected = test_selected[credit_dataset[test_selected][2][:, 0] > days]
#val_last_idx = idx_list[val_selected] # Fetch index for last obs for val customers
#val_indss = val_inds[val_selected]

full_set_idxs = day_fixer(credit_dataset, days, idx_list)
remdays_scaled = day_transform(array_indexer(credit_dataset[:][2], full_set_idxs))
remdayss_train = remdays_scaled[selected_customers]
remdayss_val = remdays_scaled[val_selected]
remdayss_test = remdays_scaled[test_selected]

train_no = torch.LongTensor(range(0, len(df_train[0])))
#train_idxs = day_fixer(train_set, days, idx_list[selected_customers]) #Use when only observations up to "days" should be used for training
train_idxs = torch.LongTensor(idx_list[selected_customers]) #Use when full time series should be used for training
df_train = credit_dataset[selected_customers, 0:max(train_idxs)]
train_set = TensorDataset(df_train[0], df_train[1], df_train[2], train_idxs, remdayss_train, train_no)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#train_last_idx[train_last_idx > idx_days] = idx_days #Sets last_idx to "idx_days" for customers who live longer than "days". Fixes problem with unpacking sequences during backprop
train_last_idxs = day_fixer(train_set, days, idx_list[selected_customers])

#%% Set up validation set
df_val = credit_dataset[val_selected]
val_no = torch.LongTensor(range(0, len(df_val[0])))
val_set = TensorDataset(df_val[0], df_val[1], df_val[2], remdayss_val, val_no)
val_idxs = day_fixer(val_set, days, idx_list[val_selected])
val_dataloader = DataLoader(val_set, batch_size=batch_size)

#%% Set up test set
df_test = credit_dataset[test_selected]
test_no = torch.LongTensor(range(0, len(df_test[0])))
test_set = TensorDataset(df_test[0], df_test[1], df_test[2], remdayss_test, test_no)
test_idxs = day_fixer(test_set, days, idx_list[test_selected])
test_dataloader = DataLoader(test_set, batch_size=batch_size)

#%% Set up LSTM and move it to device
lstm_net = LSTM(input_size, hidden_size, output_size, n_layers, dropout_prob)
lstm_net = lstm_net.to(device)

#%% Set up placeholders for hazard rate predictions
train_hazards = torch.zeros(size=(n_customers, len(train_set[0][0])), requires_grad=False)
val_hazards = torch.zeros(size=(len(val_selected), days), requires_grad=False)

#%% Set loss criterion and optimization algorithm
# Note that NLL loss is imported from different file and called uncens_nll_loss
loss_func = surv_loss
bce = nn.BCELoss()
#mse = nn.MSELoss(reduction='sum')  # For when MSE is used
optimizer = optim.SGD(lstm_net.parameters(), lr = lr, momentum = momentum)

#%% Train with BCE and NLL
save_path = run_name + '.pt'
train_loss_history = []
train_acc_history = []
train_conc_history = []
val_loss_history = []
val_acc_history = []
val_conc_history = []
best_val_conc = 0
best_loss_conc = 0
best_train_preds = 0
best_val_preds = 0
epochs=20
early_stop_patience = 6
required_improvement = 10e-4
early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, delta=required_improvement, path=save_path)
bce_weights = torch.tensor((0.9, 0.1), requires_grad=False)
start = time()
for j in range(epochs):
    running_loss = 0
    correct = 0
    total = 0
    train_conc = 0
    t0 = time()
    for i, data in enumerate(train_dataloader):
        lstm_net.train()
        X, y, z, d1, d2, n = data
        X = X.to(device)
        y = y.to(device)
        z = z.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        n = n.to(device)
        current_batchsize = X.size(0)
        h = lstm_net.init_hidden_zero(current_batchsize)
        h = tuple([e.data for e in h])
        if current_batchsize != batch_size:
            batch_end = i * batch_size + current_batchsize
        else:
            batch_end = (i + 1) * batch_size
        y_pred, h = lstm_net(X, h)
        y_pred_packd = pack_padded_sequence(y_pred, d1,
                                            batch_first=True, enforce_sorted=False)
        y_packd = pack_padded_sequence(y, d1,
                                       batch_first=True, enforce_sorted=False)
        y_pred_unpackd = pad_packed_sequence(y_pred_packd, batch_first=True)
        y_unpackd = pad_packed_sequence(y_packd, batch_first=True)
        train_hazards[n, 0:y_pred_unpackd[0].size(1)] = y_pred_unpackd[0].clone().detach()

        surv_func1 = torch.zeros(len(y_pred_unpackd[0]), requires_grad=True)
        surv_func = surv_func1.clone()
        for k in range(len(y_pred_unpackd[0])):
            surv_func[k] = (1-y_pred_unpackd[0][k, 0:y_pred_unpackd[1][k]]).prod()

        optimizer.zero_grad()
        bce_loss = bce(surv_func, torch.tensor([bce_weights[int(1-y_unpackd[0][t, 0])] for t in range(0, len(y_unpackd[0][:, 0]))], requires_grad=False)*(1-y_unpackd[0][:, 0]))
        nll_loss = uncens_nll_loss(y_pred_unpackd, y_unpackd)
        loss = loss_func(loss_weight, nll_loss, bce_loss)
        loss.backward()
        nn.utils.clip_grad_norm(lstm_net.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.round(y_pred)
        correct += preds[:, -1].eq(y[:, 1]).sum().item()
        total += y.size(0)

        days2def = array_indexer(z, d1)    # Gets the remaining time to default at day "days"
        train_conc += c_index(array_indexer(train_hazards[n], train_last_idxs[n]).numpy(), days2def, y[:, 0])

        # Validate after every 10 batches
        if i % 10 == 9:
            lstm_net.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            n_batches = 0
            val_conc = 0
            for m, val_data in enumerate(val_dataloader, 0):
                X_val, y_val, z_val, d_val, n_val = val_data
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                z_val = z_val.to(device)
                d_val = d_val.to(device)
                n_val = n_val.to(device)
                current_batchsize = X_val.size(0)
                h = lstm_net.init_hidden_zero(current_batchsize)
                # h = tuple([e.data for e in h])
                if current_batchsize != batch_size:
                    batch_end = m * batch_size + current_batchsize
                else:
                    batch_end = (m + 1) * batch_size
                y_val_pred, h = lstm_net(X_val, h)
                y_val_pred_packd = pack_padded_sequence(y_val_pred, val_idxs[m* batch_size:batch_end],
                                                        batch_first=True, enforce_sorted=False)
                y_val_packd = pack_padded_sequence(y_val, val_idxs[m * batch_size:batch_end],
                                                   batch_first=True, enforce_sorted=False)
                y_val_pred_unpackd = pad_packed_sequence(y_val_pred_packd, batch_first=True)
                y_val_unpackd = pad_packed_sequence(y_val_packd, batch_first=True)
                val_hazards[n_val, 0:y_val_pred_unpackd[0].size(1)] = y_val_pred_unpackd[0].clone().detach()

                surv_func_val = torch.zeros(len(y_val_pred_unpackd[0]), requires_grad=False)
                for p in range(len(y_val_pred_unpackd[0])):
                    surv_func_val[p] = (1 - y_val_pred_unpackd[0][p, 0:y_val_pred_unpackd[1][p]]).prod()

                nll_loss = torch.tensor(uncens_nll_loss(y_val_pred_unpackd, y_val_unpackd), requires_grad=False)
                bce_loss = bce(surv_func_val, torch.tensor([bce_weights[int(1 - y_val_unpackd[0][t, 0])] for t in range(0, len(y_val_unpackd[0][:, 0]))], requires_grad=False) * (1 - y_val_unpackd[0][:, 0]))
                loss = loss_func(loss_weight, nll_loss, bce_loss)

                val_loss += loss.item()
                val_preds = torch.round(y_val_pred)
                val_correct += val_preds[:, -1].eq(y_val[:, 1]).sum().item()
                val_total += y_val.size(0)

                days2def = array_indexer(z_val, val_idxs[m * batch_size:batch_end])
                val_conc += c_index(array_indexer(val_hazards[n_val], val_idxs[n_val]).numpy(), days2def, y_val[:, 0])

                n_batches += 1

            avg_val_loss = val_loss / n_batches
            val_acc = val_correct / val_total
            avg_val_conc = val_conc/n_batches
            early_stopping(avg_val_loss, lstm_net)
            if avg_val_conc > best_val_conc:
                best_val_conc = avg_val_conc
            if early_stopping.counter==0:
                best_loss_conc = avg_val_conc
                best_val_preds = val_hazards
                best_train_preds = train_hazards
            writer.add_scalar("Val loss", avg_val_loss, i + j * len(train_dataloader))
            #writer.add_scalar("Val accuracy", val_acc, i + j * len(train_dataloader))
            writer.add_scalar("Val concordance", avg_val_conc, i + j*len(train_dataloader))
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(val_acc)
            val_conc_history.append(avg_val_conc)
            print("Current average validation loss:", avg_val_loss)
            #print("Current validation accuracy:", val_acc)
            print("Current validation concordance:", avg_val_conc)
            if len(val_loss_history) > 1:
                print("Previous average validation loss:", val_loss_history[-2])
                #print("Previous validation accuracy:", val_acc_history[-2])
                print("Previous validation concordance:", val_conc_history[-2])

        if i % 10 == 9:  # Print some statistics every 10 mini-batches
            running_loss /= 10
            correct /= total
            avg_train_conc = train_conc/10
            writer.add_scalar("Train loss", running_loss, i + j * len(train_dataloader))
            #writer.add_scalar("Train accuracy", correct, i + j * len(train_dataloader))
            writer.add_scalar("Train concordance", avg_train_conc, i + j * len(train_dataloader))
            print("[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%" % (j + 1, i + 1, running_loss, 100 * correct))
            train_loss_history.append(running_loss)
            train_acc_history.append(correct)
            train_conc_history.append(avg_train_conc)
            running_loss = 0.0
            correct = 0.0
            total = 0
            train_conc = 0
        if early_stopping.early_stop:
            print("Stopping early after", j, "epochs and", i, "batches. Best validation concordance:", best_val_conc)
            break
    if early_stopping.early_stop:
        end = time()
        hours, minutes, secs = convert_seconds(end - start)
        print('Finished training. Time to complete', j, "epochs with batch size", batch_size, ":", hours, "hours",
              minutes,
              "minutes", secs, "seconds.")
        break
    else:
        t1 = time()
        print("Time to complete epoch", j + 1, ":", t1 - t0, "seconds.")
        if j == (epochs - 1) and total != 0:
            correct /= total
            print("Accuracy for final iterations in last epoch:", correct)
if not early_stopping.early_stop:
    end = time()
    hours, minutes, secs = convert_seconds(end - start)
    print('Finished training. Time to complete', epochs, "epochs with batch size", batch_size, ":", hours, "hours", minutes,
          "minutes", secs, "seconds.")
writer.flush()

#%% Train with MSE and BCE
save_path = run_name + '.pt'
train_loss_history = []
train_acc_history = []
train_conc_history = []
val_loss_history = []
val_acc_history = []
val_conc_history = []
best_val_conc = 0
best_loss_conc = 0
best_train_preds = 0
best_val_preds = 0
epochs=20
early_stop_patience = 6
required_improvement = 10e-4
early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, delta=required_improvement, path=save_path)
start = time()
for j in range(epochs):
    running_loss = 0
    correct = 0
    total = 0
    train_conc = 0
    t0 = time()
    for i, data in enumerate(train_dataloader):
        lstm_net.train()
        X, y, z, d1, d2, n = data
        X = X.to(device)
        y = y.to(device)
        z = z.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        n = n.to(device)
        current_batchsize = X.size(0)
        h = lstm_net.init_hidden_zero(current_batchsize)
        h = tuple([e.data for e in h])
        if current_batchsize != batch_size:
            batch_end = i * batch_size + current_batchsize
        else:
            batch_end = (i + 1) * batch_size
        y_pred, h = lstm_net(X, h)
        y_pred_packd = pack_padded_sequence(y_pred, max(train_idxs)*torch.ones(d1.size()),
                                            batch_first=True, enforce_sorted=False)
        y_packd = pack_padded_sequence(y, max(train_idxs)*torch.ones(d1.size()),
                                       batch_first=True, enforce_sorted=False)
        y_pred_unpackd = pad_packed_sequence(y_pred_packd, batch_first=True)
        y_unpackd = pad_packed_sequence(y_packd, batch_first=True)
        train_hazards[n, 0:y_pred_unpackd[0].size(1)] = y_pred_unpackd[0].clone().detach()

        surv_func1 = torch.zeros(len(y_pred_unpackd[0]), requires_grad=True)
        surv_func = surv_func1.clone()
        for k in range(len(y_pred_unpackd[0])):
            surv_func[k] = (1-y_pred_unpackd[0][k, 0:y_pred_unpackd[1][k]]).prod()

        optimizer.zero_grad()
        bce_loss = bce(surv_func, torch.tensor([bce_weights[int(1-y_unpackd[0][t, 0])] for t in range(0, len(y_unpackd[0][:, 0]))], requires_grad=False)*(1-y_unpackd[0][:, 0]))
        mse_loss = mse(y_unpackd[0][:, 0]*(1-array_indexer(y_pred_unpackd[0], d1)), y_unpackd[0][:, 0]*d2)/y_unpackd[0][:, 0].sum()
        loss = loss_func(loss_weight, mse_loss, bce_loss)
        loss.backward()
        nn.utils.clip_grad_norm(lstm_net.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.round(y_pred)
        correct += preds[:, -1].eq(y[:, 1]).sum().item()
        total += y.size(0)

        days2def = array_indexer(z, d1)     # Gets the remaining time to default at day "days"
        train_conc += c_index(array_indexer(train_hazards[n], train_last_idxs[n]).numpy(), days2def, y[:, 0])

        # Validate after every 10 batches
        if i % 10 == 9:
            lstm_net.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            n_batches = 0
            val_conc = 0
            for k, data in enumerate(val_dataloader, 0):
                X_val, y_val, z_val, d_val, n_val = data
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                z_val = z_val.to(device)
                d_val = d_val.to(device)
                n_val = n_val.to(device)
                current_batchsize = X_val.size(0)
                h = lstm_net.init_hidden_zero(current_batchsize)
                # h = tuple([e.data for e in h])
                if current_batchsize != batch_size:
                    batch_end = k * batch_size + current_batchsize
                else:
                    batch_end = (k + 1) * batch_size
                y_val_pred, h = lstm_net(X_val, h)
                y_val_pred_packd = pack_padded_sequence(y_val_pred, val_idxs[k * batch_size:batch_end],
                                                        batch_first=True, enforce_sorted=False)
                y_val_packd = pack_padded_sequence(y_val, val_idxs[k * batch_size:batch_end],
                                                   batch_first=True, enforce_sorted=False)
                y_val_pred_unpackd = pad_packed_sequence(y_val_pred_packd, batch_first=True)
                y_val_unpackd = pad_packed_sequence(y_val_packd, batch_first=True)
                val_hazards[n_val, 0:y_val_pred_unpackd[0].size(1)] = y_val_pred_unpackd[0].clone().detach()

                surv_func_val = torch.zeros(len(y_val_pred_unpackd[0]), requires_grad=False)
                for l in range(len(y_val_pred_unpackd[0])):
                    surv_func_val[l] = (1 - y_val_pred_unpackd[0][l, 0:y_val_pred_unpackd[1][l]]).prod()

                mse_loss = mse(y_val_unpackd[0][:, 0]*(1-array_indexer(val_hazards[n_val], val_idxs[n_val])), y_val_unpackd[0][:, 0]*d_val) / y_val_unpackd[0][:, 0].sum()
                bce_loss = bce(surv_func_val, 1 - y_val_unpackd[0][:, 0])
                loss = loss_func(loss_weight, mse_loss, bce_loss)

                val_loss += loss.item()
                val_preds = torch.round(y_val_pred)
                val_correct += val_preds[:, -1].eq(y_val[:, 1]).sum().item()
                val_total += y_val.size(0)

                days2def = array_indexer(z_val, val_idxs[k * batch_size:batch_end])
                val_conc += c_index(array_indexer(val_hazards[n_val], val_idxs[n_val]).numpy(), days2def, y_val[:, 0])

                n_batches += 1

            avg_val_loss = val_loss / n_batches
            val_acc = val_correct / val_total
            avg_val_conc = val_conc/n_batches
            early_stopping(avg_val_loss, lstm_net)
            if avg_val_conc > best_val_conc:
                best_val_conc = avg_val_conc
            if early_stopping.counter == 0:
                best_loss_conc = avg_val_conc
                best_val_preds = val_hazards
                best_train_preds = train_hazards
            writer.add_scalar("Val loss", avg_val_loss, i + j * len(train_dataloader))
            #writer.add_scalar("Val accuracy", val_acc, i + j * len(train_dataloader))
            writer.add_scalar("Val concordance", avg_val_conc, i + j*len(train_dataloader))
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(val_acc)
            val_conc_history.append(avg_val_conc)
            print("Current average validation loss:", avg_val_loss)
            #print("Current validation accuracy:", val_acc)
            print("Current validation concordance:", avg_val_conc)
            if len(val_loss_history) > 1:
                print("Previous average validation loss:", val_loss_history[-2])
                #print("Previous validation accuracy:", val_acc_history[-2])
                print("Previous validation concordance:", val_conc_history[-2])

        if i % 10 == 9:  # Print some statistics every 10 mini-batches
            running_loss /= 10
            correct /= total
            avg_train_conc = train_conc/10
            writer.add_scalar("Train loss", running_loss, i + j * len(train_dataloader))
            #writer.add_scalar("Train accuracy", correct, i + j * len(train_dataloader))
            writer.add_scalar("Train concordance", avg_train_conc, i + j * len(train_dataloader))
            print("[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%" % (j + 1, i + 1, running_loss, 100 * correct))
            train_loss_history.append(running_loss)
            train_acc_history.append(correct)
            train_conc_history.append(avg_train_conc)
            running_loss = 0.0
            correct = 0.0
            total = 0
            train_conc = 0
        if early_stopping.early_stop:
            print("Stopping early after", j, "epochs and", i, "batches. Best validation concordance:", best_val_conc)
            break
    if early_stopping.early_stop:
        end = time()
        hours, minutes, secs = convert_seconds(end - start)
        print('Finished training. Time to complete', j, "epochs with batch size", batch_size, ":", hours, "hours",
              minutes,
              "minutes", secs, "seconds.")
        break
    else:
        t1 = time()
        print("Time to complete epoch", j + 1, ":", t1 - t0, "seconds.")
        if j == (epochs - 1) and total != 0:
            correct /= total
            print("Accuracy for final iterations in last epoch:", correct)
if not early_stopping.early_stop:
    end = time()
    hours, minutes, secs = convert_seconds(end - start)
    print('Finished training. Time to complete', epochs, "epochs with batch size", batch_size, ":", hours, "hours", minutes,
          "minutes", secs, "seconds.")
writer.flush()

#%% Get results on test set
test_hazards = torch.zeros(size=(len(test_selected), days), requires_grad=False)
lstm_net.eval()
for m, test_data in enumerate(test_dataloader, 0):
    X_test, y_test, z_test, d_test, n_test = test_data
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    z_test = z_test.to(device)
    d_test = d_test.to(device)
    n_test = n_test.to(device)
    current_batchsize = X_test.size(0)
    h = lstm_net.init_hidden_zero(current_batchsize)
    if current_batchsize != batch_size:
        batch_end = m * batch_size + current_batchsize
    else:
        batch_end = (m + 1) * batch_size
    y_test_pred, h = lstm_net(X_test, h)
    y_test_pred_packd = pack_padded_sequence(y_test_pred, test_idxs[m * batch_size:batch_end],
                                            batch_first=True, enforce_sorted=False)
    y_test_packd = pack_padded_sequence(y_test, test_idxs[m * batch_size:batch_end],
                                       batch_first=True, enforce_sorted=False)
    y_test_pred_unpackd = pad_packed_sequence(y_test_pred_packd, batch_first=True)
    y_test_unpackd = pad_packed_sequence(y_test_packd, batch_first=True)
    test_hazards[n_test, 0:y_test_pred_unpackd[0].size(1)] = y_test_pred_unpackd[0].clone().detach()

#%% Put results in desired format
test_days_left = array_indexer(test_set[:][2], test_idxs)
test_conc = c_index(array_indexer(test_hazards, test_idxs).numpy(), test_days_left, test_set[:][1][:, 0])

val_days_left = array_indexer(val_set[:][2], val_idxs)
val_conc = c_index(array_indexer(val_hazards, val_idxs).numpy(), val_days_left, val_set[:][1][:, 0])

#%% Save results. Note that model itself is not saved here
save_results(train_hazards, train_last_idxs, val_hazards, val_idxs, test_hazards, test_idxs, run_name + '_results')

