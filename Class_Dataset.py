### THIS FILE CONTAINS THE DEFINITION OF THE DATASET GENERATOR CLASS ###
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split
from utils.Utils_Dataset import create_acc_ind_dict, remove_negative_days
from utils.Utils_General import dict_indexer

class CreditDatasetGenerator(TensorDataset):
    def __init__(self, filename, num_vars, drop_vars):
        self.num_vars = pd.Series(num_vars)
        self.drop_vars = pd.Series(drop_vars)
        self.df = pd.read_csv(filename, sep=";", decimal=",")
        self.all_vars = pd.Series(self.df.columns)
        self.no_use = pd.Series(['RK_ACCOUNT_ID', 'AccountCreatedDateId', 'CashBackStatus',
                        'AccountBalanceDateId', 'prevPeriodId', 'Segment9Name', 'Segment23Name'])
        self.df.drop(self.no_use, axis=1, inplace=True)
        temp_bool = self.all_vars.isin(self.no_use)
        self.used_vars = self.all_vars.drop(self.all_vars[temp_bool].index)
        self.used_vars.reset_index(drop=True, inplace=True)
        self.acc_ind = create_acc_ind_dict(self.df)

    def train_val_test_split(self, test_size, seed=None):
        ids = np.array([key for key, value in self.acc_ind.items()])
        train_ids, self.test_ids = train_test_split(ids, test_size=test_size)
        self.train_ids, self.val_ids = train_test_split(train_ids, test_size=test_size/(1-test_size))

    def get_num_vars(self):
        return(self.num_vars)

    def get_drop_vars(self):
        return(self.drop_vars)

    def get_used_vars(self):
        return(self.used_vars)

    def set_num_vars(self, new_vars):
        self.num_vars = pd.Series(new_vars)

    def set_drop_vars(self, new_vars):
        self.drop_vars = pd.Series(new_vars)

    def set_used_vars(self, new_vars):
        self.used_vars = pd.Series(new_vars)

    def add_num_vars(self, new_vars):
        self.num_vars = self.num_vars.append(pd.Series(new_vars), ignore_index=True)

    def add_drop_vars(self, new_vars):
        self.drop_vars = self.drop_vars.append(pd.Series(new_vars), ignore_index=True)

    def add_used_vars(self, new_vars):
        self.used_vars = self.used_vars.append(pd.Series(new_vars), ignore_index=True)

    def remove_num_vars(self, variables):
        bools = self.num_vars.isin(variables)
        self.num_vars.drop(self.num_vars[bools].index, inplace=True)
        self.num_vars.reset_index(drop=True, inplace=True)

    def remove_drop_vars(self, variables):
        bools = self.drop_vars.isin(variables)
        self.drop_vars.drop(self.drop_vars[bools].index, inplace=True)
        self.drop_vars.reset_index(drop=True, inplace=True)

    def remove_used_vars(self, variables):
        bools = self.used_vars.isin(variables)
        self.used_vars.drop(self.used_vars[bools].index, inplace=True)
        self.used_vars.reset_index(drop=True, inplace=True)

    def update_used(self):
        bools = self.used_vars.isin(self.drop_vars)
        if bools.any():
            self.used_vars.drop(self.used_vars[bools].index, inplace=True)
            self.used_vars.reset_index(drop=True, inplace=True)
        else:
            print("used_vars is already up to date.")
        print("Currently used variables are:", self.used_vars)

    def restore_used(self):
        bools = self.all_vars.isin(self.no_use)
        self.used_vars = self.all_vars.drop(self.all_vars[temp_bool].index)
        self.used_vars.reset_index(drop=True, inplace=True)

    def prepare_data(self):
        df = self.df
        self.update_used()
        df.drop(self.drop_vars, axis=1, inplace=True)
        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        df = pd.get_dummies(df, drop_first=True)
        self.df = df
        self.col_names = df.columns

    # def normal_transform(self):
    #     df = self.df
    #     self.update_used()
    #     bool_nums = self.num_vars.isin(self.used_vars)
    #     use_nums = self.num_vars[bool_nums]
    #     train_ids_bools = df.BK_ACCOUNT_ID.isin(self.train_ids)
    #     standard_scaler = skp.StandardScaler().fit(df[train_ids_bools][use_nums])
    #     df[use_nums] = standard_scaler.transform(df[use_nums])
    #     self.df = df

    def setup_tensor_dataset(self):
        df = self.df
        n_custs = len(self.acc_ind)
        n_feats = len(df.columns) - 2
        max_length = max(df['RemaningLifetime'])
        df_tensor = torch.empty(n_custs, max_length, n_feats)
        def_ind = torch.empty(n_custs, max_length)
        rem_days = torch.empty(n_custs, max_length)
        prev_idx = 0
        customer_idx = 0
        idx_list = []
        for k, v in self.acc_ind.items():
            length = v - prev_idx
            idx_list.append(length-1)
            def_ind[customer_idx, 0:length] = torch.tensor(df['DC2Ind'].iloc[prev_idx + 1:v + 1].to_numpy())
            rem_days[customer_idx, 0:length] = torch.tensor(df['RemaningLifetime'].iloc[prev_idx + 1:v + 1].to_numpy())
            df_tensor[customer_idx, 0:length, :] = torch.tensor(
                df.iloc[prev_idx + 1:v + 1].drop(['DC2Ind', 'RemaningLifetime'], axis=1).to_numpy())
            if length < max_length:
                df_tensor[customer_idx, length:max_length, :] = np.NaN
            customer_idx += 1
            prev_idx = v
        self.df = 0
        self.df_tensor = df_tensor
        self.def_ind = def_ind
        self.rem_days = rem_days
        self.idx_list = np.array(idx_list)

    def normal_transform(self):
        self.update_used()
        bool_nums = self.num_vars.isin(self.used_vars)
        use_nums = self.num_vars[bool_nums]
        colname_to_idx = {key: i for i, key in enumerate(self.col_names)}
        var_index = dict_indexer(colname_to_idx, use_nums)
        bka_to_idx = {key: i for i, key in enumerate(self.acc_ind.keys())}
        train_inds = dict_indexer(bka_to_idx, self.train_ids)
        train_last_idx = self.idx_list[train_inds]
        for i in range(0, max(train_last_idx)):
            standard_scaler = skp.StandardScaler().fit(self.df_tensor[train_inds, i][:, var_index])
            self.df_tensor[:, i][:, var_index] = torch.tensor(standard_scaler.transform(self.df_tensor[:, i][:, var_index]),
                                                         dtype=torch.float32)
    def assemble_tensor_dataset(self):
        self.df_tensor[torch.isnan(self.df_tensor)] = 0
        self.dataset = TensorDataset(self.df_tensor, self.def_ind, self.rem_days)

    def get_df_tensor(self):
        return self.df_tensor

    def get_tensor_dataset(self):
        return self.dataset

    def get_idx_list(self):
        return self.idx_list

    def get_colnames(self):
        return self.col_names

    def get_train_val_test_ids(self):
        return self.train_ids, self.val_ids, self.test_ids

    def get_acc_ind(self):
        return self.acc_ind



if __name__ == '__main__':
    pass