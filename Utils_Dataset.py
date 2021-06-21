### THIS FILE CONTAINS HELPER FUNCTIONS NEEDED FOR THE DATASET CLASS ###


import torch
import pandas as pd
import numpy as np


# Creates dictionary that connects account ids to indices in dataset
def create_acc_ind_dict(df):
    acc_ind = {}
    temp = df['BK_ACCOUNT_ID'][0]
    for i in(range(0, len(df['BK_ACCOUNT_ID']))):
        if temp!=df['BK_ACCOUNT_ID'][i]:
            acc_ind[temp] = i-1
            temp = df['BK_ACCOUNT_ID'][i]
        if i==len(df['BK_ACCOUNT_ID']) - 1:
            acc_ind[temp] = i
    return acc_ind


# Removes entries with negative remaining lifetime
def remove_negative_days(df, inplace=True):
    idx = np.where(df['RemaningLifetime'].to_numpy() < 0)
    if inplace:
        df.drop(idx[0], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        out_df = df.drop(idx)
        out_df.reset_index(drop=True, inplace=True)
        return out_df


# Returns the correct indices for the observation corresponding to a certain day
def day_fixer(dataset, obs_length, last_idx_original):
    out_arr = torch.LongTensor([0 for i in range(0, len(dataset))])
    for i in range(0, len(dataset)):
        first = dataset[i][2][0]
        # last = dataset[i][2][obs_length[i]]
        possible_days = torch.tensor([first-obs_length + i for i in range(0, 10)])
        if first > obs_length:
            for j in possible_days:
                is_day = dataset[i][2]==j
                if is_day.any():
                    out_arr[i] = is_day.nonzero()
                    break
        else:
            out_arr[i] = last_idx_original[i]
    return out_arr


# Looks through df and takes note of any account that has an observation gap of more than n days
# remove_accounts=True means accounts with gaps of size larger than n are removed from the dataset. Notice that this is done in-place
# show_progress=True means every 1000th iteration is marked by printing its number
# Have used n=10 earlier in order to fix anomaly problem (some accounts seemed to have multiple RemaningDays "timelines")
# Some index reseting and stuff might be necessary after the changes are made
def find_anomalies(df, n, remove_accounts=False, show_progress=True):
    prev = 0
    ids = np.array([], dtype=np.int32)
    bk_ids = np.array([], dtype=np.int32)
    for i in range(1, len(df)):
        if df.iloc[i]['RemaningLifetime'] < df.iloc[prev]['RemaningLifetime'] - n:
            if df.iloc[i]['BK_ACCOUNT_ID'] == df.iloc[prev]['BK_ACCOUNT_ID']:
                ids = np.append(ids, i)
                if df.iloc[i]['BK_ACCOUNT_ID'] not in bk_ids:
                    bk_ids = np.append(bk_ids, df.iloc[i]['BK_ACCOUNT_ID'])
                    print("Accounts with faults: ", len(bk_ids))
        prev += 1
        if i % 1000 == 999 and show_progress:
            print(i)
    for i in range(0, len(bk_ids)):
        df.drop(df[df.BK_ACCOUNT_ID==bk_ids[i]].index, inplace=True)
    return ids, bk_ids


# Creates aggregated dataset based on whole dataset
def create_shortened_set(df, filename, write_to=None, output=False,
        mean_vars = ['CREDIT_LIMIT_AMT',
       'IntCashSum','IntPurchaSum',
       'CustomerAge',
       'BALANCE_AMT', 'IEL_AMT', 'CASH_BALANCE_AMT', 'OVERDUE_AMT',
       'DomCashSum', 'DomPurchaseSum',
       'Transfersum', 'FeeSum', 'SumAirlineL12M',
       'SumELECTRIC_APPLIANCEL12M', 'SumFOOD_STORES_WAREHOUSEL12M',
       'SumHOTEL_MOTELL12M', 'SumHARDWAREL12M', 'SumINTERIOR_FURNISHINGSL12M',
       'SumOTHER_RETAILL12M', 'SumOTHER_SERVICESL12M',
       'SumOTHER_TRANSPORTL12M', 'SumRECREATIONL12M',
       'SumRESTAURANTS_BARSL12M', 'SumSPORTING_TOY_STORESL12M',
       'SumTRAVEL_AGENCIESL12M', 'SumVEHICLESL12M', 'SumQuasiCashL12M',
       'SumAirlineL3M', 'SumELECTRIC_APPLIANCEL3M',
       'SumFOOD_STORES_WAREHOUSEL3M', 'SumHOTEL_MOTELL3M', 'SumHARDWAREL3M',
       'SumINTERIOR_FURNISHINGSL3M', 'SumOTHER_RETAILL3M',
       'SumOTHER_SERVICESL3M', 'SumOTHER_TRANSPORTL3M', 'SumRECREATIONL3M',
       'SumRESTAURANTS_BARSL3M', 'SumSPORTING_TOY_STORESL3M',
       'SumTRAVEL_AGENCIESL3M', 'SumVEHICLESL3M', 'SumQuasiCashL3M',
       'UtilizationL12', 'UtilizationL3',
       'AvgRevBalL3onL12'], cum_vars = ['Transfernum', 'FeeNum', 'DomCashNum', 'DomPurchaseNum', 'IntCashNum', 'IntPurchaseNum']):
    for_dropping = pd.concat([pd.Series(mean_vars), pd.Series(cum_vars)], ignore_index=True)
    other_vars = df.columns.drop(for_dropping)
    other_vars.drop(['RemaningLifetime'])
    acc_ind = create_acc_ind_dict(df)
    df_small = df_small = pd.DataFrame(data=df.loc[0:len(acc_ind), df.columns]) #pd.DataFrame(data=df.iloc[0:len(acc_ind)][df.columns])
    j=0
    prev_ind=0
    for k, v in acc_ind.items():
        df_small.loc[j, mean_vars] = np.mean(df.loc[prev_ind:v+1, mean_vars], axis=0)
        df_small.loc[j, cum_vars] = df.loc[prev_ind:v+1, cum_vars].cumsum().iloc[-1]
        df_small.loc[j, other_vars] = df.loc[v, other_vars]
        df_small.loc[j, 'RemaningLifetime'] = df.loc[prev_ind, 'RemaningLifetime']
        prev_ind = v+1
        j+=1
    if write_to=='excel':
        df_small.to_excel('C:/Skolearbeid/Vår 2021/Master/Datasett/' + filename)
    elif write_to=='csv':
        df_small.to_csv('C:/Skolearbeid/Vår 2021/Master/Datasett/' + filename, sep=';', decimal=',', index=False)
    if output:
        return df_small


# Creates aggregated dataset based on days from "start" to "end"
def create_short_from_days(df, start, end, write_to_excel=False, write_to_csv = False, filename=0, output=True,
        mean_vars = ['CREDIT_LIMIT_AMT','BALANCE_AMT', 'IEL_AMT', 'CASH_BALANCE_AMT', 'OVERDUE_AMT',
       'DomCashSum', 'DomPurchaseSum','IntCashSum','IntPurchaSum','Transfersum', 'FeeSum', 'SumAirlineL12M','SumELECTRIC_APPLIANCEL12M', 'SumFOOD_STORES_WAREHOUSEL12M',
       'SumHOTEL_MOTELL12M', 'SumHARDWAREL12M', 'SumINTERIOR_FURNISHINGSL12M','SumOTHER_RETAILL12M', 'SumOTHER_SERVICESL12M','SumOTHER_TRANSPORTL12M', 'SumRECREATIONL12M', 'SumRESTAURANTS_BARSL12M', 'SumSPORTING_TOY_STORESL12M',
       'SumTRAVEL_AGENCIESL12M', 'SumVEHICLESL12M', 'SumQuasiCashL12M', 'SumAirlineL3M', 'SumELECTRIC_APPLIANCEL3M',
       'SumFOOD_STORES_WAREHOUSEL3M', 'SumHOTEL_MOTELL3M', 'SumHARDWAREL3M','SumINTERIOR_FURNISHINGSL3M', 'SumOTHER_RETAILL3M', 'SumOTHER_SERVICESL3M', 'SumOTHER_TRANSPORTL3M', 'SumRECREATIONL3M', 'SumRESTAURANTS_BARSL3M', 'SumSPORTING_TOY_STORESL3M',
       'SumTRAVEL_AGENCIESL3M', 'SumVEHICLESL3M', 'SumQuasiCashL3M','UtilizationL12', 'UtilizationL3','AvgRevBalL3onL12'],
        cum_vars = ['Transfernum', 'FeeNum', 'DomCashNum', 'DomPurchaseNum', 'IntCashNum', 'IntPurchaseNum']
        ):
    acc_ind = create_acc_ind_dict(df)
    for_dropping = pd.concat([pd.Series(mean_vars), pd.Series(cum_vars)], ignore_index=True)
    other_vars = df.columns.drop(for_dropping)
    df_small = pd.DataFrame(data=df.loc[0:len(acc_ind), df.columns])
    j=0
    prev_ind=0
    b = df['RemaningLifetime'].to_numpy()
    t0 = time.time()
    for k, v in acc_ind.items():
        a = b[prev_ind:v+1]
        d = np.where(np.logical_and(a > a[0] - end, a < a[0] - start, a > 0))
        if len(d[0] > 5):
            c = d[0][-1]+prev_ind
            df_small.loc[j, mean_vars] = np.mean(df.loc[d[0]+prev_ind, mean_vars], axis=0)
            df_small.loc[j, cum_vars] = df.loc[d[0]+prev_ind, cum_vars].cumsum().iloc[-1]
            df_small.loc[j, other_vars] = df.loc[c, other_vars]
        else:
            df_small.loc[j, 'RemaningLifetime'] = -1
        j+=1
        prev_ind=v+1
    invals = np.where(df_small['RemaningLifetime'].to_numpy() < 0)
    if len(invals[0] > 0):
        df_small.drop(invals[0], inplace=True)
        df_small.reset_index(drop=True, inplace=True)
    t1 = time.time()
    print("Time to create dataset:", t1-t0, "seconds.")
    if write_to_excel:
        df_small.to_excel('C:/Skolearbeid/Høst 2020/Prosjektoppgave/Datasets/' + filename)
        print("File saved as", filename)
    if write_to_csv:
        df_small.to_csv('C:/Skolearbeid/Høst 2020/Prosjektoppgave/Datasets/' + filename, sep=';',
                        decimal=',', index=False)
        print("File saved as", filename)
    if output:
        return df_small


if __name__ == '__main__':
    pass