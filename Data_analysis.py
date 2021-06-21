### THIS FILE CONTAINS CODE USED TO ANALYZE THE DATASET USED IN THE THESIS ###

# Import relevant modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.Utils_Dataset import *
from utils.Utils_DataAnalysis import *
import importlib

#%% Define helper functions
# Sets up dataframe with variable that is to be analysed
def setup_temp_df(variable, length):
    temp_df = pd.DataFrame(-np.ones(shape=(len(df_short), length+2)))
    first = 0
    ind = 0
    for i in df_short.BK_ACCOUNT_ID:
        last = acc_ind[i]
        temp_df.loc[ind, 0] = i
        temp_df.loc[ind, 1] = df_short[df_short.BK_ACCOUNT_ID==i]['DC2Ind'].to_numpy()
        temp_df.loc[ind, 2:last+2-first] = df[df.BK_ACCOUNT_ID==i][variable].to_numpy()
        first=last+1
        ind+=1
    return temp_df


# Calculates mean, median, standard deviation and quantiles q for all time steps in temp_df
def calculate_statistics(temp_df, max_lifetime, q=[80, 90]):
    mean = np.zeros(max_lifetime)
    median = np.zeros(max_lifetime)
    variance = np.zeros(max_lifetime)
    quantiles = []
    for i in q:
        quantiles.append(np.zeros((2, max_lifetime)))
    quantile80 = np.zeros((2, max_lifetime))
    quantile90 = np.zeros((2, max_lifetime))
    for j in range(2, max_lifetime + 2):
        mean[j-2] = temp_df.loc[:, j][temp_df.loc[:, j] != -1].mean()
        median[j-2] = temp_df.loc[:, j][temp_df.loc[:, j] != -1].median()
        variance[j-2] = temp_df.loc[:, j][temp_df.loc[:, j] != -1].var()
        for k in range(0, len(quantiles)):
            lower = (100 - q[k])/200
            quantiles[k][:, j-2] = temp_df.loc[:, j][temp_df.loc[:, j] != -1].quantile((lower, 1-lower))
    return [mean, median, variance], quantiles, q


# Sets up plots of time series and statistics for the variable in question
def setup_plot(ax, temp_df, max_lifetime, colors, q, to_plot):
    stat_names = ['Mean', 'Median', 'Variance']
    stat_list, quantiles, quantile_values = calculate_statistics(temp_df, max_lifetime, q)
    seq_len = len(temp_df.iloc[0, 2:][temp_df.iloc[0, 2:] > -1])
    ax.plot(np.arange(0, seq_len), temp_df.iloc[0, 2:seq_len+2], color=colors[0], label='Observations')
    for i in range(1, len(temp_df)):
        seq_len = len(temp_df.iloc[i, 2:][temp_df.iloc[i, 2:] > -1])
        ax.plot(np.arange(0, seq_len), temp_df.iloc[i, 2:seq_len+2], color=colors[0], linewidth=0.2)
    for j in range(0, len(to_plot)):
        if to_plot[j]==1 and j!=3:
            seq_len = len(stat_list[j][np.isfinite(stat_list[j])])
            ax.plot(np.arange(0, seq_len), stat_list[j][0:seq_len],
                    color=colors[j+1], label=stat_names[j])
        elif to_plot[j]==1 and j==3:
            for k in range(0, len(quantiles)):
                seq_len = len(quantiles[k][0, np.isfinite(quantiles[k][0])])
                ax.plot(np.arange(0, seq_len), quantiles[k][0, 0:seq_len], color=colors[j+k+1], label='Q'+str(q[k]))
                ax.plot(np.arange(0, seq_len), quantiles[k][1, 0:seq_len], color=colors[j+k+1])
    return ax


# Makes final preparations and returns figure and axes for finished plot
def plot_timeseries(df, variable, max_lifetime_defs,
                    max_lifetime_ndefs, quantiles=[80, 90], to_plot=[1, 0, 0, 0],
                    y_lower=None, y_upper=None, colors=['grey', 'yellow', 'blue', 'red']):
    fig, ax = plt.subplots(1, 2)
    temp_df = setup_temp_df(variable, max_lifetime_ndefs)
    temp_df_def = temp_df[temp_df[1] == 1]
    temp_df_ndef = temp_df[temp_df[1] == 0]
    ax[0] = setup_plot(ax[0], temp_df_def, max_lifetime_defs, colors, quantiles, to_plot)
    if y_lower is not None and y_upper is not None:
        ax[0].set_ylim(y_lower, y_upper)
    ax[0].set_title("Defaulters")
    ax[0].legend(prop={'size':6}, loc='upper right')
    ax[1] = setup_plot(ax[1], temp_df_ndef, max_lifetime_ndefs, colors, quantiles, to_plot)
    if y_lower is not None and y_upper is not None:
        ax[1].set_ylim(y_lower, y_upper)
    ax[1].set_title("Non-defaulters")
    ax[1].legend(prop={'size':6}, loc='upper right')
    plt.suptitle("Timeseries plot for " + variable)
    plt.tight_layout()
    plt.show()
    return fig, ax


# Deprecated version of setup_plot
def setup_plot_old(ax, obss, variable, max_lifetime, colors):
    summ = np.zeros(max_lifetime)
    n_alive = np.zeros(max_lifetime)
    a = df[df.BK_ACCOUNT_ID == obss.iloc[0]][variable]
    ax.plot(np.arange(0, len(a)), a, color=colors[0], label='Observations')
    summ += np.pad(a, pad_width=(0, max_lifetime - len(a)), mode='constant', constant_values=0)
    n_alive[0:len(a)] += 1
    for i in obss.iloc[1:]:
        a = df[df.BK_ACCOUNT_ID == i][variable]
        ax.plot(np.arange(0, len(a)), a, color=colors[0])
        summ += np.pad(a, pad_width=(0, max_lifetime - len(a)), mode='constant', constant_values=0)
        n_alive[0:len(a)] += 1
    n_alive = n_alive[n_alive != 0]
    summ = summ[0:len(n_alive)]
    summ /= n_alive
    ax.plot(np.arange(0, len(summ)), summ, color=colors[1], label='Mean')
    return ax

#%% Creates aggregated dataset
create_shortened_set(df, filename='lifetimes_augm_short.csv', write_to='csv', output=True)

#%% Read dataset and create dictionary for associating bank id's with their positions in the dataset
df = pd.read_csv('C:/Skolearbeid/Vår 2021/Master/Datasett/lifetimes_augm.csv', sep=";", decimal=",")
acc_ind = create_acc_ind_dict(df)
#remove_negative_days(df, inplace=True)

#%% Read aggregated dataset and create dictionary for associating bank id's with their positions in the dataset
df_short = pd.read_csv('C:/Skolearbeid/Vår 2021/Master/Datasett/lifetimes_augm_short.csv', sep=';', decimal=',')
acc_ind_short = create_acc_ind_dict(df_short)

#%% Scale two variables so they are positive
df_short['CASH_BALANCE_AMT']*=(-1)
df_short['OVERDUE_AMT']*=(-1)
df['CASH_BALANCE_AMT']*=(-1)
df['OVERDUE_AMT']*=(-1)

#%% Set up design and target, does one-hot encoding for categorical variables
X, y, z = create_design_and_target(df_short, False, ['RK_ACCOUNT_ID', 'BK_ACCOUNT_ID', 'AccountCreatedDateId',
            'AccountBalanceDateId', 'prevPeriodId',
            'DC2Ind', 'RemaningLifetime', 'Segment9Name', 'Segment23Name', 'CashBackStatus',
            'SumAirlineL12M','SumELECTRIC_APPLIANCEL12M','SumFOOD_STORES_WAREHOUSEL12M','SumHOTEL_MOTELL12M',
            'SumHARDWAREL12M','SumINTERIOR_FURNISHINGSL12M','SumOTHER_RETAILL12M','SumOTHER_SERVICESL12M',
            'SumOTHER_TRANSPORTL12M','SumRECREATIONL12M','SumRESTAURANTS_BARSL12M',
            'SumSPORTING_TOY_STORESL12M','SumTRAVEL_AGENCIESL12M','SumVEHICLESL12M','SumQuasiCashL12M'
])

#%% Get bank account ids of defaulters and non-defaulters
defs = df_short[df_short.DC2Ind==1]['BK_ACCOUNT_ID']
nondefs = df_short[df_short.DC2Ind==0]['BK_ACCOUNT_ID']
max_lifetime_defs = df_short[df_short.DC2Ind==1]['RemaningLifetime'].max()
max_lifetime_ndefs = df_short[df_short.DC2Ind==0]['RemaningLifetime'].max()


#%%
### LONGITUDINAL DATA ANALYSIS ###
# The next cells create time series plots used in the thesis
fig_oa, ax_oa = plot_timeseries(df, 'OVERDUE_AMT', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=0, y_upper=8000,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_oa[0].set_ylim(0, 6000)
ax_oa[1].set_ylim(0, 6000)
fig_oa.show()

#%%
fig_oa.savefig("TS_OVERDUE", format='pdf')

#%%
fig_ba, ax_ba = plot_timeseries(df, 'BALANCE_AMT', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_ba[0].set_ylim(0, 110000)
ax_ba[1].set_ylim(0, 110000)
#fig_ba.show()

fig_cb, ax_cb = plot_timeseries(df, 'CASH_BALANCE_AMT', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_cb[0].set_ylim(0, 70000)
ax_cb[1].set_ylim(0, 70000)
#fig_cb.show()

fig_iel, ax_iel = plot_timeseries(df, 'IEL_AMT', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_iel[0].set_ylim(0, 100000)
ax_iel[1].set_ylim(0, 100000)
#fig_iel.show()

fig_cl, ax_cl = plot_timeseries(df, 'CREDIT_LIMIT_AMT', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_cl[0].set_ylim(0, 100000)
ax_cl[1].set_ylim(0, 100000)
#fig_cl.show()

fig_u3, ax_u3 = plot_timeseries(df, 'UtilizationL3', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_u3[0].set_ylim(0, 3)
ax_u3[1].set_ylim(0, 3)
#fig_u3.show()

fig_u12, ax_u12 = plot_timeseries(df, 'UtilizationL12', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_arb, ax_arb = plot_timeseries(df, 'AvgRevBalL3onL12', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_arb[0].set_ylim(0, 9)
ax_arb[1].set_ylim(0, 9)
#fig_arb.show()

fig_ba.savefig("TS_BALANCE", format='pdf')
fig_cb.savefig("TS_CASH_BALANCE", format='pdf')
fig_iel.savefig("TS_IEL", format='pdf')
fig_cl.savefig("TS_CREDITLIM", format='pdf')
fig_u3.savefig("TS_UTIL3", format='pdf')
fig_u12.savefig("TS_UTIL12", format='pdf')
fig_arb.savefig("TS_ARB", format='pdf')

#%%
fig_trans, ax_trans = plot_timeseries(df, 'Transfersum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_trans[0].set_ylim(0, 20000)
ax_trans[1].set_ylim(0, 20000)
fig_trans.show()

#%%
fig_intp, ax_intp = plot_timeseries(df, 'IntPurchaSum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_intp[0].set_ylim(0, 10000)
ax_intp[1].set_ylim(0, 10000)
fig_intp.show()

fig_intc, ax_intc = plot_timeseries(df, 'IntCashSum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_domp, ax_domp = plot_timeseries(df, 'DomPurchaseSum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_domc, ax_domc = plot_timeseries(df, 'DomCashSum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_fee, ax_fee = plot_timeseries(df, 'FeeSum', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

#%%
fig_quasi, ax_quasi = plot_timeseries(df, 'SumQuasiCashL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_quasi[0].set_ylim(0, 20000)
ax_quasi[1].set_ylim(0, 20000)
#fig_quasi.show()

fig_food, ax_food = plot_timeseries(df, 'SumFOOD_STORES_WAREHOUSEL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_food[0].set_ylim(0, 8000)
ax_food[1].set_ylim(0, 8000)
#fig_food.show()

fig_interior, ax_interior = plot_timeseries(df, 'SumINTERIOR_FURNISHINGSL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_interior[0].set_ylim(0, 3000)
ax_interior[1].set_ylim(0, 3000)
#fig_interior.show()

fig_oretail, ax_oretail = plot_timeseries(df, 'SumOTHER_RETAILL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_oretail[0].set_ylim(0, 5000)
ax_oretail[1].set_ylim(0, 5000)
#fig_oretail.show()

fig_oservices, ax_oservices = plot_timeseries(df, 'SumOTHER_SERVICESL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_oservices[0].set_ylim(0, 3000)
ax_oservices[1].set_ylim(0, 3000)
#fig_oservices.show()

fig_rest, ax_rest = plot_timeseries(df, 'SumRESTAURANTS_BARSL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_rest[0].set_ylim(0, 3000)
ax_rest[1].set_ylim(0, 3000)
#fig_rest.show()

fig_quasi.savefig("TS_QUASI.pdf", format='pdf')
fig_food.savefig("TS_FOOD.pdf", format='pdf')
fig_interior.savefig("TS_INTERIOR.pdf", format='pdf')
fig_oretail.savefig("TS_ORETAIL.pdf", format='pdf')
fig_oservices.savefig("TS_OSERVICES.pdf", format='pdf')
fig_rest.savefig("TS_REST.pdf", format='pdf')

#%%
fig_airline, ax_airline = plot_timeseries(df, 'SumAirlineL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_airline[0].set_ylim(0, 5000)
ax_airline[1].set_ylim(0, 5000)
fig_airline.show()

fig_electric, ax_electric = plot_timeseries(df, 'SumELECTRIC_APPLIANCEL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_electric[0].set_ylim(0, 5000)
ax_electric[1].set_ylim(0, 5000)
fig_electric.show()

fig_hotel, ax_hotel = plot_timeseries(df, 'SumHOTEL_MOTELL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_hotel[0].set_ylim(0, 5000)
ax_hotel[1].set_ylim(0, 5000)
fig_hotel.show()

fig_hardw, ax_hardw = plot_timeseries(df, 'SumHARDWAREL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_hardw[0].set_ylim(0, 5000)
ax_hardw[1].set_ylim(0, 5000)
fig_hardw.show()

fig_otransport, ax_otransport = plot_timeseries(df, 'SumOTHER_TRANSPORTL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_otransport[0].set_ylim(0, 2000)
ax_otransport[1].set_ylim(0, 2000)
fig_otransport.show()

fig_recr, ax_recr = plot_timeseries(df, 'SumRECREATIONL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])
ax_recr[0].set_ylim(0, 2000)
ax_recr[1].set_ylim(0, 2000)
fig_recr.show()

fig_sport, ax_sport = plot_timeseries(df, 'SumSPORTING_TOY_STORESL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_travel, ax_travel = plot_timeseries(df, 'SumTRAVEL_AGENCIESL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])

fig_vehic, ax_vehic = plot_timeseries(df, 'SumVEHICLESL3M', max_lifetime_defs, max_lifetime_ndefs,
                quantiles=[80, 90], to_plot=[1, 1, 0, 1], y_lower=None, y_upper=None,
                colors=['grey', 'yellow', 'blue', 'no', 'red', 'orange'])


#%% #### STATIC DATA ANALYSIS ####
# Does some set up for the static data analysis
uncens_lt = df_short.RemaningLifetime[df_short['DC2Ind']==1]
cens_lt = df_short.RemaningLifetime[df_short['DC2Ind']==0]
bins = [77, 100, 150, 200, 300, 400, 500, 1000]
bin_labels = ['77-100', '100-150', '150-200', '200-300', '300-400', '400-500', '>500']
uncens_binned = pd.cut(uncens_lt, bins=bins, labels=bin_labels)
cens_binned = pd.cut(cens_lt, bins=bins, labels=bin_labels)
all_binned = pd.cut(df_short['RemaningLifetime'], bins=bins, labels=bin_labels)

max_lifetime = df_short['RemaningLifetime'].max()
min_lifetime = df_short['RemaningLifetime'].min()

uncens_lt.describe()

uncens_q = pd.qcut(uncens_lt, q=4)
cens_q = pd.qcut(cens_lt, q=4)
all_q = pd.qcut(df_short['RemaningLifetime'], q=4)

#%% Creates a histogram and a KDE plot of lifetimes for defaulters
fig, ax1 = plt.subplots()
sns.histplot(data=uncens_lt, stat='count', legend=True, label='Histogram', ax=ax1)
plt.legend(loc=(0.74, 0.84))
ax2 = plt.twinx()
sns.kdeplot(data=uncens_lt, color='orange', legend=True, label='KDE', ax=ax2)
ax1.set_xlabel('Number of days')
plt.title("Histogram and KDE for lifetimes")
plt.legend(loc=(0.74, 0.91))
plt.tight_layout()
#plt.show()
plt.savefig("KDE_HISR_LIFETIMES.pdf", format='pdf')

#%% Creates histogram of variable passed to argument 'x'
hist = sns.displot(df_short, x='HAS_DIRECT_DEBIT_AGREEMENT_IND', col='DC2Ind',
                   facet_kws={'sharey': False})
labs = ['Female', 'Male']
for axes in hist.axes.flat:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, horizontalalignment='right')
    #axes.set_xlim(0, 200)
#plt.show()
plt.savefig("HIST_DIRECTDEBIT.pdf", format='pdf')

#%%
fig1, ax1 = plt.subplots()
yaxis2 = ax1.twinx()
ax1.set_xlim(0, 700)
ax1.set_xlabel("Days")
ax1.set_ylabel("Frequency")
yaxis2.set_ylabel("Relative frequency")
p1 = ax1.plot()
p2 = yaxis2.plot()

#%%
#ax1.hist(uncens_lt)
#ax1.kde(uncens_lt)
#ax1 = uncens_lt.plot.kde()
#ax2 = uncens_lt.plot.hist()
plt.xlim(0, 700)
plt.xlabel('Days')
plt.ylabel('Relative frequency')
plt.title('Histogram and KDE for defaults')
plt.show()
