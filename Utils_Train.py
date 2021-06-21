### THIS FILE CONTAINS HELPER FUNCTIONS NEEDED FOR THE TRAINING PROCEDURE ###


import torch
import numpy as np
import torch.nn as nn
from lifelines.utils import concordance_index


# Implements negative log-likelihood loss for uncensored observations
def uncens_nll_loss(y_pred_unpackd, y_unpackd):
    #censored_obss_pred = y_pred_unpackd[0][y_unpackd[0][:, 0] == 0].clone().detach()
    #uncensored_obss_pred = y_pred_unpackd[0][y_unpackd[0][:, 0] == 1].clone().detach()
    #days = y_pred_unpackd[1][y_unpackd[0][:, 0] == 1]
    out = torch.tensor(0.0)

    #err = torch.log(y_pred_unpackd[0][:, y_pred_unpackd[1][:]-1])
    #y_unpackd[0][:, 0] * (torch.log(y_pred_unpackd[0][:, y_pred_unpackd[1][:]-1]) + torch.log(-y_pred_unpackd[0])) #(128, 300)

    for i in range(len(y_pred_unpackd[0])):
        #day = days[i]
        out -= y_unpackd[0][i, 0]*(torch.log(y_pred_unpackd[0][i, y_pred_unpackd[1][i]-1]) + torch.log(1-y_pred_unpackd[0][i, 0:y_pred_unpackd[1][i]-1]).sum())
        #uncensored_obss_pred[i, day-1]) - torch.log(-uncensored_obss_pred[i, 0:day] + 1).sum()
    return out


# Implements a BCE loss with internal computation of survival functions
def surv_BCE(y_pred_unpackd, y_unpackd):
    surv_func = torch.zeros(len(y_pred_unpackd[0]))
    for i in range(len(y_pred_unpackd[0])):
        surv_func[i] = (-y_pred_unpackd[0][i, 0:y_pred_unpackd[1][i]] + 1).prod()
    bce_loss = nn.BCELoss()
    out = bce_loss(surv_func, 1-y_unpackd[0][:, 0])
    return out


# Implements the convex combination of loss functions
def surv_loss(alpha, nll_loss, bce_loss):
    return alpha*nll_loss + (1-alpha)*bce_loss


# Computes the C-index based on hazard rates
def c_index(hazards, surv_times, event_status):
    hazard_res = 1-hazards
    return concordance_index(surv_times, hazard_res, event_observed=event_status)


if __name__ == '__main__':
    pass


