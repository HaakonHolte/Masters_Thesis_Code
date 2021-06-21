### THIS FILE CONTAINS GENERAL HELPER FUNCTIONS FOR THE THESIS ###

import torch
import pandas as pd
import numpy as np


# Allows for indexing a dictionary with a list or array
def dict_indexer(dictt, arrayy):
    out_array = np.array([], dtype='int64')
    for i in arrayy:
        out_array = np.append(out_array, dictt[i])
    return out_array


# Convert seconds into hours, minutes and seconds
def convert_seconds(seconds):
    hours = int(seconds // 3600)
    temp = seconds - hours*3600
    minutes = int(temp // 60)
    secs = round(temp - minutes*60)
    return hours, minutes, secs


# Index mth row of array1 with mth element of array2
def array_indexer(array1, array2, out_type='tensor'):
    if out_type=='tensor':
        out = torch.ones(len(array2), dtype=torch.float32)
        for i in range(len(array2)):
            out[i] = array1[i, array2[i]-1] # array2[i]-1
    elif out_type=='numpy':
        out = np.ones(len(array2), dtype=np.float32)
        for i in range(len(array2)):
            out[i] = array1[i, array2[i]-1] # array2[i]-1
    else:
        print("Only 'tensor' and 'numpy' are supported arguemtns for out_type.")
        return 0
    return out


# Slice mth row of array1 with the previous n_elements from index given by mth element of array2
def array_slicer(array1, array2, n_elements, out_type='tensor'):
    if out_type=='tensor':
        out = torch.ones(size=(len(array2), n_elements), dtype=torch.float32)
        for i in range(len(array2)):
            out[i] = array1[i, array2[i]-(n_elements+1):array2[i]-1] # array2[i]-1
    elif out_type=='numpy':
        out = np.ones((len(array2), n_elements), dtype=np.float32)
        for i in range(len(array2)):
            out[i] = array1[i, array2[i]-(n_elements+1):array2[i]-1] # array2[i]-1
    else:
        print("Only 'tensor' and 'numpy' are supported arguemtns for out_type.")
        return 0
    return out


# Compute exponentially weighted average of array with decay factor alpha
def ewa(alpha, array):
    if len(array) > 0:
        out = (1-alpha)*array[-1] + alpha*ewa(alpha, array[:-1])
        return out
    else:
        return 0

if __name__ == '__main__':
    pass