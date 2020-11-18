# File name : basic_tools.py
# Author : Simon Bastide
# Encoding : utf-8
# Function : Common basic tools 
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from datetime import datetime

def get_date_iso():
    return (datetime.now().replace().isoformat()).split('T')[0]


def filter(data, sample_rate, low_pass = 10, order = 4):
    """Applique un filtre basique passe bas (butter, ordre 4) pour lisser le signal"""

    # TODO: Ameliorer la gestion des NaN. Pour le moment les NaN A la fin de l'enregistrement sont supprimé et les autres sont remplacés par 0.
    try:
        data = np.array(data)
        low_pass = low_pass/(sample_rate/2)
        b, a = signal.butter(order, low_pass, btype = 'lowpass')
        data_filtered = np.zeros(data.shape)
        if len(data.shape) > 1:
            #Supression des nan en fin de signal 
            while np.isnan(data[-1,:]).any():
                data = data[:-1,:]
                data_filtered = data_filtered[:-1,:]
            data = np.nan_to_num(data)
            for col in range(data.shape[1]):
                data_filtered[:,col] = signal.filtfilt(b, a, data[:,col])
        else:
            while np.isnan(data[-1]):
                data = data[:-1]
                data_filtered = data_filtered[:-1]
            data = np.nan_to_num(data)
            
            data_filtered = signal.filtfilt(b, a, data)
        if np.isnan(data_filtered).any():
            print("Output contain NaN")

    except ValueError:
        print("Value Error, output will be full of 0")
        data_filtered =  np.full(1000,0)
    return data_filtered
    


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def match(s1, s2):
    # Check if two string match allowing one character difference
    # https://stackoverflow.com/questions/25216328/compare-strings-allowing-one-character-difference
    # Solution from Marco Sulla
    ok = False

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if ok:
                return False
            else:
                ok = True

    return ok


# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def resample_non_uniform(x, y, new_sample_rate):
    """[summary]
    Fonction inspired from https://stackoverflow.com/a/20889651/13360654
    Args:
        x (numpy.array): non-uniform time (in s)
        y (numpy array): signal
    """
    f = interpolate.interp1d(x, y)
    sample_rate = len(x)/x[-1]
    new_sample_rate = 200
    num = int(np.ceil((len(y)*new_sample_rate)/sample_rate))
    xx = np.linspace(x[0], x[-1], num)
    
    return f(xx)
