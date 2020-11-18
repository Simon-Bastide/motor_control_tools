"""This module contains tools for emg processing"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy import stats


def filt_rect(emg, sample_rate, high_pass = 20, low_pass = 450, order = 4):
    """Rectify the signal and apply a band-pass butterworth filter

    This function is made to clean electromygraphie (emg) datas

    Args:
        emg (numpy.array): The emg signal
        sample_rate (int): Sample rate of the signal
        high_pass (int, optional): High frequency of the band-pass filter. Defaults to 20.
        low_pass (int, optional): Low frequency of the band-pass filter. Defaults to 450.
        order (int, optional): Order of the band-pass filter. Defaults to 4.

    Returns:
        numpy.array: The rectified and filtered signal
    """
    emg = emg[~np.isnan(emg)]
    high_pass = high_pass/(sample_rate/2) # cut-off frequency/Nyquist frequency
    low_pass = low_pass/(sample_rate/2)
    b, a = signal.butter(order, [high_pass, low_pass], btype = 'bandpass')
    emg_correctmean = emg - np.mean(emg)
    if 3 * max(len(a), len(b)) < len(emg_correctmean):
        emg_filt_rect = abs(signal.filtfilt(b, a, emg_correctmean))
    else:
        emg_filt_rect =  np.full(len(emg),0)
        # TODO: Warning signal here
        # Manage this case
    return emg_filt_rect

def envelope(emg, sample_rate, low_pass = 10, order = 5):
    """Apply a low pass filter to get the enveloppe of the emg signal

    Args:
        emg (numpy.array): The emg signal
        sample_rate (int): Sample rate of the signal
        low_pass (int, optional): Low frequency of the low-pass filter. Defaults to 10.
        order (int, optional): Order of the band-pass filter. Defaults to 4.

    Returns:
        numpy.array: filtered signal
    """
    # TODO: Change the name of the function --> filter. Option for params for emg enveloppe.
    low_pass = low_pass/(sample_rate/2)
    b, a = signal.butter(order, low_pass, btype = 'lowpass')
    if 3 * max(len(a), len(b)) < len(emg):
        emg_envelope = signal.filtfilt(b, a, emg)
    else:
        emg_envelope = np.full(100,0)
    return emg_envelope

def simple_params(emg, name, sample_rate):
    """Return simple parameters on emg signal. Rectified and filtered emg signal should be
    given in input (Use filt_rect() function before).

    parameters computed : 
        -maximum of the signal
        -median of the signal
        -maximum of the enveloppe
        -root mean square (rms) of the enveloppe

    Args:
        emg (numpy.array): The emg signal 
        name (string): emg's name. params will be return with the label 'name_param'
        sample_rate (int): Sample rate of the signal

    Returns:
        dict: dictionary containing computed parameters.
    """
    emg_env = envelope(emg, sample_rate)

    emg_max = np.max(emg)
    emg_med = np.median(emg)
    emg_max_env = np.max(emg_env)
    rms = np.sqrt(np.mean(emg_env**2))

    return ({'muscle' : name, 'max':emg_max, 'med':emg_med,\
              'max_env':emg_max_env, 'rms': rms})

def get_emgs_maximum_activation(subject_id, parameters):
    """Run through a subject's data to obtain the rectified signal maximum activation of each
    muscle during the experiment.

    Args:
        subject_id (string): folder name of the subject. 'S1' for example
        parameters (module): Module parameters. obtained with an 'import parameters'

    Returns:
        dict: Dictionary with maximum activation for each muscle
    """
    max_subject = []
    for condition in parameters.conditions:
        for block in range(1,parameters.block_numbers.get(condition)+1):
            file = '_'.join([subject_id, condition, str(block)])
            emg_file_path = os.path.join(parameters.data_path,subject_id, file + "_emg.csv")
            if os.path.isfile(emg_file_path):
                trials_sep = pd.read_csv(os.path.join(parameters.data_path,subject_id,"trials_separators_" + file + ".csv")).astype(int)
                blocks_sep = pd.read_csv(os.path.join(parameters.data_path,subject_id,"blocks_separators_" + file + ".csv"))
                df_emg = pd.read_csv(emg_file_path)
                for mov_n, (start, stop) in enumerate(zip(trials_sep.sep_emg, trials_sep.sep_emg[1:])):
                    start = start + blocks_sep.sep_kin[0]
                    stop = stop + blocks_sep.sep_kin[0]
                    df_emg_mov = df_emg.iloc[start:stop]
                    df_emg_mov = df_emg_mov.dropna()
                    if not df_emg_mov.empty:
                        emg_filt_rect = df_emg_mov.apply(filt_rect, args = [parameters.sample_rate_emg])
                        max_subject.append(emg_filt_rect.max())
    max_df = pd.DataFrame(max_subject)
    cleaned_max_df = pd.DataFrame()
    # TODO Remplacer par un .apply()
    for colName, colData in max_df.iteritems():
        outliers = np.abs(stats.zscore(colData)) < 1
        data_without_outliers = colData[outliers]
        cleaned_max_df = cleaned_max_df.assign(**{colName : data_without_outliers})
    return dict(cleaned_max_df.drop(columns = 'time').max())


def hampel(vals_orig, k=10, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    # from - https://stackoverflow.com/a/51731332/13360654
    # see also - https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d 

    #Make copy so original not edited
    vals = vals_orig.copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)

def is_valid(emg):
    # False si une partie du signal est constant
    if np.sum(np.diff(emg)*2000 == 0) < 200:
        valid = True
    elif all(np.diff(emg)*2000 < 10e-6):
        valid = False
        print("Constant signal")
    elif any(np.diff(emg)*2000 > 0):
        valid = False
        print("A part of the signal is constant")

    return valid