import copy

import numpy as np
import pywt


def separate_signal_dwt(f0_contour, wavelet_type="db10", split_boundary=None):
    """
    Split signal to low- and high-frequency signal.

    Args:
        f0_contour(np.array)    : Input signal (1-dimensional signal)
        wavelet_type(str)       : Type of mother wavelet function
        split_boundary(int)     : Cutoff level of DWT

    Returns:
        low_signal(np.array)    : Low-frequency signal
        high_signal(np.array)   : High-frequency signal
    """

    # Separate a signal by DWT into low- and high-frequency coefficients
    coefs = pywt.wavedec(
        f0_contour,
        wavelet_type,
        level=split_boundary,
    )
    low_coef = copy.deepcopy(coefs)
    high_coef = copy.deepcopy(coefs)

    # Reconstruct low-frequency contour
    low_signal = np.zeros(len(f0_contour))

    # Masking high frequency information
    for j in range(1, split_boundary + 1):
        low_coef[j] = np.zeros_like(low_coef[j])
    low_signal = pywt.waverec(low_coef, wavelet_type)

    # Postprocess different sequence length caused by dwt
    if len(low_signal) != len(f0_contour):
        low_signal = low_signal[:-1]

    # Reconstruct high-frequency contour
    high_signal = np.zeros(len(f0_contour))
    high_coef[0] = np.zeros_like(high_coef[0])
    high_signal = pywt.waverec(high_coef, wavelet_type)

    # Postprocess different sequence length caused by dwt
    if len(high_signal) != len(f0_contour):
        high_signal = high_signal[:-1]

    return low_signal, high_signal
