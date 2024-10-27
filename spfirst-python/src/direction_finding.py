# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:02:03 2024

@author: larsh
"""
import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, exp     # For readability, also covered by numpy


def make_cos(A, f0, phase, duration):
    """Make a cosine-function from specified parameters.

    Parameters
    ----------
    A : float
        Amplitude
    f0: float
        Frequency [Hz]
    phase: float
        Phase [radians]
    duration: float
        Duration of signal [seconds}

    Returns
    -------
    x: 1D array of float
        Cosine-wave
    t: 1D array of float
        Time vactor in seconds
    """

    T0 = 1/f0
    dt = T0/32
    t = np.arange(0, duration, dt)
    x = A * np.cos(2 * pi * f0 * t + phase)

    return x, t


def make_summed_cos(fk, Xk, fs, duration, t_start=0):
    """Synthesize a signal as a sum of cosine waves

    Parameters
    ----------
    fk: 1D array of float
        Frequencies [Hz]
    Xk: 1D array of float
        Complex amplitudes (phasors)
    fs: float
        Sample rate [Samples/s]
    duration: float
        Duration of signal [s]
    t_start : float, optional
        Start time, first point of time-vector [s]

    fk and Xk must have the same lengths.

    Returns
    -------
    x: 1D array of float
        Signal as the sum of the frequency components
    t: 1D array of float
        Time vactor in seconds
    """

    n_frequencies = len(fk)
    if len(Xk) != n_frequencies:
        print("Frequency and amplitude vectors are of different lengths.")
        return -1

    dt = 1/fs
    t = np.arange(t_start, t_start+duration, dt)

    X = 0
    for k in range(n_frequencies):
        X = X + Xk[k] * np.exp(2j * pi * fk[k] * t)

    x = X.real

    return x, t
