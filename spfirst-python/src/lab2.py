"""Functions for Lab 2, TSE 2280."""
import numpy as np					# Handle signals as arrays
import matplotlib.pyplot as plt		# Show results as graphs and images
from math import pi, cos, sin, tan	# Mathematical functions on scalars
from cmath import exp, sqrt     	# Complex mathematical functions on scalars

from scipy.fft import fft, fftshift, fftfreq	# FFT and helper functions
from scipy import signal			# Signal processing functions

import sounddevice as sd			# Play NumPy array as sound

import lab1


def plot_spectrum(x, fs):
    """Plot Fourier coefficients of x.

    Parameters
    ----------
    x: Numpy array of floats
        Signal in time-domain
    fs: float
        Sample rate [Samples/s]

    Returns
    -------
    ax1: Axes object1D array of float
        Handle to the magnitude plot
    ax2: Axes object1D array of float
        Handle to the phase plot
    """

    n_samples = len(x)            # No. of samples in signal
    ft_x = fft(x)/n_samples       # Fourier coefficients, correctly scaled
    f = fftfreq(n_samples, 1/fs)  # Frequency vector
    f = fftshift(f)               # Move negative frequencies to start
    ft_x = fftshift(ft_x)

    # Plot Fourier coefficients
    fig = plt.figure(figsize=([16, 6]))				# Define figure for plots

    ax1 = fig.add_subplot(1, 2, 1)  # Subplot for magnitudes
    ax1.stem(f, np.abs(ft_x))		    # Magnitude of spectral components as stem-plot
    ax1.set(xlabel="Frequency [Hz]",
            ylabel="Magnitude")
    ax1.grid(True)

    ax2 = fig.add_subplot(1, 2, 2)  # Subplot for phase
    ax2.stem(f, np.angle(ft_x))		# Phase of spectral components as stem-plot
    ax2.set(xlabel="Frequency [Hz]",
            ylabel="Phase [radians]")
    ax2.grid(True)

    return ax1, ax2


def plot_spectrogram(x, t, fs, n_segment, f_max):
    """Plot spectrogram of signal x.

    Parameters
    ----------
    x: Numpy array of floats
        Signal in time-domain
    t: Numpy array of floats
        Time vector for x
    fs: float
        Sample rate [Samples/s]
    n_segmend: int
        No. of samples in segment for spctrogram calculation
    f_max: float
        Max. on frequency axis

    Returns
    -------
    ax: Axes object1D
        Handle to the spectrogram plot
    """
    # Configure spectrogram
    s_min = -40       # Minimum on the intensity plot. Lower values are 'black'

    # Calculate spectrogram
    f, t, sx = signal.spectrogram(x, fs, nperseg=n_segment, detrend=False)
    sx_db = 10*np.log10(sx/sx.max())   # Convert to dB

    # Plot spectrogram
    fig = plt.figure(figsize=(16, 6))  # Define figure for results

    ax = fig.add_subplot(1, 1, 1)
    sx_image = ax.pcolormesh(t, f, sx_db, vmin=s_min, cmap="viridis")

    ax.set(xlabel="Time [s]",
           ylabel="Frequency [Hz]",
           ylim=(0, f_max))

    fig.colorbar(sx_image, label="Magnitude [dB]")  # Colorbar for intensity

    return ax, sx_image


def make_beat(A, fc, df, fs, duration):
    """Synthesize a beat tone from two cosine waves.

    Parameters
    ----------
    A: List of floats
        Amplitudes of the two cosine waves
    fc: float
        Centre frequency [Hz]
    fc: float
        Frequency difference [Hz]
    fs: float
        Sample rate	[samples/s]
    duration
        Duration of signal [s]

    Returns
    -------
    x: 1D array of float
        Signal as the sum of the frequency components
    t: 1D array of float
        Time vector [seconds]
    """
    fk = [fc-df, fc+df]
    x, t = lab1.make_summed_cos(fk, A, fs, duration)

    return x, t


def make_chirp(f1, f2, fs, duration, phase=0):
    """Synthesize a beat tone from start and end frequencies.
    Parameters
    ----------
    f1: float
        Start frequency [Hz]
    f2: float
        End frequency [Hz]
    fs: float
        Sample rate	[Samples/s]
    duration
        Duration of signal [s]
    phase=0: float, optional
        Constant phase [radians]

    Returns
    -------
    x: 1D array of float
        Signal as the sum of the frequency components
    t: 1D array of float
        Time vector [seconds]
        mu: float
        Slope constant of chirp, mu
        """
    t = np.arange(0, duration, 1/fs)
    f0 = f1
    mu = (f2-f1)/(2*duration)

    psi = 2*pi*mu*t**2 + 2*pi*f0*t + phase

    x = np.cos(psi)

    return x, t, mu
