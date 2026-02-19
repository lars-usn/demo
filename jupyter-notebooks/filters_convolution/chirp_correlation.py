# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:21:59 2025

@author: larsh
"""

# Illustration of the convolution operation, applied to FIR filtering
from math import pi
import numpy as np
from scipy.signal import convolve
from scipy import signal
import ipywidgets
import matplotlib.pyplot as plt


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Chirp():
    """Create and demonstrate linear chirp."""

    def __init__(self):
        self.f1 = 125e3          # Start frequencvy
        self.f2 = 200e3          # End frequency
        self.fs = 10e6
        self.chirp_dur = 100e-6  # Chirp duration

        self.start = -150e-6

        pad = 2
        n_chirp = int(self.chirp_dur*self.fs)  # No. of points in chirp
        self.n_pad = int(pad*n_chirp)               # No. of points to pad ends
        self.n_pts = int(n_chirp + 2*self.n_pad)

        self.t = np.arange(0, self.n_pts) / self.fs

        rng = np.random.default_rng()
        self.noise_base = rng.standard_normal(self.n_pts)
        self.noise_level = 0.0

        self.magnitude = False

        self.ax = self.initialise_graphs()
        self.scale_axes()
        self.widget = self._create_widgets()

        return

    def chirp(self):
        """Create linear chirp."""
        n_p = self.chirp_dur * self.fs
        ts = np.arange(0, n_p) / self.fs
        mu = (self.f2-self.f1) / (2 * self.chirp_dur)
        psi = 2 * pi*(mu*ts**2 + self.f1*ts)
        p = np.sin(psi)

        return p

    def signal(self):
        """Signal, chirp with zeres padded."""
        c = self.chirp()

        idx = self.n_pad + np.arange(len(c))

        s = np.zeros(self.n_pts)
        s[idx] = c

        return s

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra.

        Returns
        -------
        ax : List of axis objects
            Axes where results are plotted
        """
        plt.close('all')
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num="Matched Filter Demo")

        ax = fig.subplots(4, 1)

        ax[0].set_title('Pulse $x(n)$')
        ax[1].set_title('Shifted pulse $y(n+k)$')
        ax[2].set_title('Product $x(n) y(n+k)$ ')
        ax[3].set_title('Correlation $\sum x(n) y(n+k)$ ')
        ax[3].set_xlabel(r'Time [$\mu$s] ')

        for a in ax[0:3]:
            a.set(ylim=(-2, 2))
   
        return ax

    def display(self):
        """Plot all signals and spectra.

        Arguments
        ---------
        s : Signal object
            Input signal
        h : Signal object
            Filter impulse response
        n : int
            Sample position to show in plot
        """
        # Clear old graphs, add indicators
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        ax = self.ax
        n_start = int(self.start*self.fs)

        s = self.signal()    # Clean signal
        rng = np.random.default_rng()
        noise = self.noise_level * rng.standard_normal(len(s))

        sn = s + noise                      # Signal with noise
        sc = np.roll(sn, n_start) + noise   # Shifted, noisy signal
        scorr = s*sc  # Correlation vector, clean to noisy shifted signal

        t_us = self.t*1e6     # Time vector in us
        ax[0].plot(t_us, s, 'C0')
        ax[1].plot(t_us, s, 'C0')
        ax[1].plot(t_us, sc, 'C1')
        ax[2].plot(t_us, scorr, 'C0')

        for a in ax[0:4]:
            a.axhline(y=0, color='gray')

        # Cross-correlation, clean with noisy signal
        ac = signal.correlate(s, sn)
        n = len(s)
        tc = signal.correlation_lags(n, n) / self.fs
        if self.magnitude:
            ac = abs(ac)
        ax[3].plot(tc*1e6, ac, color='C0')
        ax[3].axvline(x=self.start*1e6, color='C1')

        return 0

    def scale_axes(self):
        t0 = self.n_pad / self.fs
        t_pad = 1.0*self.chirp_dur
        t_min = t0 - t_pad
        t_max = t0 + self.chirp_dur + t_pad
        t_span = t_max - t_min

        tlim_us = np.array([t_min, t_max])*1e6
        for a in self.ax[0:3]:
            a.set(xlim=tlim_us)

        tspan_us = np.array([-t_span, t_span])/2*1e6
        self.ax[3].set(xlim=tspan_us)

        return

    def interact(self, start=None, f1=None, f2=None, 
                 noise_level=None, magnitude=None):
        if start is not None:
            self.start = 1e-6*start
        if f1 is not None:
            self.f1 = 1e3*f1
        if f2 is not None:
            self.f2 = 1e3*f2
        if noise_level is not None:
            self.noise_level = noise_level
        if magnitude is not None:
            self.magnitude = magnitude

        self.display()

        return

    # --- Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = 'Matched Filter: Correlation of chirps (FM pulses)'
        title_widget = ipywidgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': ipywidgets.Layout(width='10%'),
            'style': {'description_width': '50%'}}

        slider_layout = {
            'continuous_update': True,
            'layout': ipywidgets.Layout(width='60%'),
            'style': {'description_width': '15%'}}

        checkbox_layout = {
            'layout': ipywidgets.Layout(width='20%'),
            'style': {'description_width': '50%'}}

        # Individual widgats
        f1_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            value=self.f1/1e3,
            description='Start [kHz]',
            readout_format='.0f',
            **text_layout)

        f2_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            value=self.f2/1e3,
            description='End [kHz]',
            readout_format='.0f',
            **text_layout)

        noise_widget = ipywidgets.BoundedFloatText(
            min=0.0,
            max=2.0,
            step=0.01,
            value=self.noise_level,
            description='Noise level',
            readout_format='.2f',
            **text_layout)

        shift_widget = ipywidgets.FloatSlider(
            min=-150,
            max=150,
            step=0.5,
            value=self.start*1e6,
            description='Ref. position [$\mu$s]',
            readout_format='.1f',
            **slider_layout)

        magnitude_widget = ipywidgets.Checkbox(
            value=self.magnitude,
            description='Magnitude',
            **checkbox_layout)

        # Arrange in columns and lines
        widget_layout = ipywidgets.HBox([f1_widget,
                                         f2_widget,
                                         shift_widget,
                                         noise_widget,
                                         magnitude_widget])
        
        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'f1_widget': f1_widget,
                  'f2_widget': f2_widget,
                  'shift_widget': shift_widget,
                  'noise_widget': noise_widget,
                  'magnitude_widget': magnitude_widget
                 }

        w = WidgetLayout(widget_layout, widget)

        return w
