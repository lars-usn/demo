# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:21:59 2025

@author: larsh
"""

# Illustration of the convolution operation, applied to FIR filtering
from math import pi
import numpy as np
from scipy import signal
import ipywidgets
import matplotlib.pyplot as plt

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Matched Filter Demo"


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Chirp():
    """Create and demonstrate linear chirp."""

    def __init__(self):
        self.start_frequency = 125e3    # Start frequencvy
        self.end_frequency = 200e3      # End frequency
        self.sample_rate = 10e6
        self.chirp_dur = 200e-6  # Chirp duration
        self.window = 'tukey'    # Tapering window
        self.window_par = 0.2    # Parameter to window function

        self.start = -200e-6     # Start of received signal

        # No. of points in chirp
        n_chirp = int(self.chirp_dur * self.sample_rate)
        pad = 2
        self.n_pad = int(pad*n_chirp)          # No. of points to pad ends
        self.n_points = int(n_chirp + 2*self.n_pad)

        self.t = np.arange(0, self.n_points) / self.sample_rate

        rng = np.random.default_rng()
        self.noise_base = rng.standard_normal(self.n_points)
        self.noise_level = 0.0

        self.magnitude = False

        self.axes = self.initialise_graphs()
        self.scale_axes()
        self.widget = self._create_widgets()

        return

    @property
    def start_index(self):
        """Calculate shifted start index."""
        return int(self.start * self.sample_rate)

    @property
    def correlation_time(self):
        """Calculate time-vector for correlated signal."""
        return signal.correlation_lags(
            self.n_points,
            self.n_points) / self.sample_rate

    @property
    def noise(self):
        """
        Create random noise vector.

        Returns
        -------
        ndarray
            Noise vector
        """
        rng = np.random.default_rng()
        return self.noise_level * rng.standard_normal(self.n_points)

    @property
    def chirp(self):
        """
        Create linear chirp.

        Returns
        -------
        ndarray
            Frequency sweep with envelope
        """
        n_points = self.chirp_dur * self.sample_rate
        t = np.arange(0, n_points) / self.sample_rate
        mu = (self.end_frequency-self.start_frequency) / (2 * self.chirp_dur)
        psi = 2 * pi*(mu * t**2 + self.start_frequency * t)
        sweep = np.cos(psi)

        envelope = signal.windows.get_window(
            (self.window, self.window_par),
            len(sweep)
        )

        return envelope * sweep

    @property
    def pulse(self):
        """
        Create full signal as chirp with zeros padded.

        Returns
        -------
        ndarray
            Frequency sweep with padded ends

        """
        chirp = self.chirp
        idx = self.n_pad + np.arange(len(chirp))  # Shift index by n_pad

        pulse = np.zeros(self.n_points)
        pulse[idx] = chirp

        return pulse

    @property
    def noisy_pulse(self):
        """
        Create noisy signal

        Returns
        -------
        ndarray
            Frequency sweep with padded ends and noise

        """
        return self.pulse + self.noise

    @property
    def shifted_pulse(self):
        """
        Shift returned noisy pulse.

        Returns
        -------
        ndarray
            Noisy pulse shifted in time,
        """
        return np.roll(self.noisy_pulse, self.start_index)

    @property
    def correlated_pulses(self):
        """
        Correlate original pulse with returned noisy pulse

        Returns
        -------
        ndarray
            Correlation result
        """

        return signal.correlate(self.pulse, self.noisy_pulse)

    def initialise_graphs(self):
        """
        Initialise graphs for signals and spectra.

        Returns
        -------
        ax : List of axis objects
            Axes where results are plotted
        """
        plt.close(FIGURE_NAME)
        fig = plt.figure(
            figsize=[14, 6],
            constrained_layout=True,
            num=FIGURE_NAME,
        )

        axes = fig.subplots(4, 1)

        axes[0].set_title('Pulse $x(n)$')
        axes[1].set_title('Shifted pulse $y(n+k)$')
        axes[2].set_title('Product $x(n) y(n+k)$ ')
        axes[3].set_title(r'Correlation $\sum x(n) y(n+k)$ ')
        axes[3].set_xlabel(r'Time [$\mu$s] ')

        for ax in axes[0:3]:
            ax.set(ylim=(-2, 2))

        return axes

    def display(self):
        """Plot all signals and spectra."""
        # Clear old graphs, add indicators
        for ax in self.axes:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        ax = self.axes
        shifted_signal = np.roll(
            self.noisy_pulse,
            self.start_index
        )

        multiplied = self.pulse * shifted_signal

        t_us = self.t * 1e6
        ax[0].plot(t_us, self.pulse, 'C0')

        ax[1].plot(t_us, self.pulse, 'C0',
                   t_us, self.shifted_pulse, 'C1')

        ax[2].plot(t_us, multiplied, 'C0')

        for a in ax[0:4]:
            a.axhline(y=0, color='gray')

        # Cross-correlation
        correlated = self.correlated_pulses
        tc = self.correlation_time()

        if self.magnitude:
            correlated = abs(correlated)

        ax[3].plot(tc*1e6, correlated, color='C0')
        ax[3].axvline(x=self.start*1e6, color='C1')

        return

    def scale_axes(self):
        """Set axes to fixed scales."""
        t0 = self.n_pad / self.sample_rate
        t_pad = 1.0*self.chirp_dur
        t_min = t0 - t_pad
        t_max = t0 + self.chirp_dur + t_pad
        t_span = t_max - t_min

        tlim_us = np.array([t_min, t_max])*1e6
        for a in self.axes[0:3]:
            a.set(xlim=tlim_us)

        tspan_us = np.array([-t_span, t_span])/2*1e6
        self.axes[3].set(xlim=tspan_us)

        return

    def interact(self, start=None, start_frequency=None, end_frequency=None,
                 noise_level=None, magnitude=None):
        if start is not None:
            self.start = 1e-6*start
        if start_frequency is not None:
            self.start_frequency = 1e3*start_frequency
        if end_frequency is not None:
            self.end_frequency = 1e3*end_frequency
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
            'layout': ipywidgets.Layout(width='90%'),
            'style': {'description_width': '60%'}}

        slider_layout = {
            'continuous_update': True,
            'layout': ipywidgets.Layout(width='60%'),
            'style': {'description_width': '15%'}}

        checkbox_layout = {
            'layout': ipywidgets.Layout(width='20%'),
            'style': {'description_width': '10%'}}

        # Individual widgets
        start_frequency_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            value=self.start_frequency/1e3,
            description='Start [kHz]',
            readout_format='.0f',
            **text_layout)

        # start_frequency_widget.observe(
        #     self._frequency_change_callback,
        #     names="value",
        # )

        end_frequency_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            value=self.end_frequency/1e3,
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
            min=-250,
            max=250,
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
        widget_f_layout = ipywidgets.VBox([start_frequency_widget,
                                           end_frequency_widget,
                                           noise_widget])

        widget_par_layout = ipywidgets.VBox([noise_widget,
                                             magnitude_widget])

        widget_layout = ipywidgets.HBox([widget_f_layout,
                                         shift_widget,
                                         magnitude_widget])

        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'start_frequency_widget': start_frequency_widget,
                  'end_frequency_widget': end_frequency_widget,
                  'shift_widget': shift_widget,
                  'noise_widget': noise_widget,
                  'magnitude_widget': magnitude_widget
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
