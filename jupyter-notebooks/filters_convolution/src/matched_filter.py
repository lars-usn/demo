# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:21:59 2025

@author: larsh
"""

# Illustration of the convolution operation, applied to FIR filtering
from math import pi
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets
from pathlib import Path

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Matched Filter Demo"


class Chirp:
    """Create and demonstrate linear chirp."""

    def __init__(self, create_widgets=False):

        # Normally fixed parameters
        self.sample_rate = 10e6
        self.chirp_duration = 200e-6  # Chirp duration
        self.window = "tukey"  # Tapering window
        self.window_par = 0.2  # Parameter to window function

        # Changed during runtime
        self.start_frequency = 125e3  # Start frequencvy
        self.end_frequency = 200e3  # End frequency
        self.reference_time = -200e-6  # Start of received signal
        self.noise_level = 0.0

        # No. of points in chirp
        n_chirp = int(self.chirp_duration * self.sample_rate)
        pad = 2
        self.n_pad = int(pad * n_chirp)  # No. of points to pad ends
        self.n_points = int(n_chirp + 2 * self.n_pad)

        self.t = np.arange(0, self.n_points) / self.sample_rate

        self.noise_base = self.create_noise()

        self.magnitude = False

        # Initialisation
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.scale_axes()

        self.update_pulse()
        self.update_received()
        self.update_multiplied()
        self.update_correlated()
        self.update_referenceline()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    @property
    def reference_index(self):
        """Calculate shifted start index."""
        return int(self.reference_time * self.sample_rate)

    @property
    def correlated_time(self):
        """Calculate time-vector for correlated signal."""
        return (
            signal.correlation_lags(
                self.n_points,
                self.n_points,
                mode="full",
            )
            / self.sample_rate
        )

    def create_noise(self):
        """
        Create random noise vector of unit amplitude

        Returns
        -------
        ndarray
            Noise vector
        """
        rng = np.random.default_rng()
        return rng.standard_normal(self.n_points)

    @property
    def chirp(self):
        """
        Create linear chirp.

        Returns
        -------
        ndarray
            Frequency sweep with envelope
        """
        n_points = int(self.chirp_duration * self.sample_rate)
        t = np.arange(0, n_points) / self.sample_rate
        mu = (self.end_frequency - self.start_frequency) / (
            2 * self.chirp_duration
        )
        psi = 2 * pi * (mu * t**2 + self.start_frequency * t)
        sweep = np.cos(psi)

        envelope = signal.windows.get_window(
            (self.window, self.window_par), len(sweep)
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
        return self.pulse + self.noise_level * self.noise_base

    @property
    def shifted_pulse(self):
        """
        Shift returned noisy pulse.

        Returns
        -------
        ndarray
            Noisy pulse shifted in time,
        """
        y = np.zeros_like(self.noisy_pulse)
        k = self.reference_index

        if k >= 0:
            y[k:] = self.noisy_pulse[:-k] if k else self.noisy_pulse
        else:
            y[:k] = self.noisy_pulse[-k:]

        return y

    @property
    def multiplied_pulses(self):
        """
        Shift returned noisy pulse.

        Returns
        -------
        ndarray
            Noisy pulse shifted in time,
        """
        return self.pulse * self.shifted_pulse

    @property
    def correlation_output(self):
        """
        Correlate original pulse with returned noisy pulse

        Returns
        -------
        ndarray
            Correlation result
        """
        corr = signal.correlate(
            self.pulse,
            self.noisy_pulse,
            mode="full",
        )
        return corr / np.max(np.abs(corr))

    def envelope(self, x):
        """Calculate envelope of signal x."""
        hx = signal.hilbert(x)
        return np.abs(hx)

    def _initialise_graphs(self):
        """
        Initialise graphs for signals and spectra.

        Returns
        -------
        ax : List of axis objects
            Axes where results are plotted
        """
        # Create figure and axes
        plt.close(FIGURE_NAME)

        n = 5
        fig, axes = plt.subplot_mosaic(
            [
                ["."] + ["pulse"] * n,
                ["."] + ["received"] * n,
                ["."] + ["multiplied"] * n,
                ["logo"] + ["correlated"] * n,
            ],
            figsize=(12, 6),
            layout="constrained",
            num=FIGURE_NAME,
        )

        # Define graphs for results
        self._create_logo(axes["logo"])

        t_us = self.t * 1e6
        zero_pulse = np.zeros_like(t_us)
        tc_us = self.correlated_time * 1e6
        zero_correlation = np.zeros_like(tc_us)

        graphs = {}
        (graphs["pulse_def"],) = axes["pulse"].plot(t_us, zero_pulse, "C0")
        (graphs["transmitted"],) = axes["received"].plot(
            t_us, zero_pulse, "C0"
        )
        (graphs["received"],) = axes["received"].plot(t_us, zero_pulse, "C1")
        (graphs["multiplied"],) = axes["multiplied"].plot(
            t_us, zero_pulse, "C0"
        )
        (graphs["correlated"],) = axes["correlated"].plot(
            tc_us, zero_correlation, "C0"
        )
        (graphs["envelope"],) = axes["correlated"].plot(
            tc_us, zero_correlation, color="C0", linestyle="solid"
        )
        graphs["referenceline"] = axes["correlated"].axvline(
            x=self.reference_time * 1e6, color="C1"
        )

        # Scale and format axes
        for name in ["pulse", "received", "multiplied", "correlated"]:
            axes[name].axhline(y=0, color="gray")

        axes["pulse"].set_title("Pulse $x(n)$")
        axes["received"].set_title("Shifted pulse $y(n+k)$")
        axes["multiplied"].set_title("Multiplied pulses. $x(n) y(n+k)$ ")
        axes["correlated"].set_title("Correlated pulses. $\sum x(n) y(n+k)$ ")
        axes["correlated"].set_xlabel(r"Time [$\mu$s] ")

        return fig, axes, graphs

    def update_pulse(self):
        """Update graph showing original pulse."""
        self.graphs["pulse_def"].set_ydata(self.pulse)
        self.graphs["transmitted"].set_ydata(self.pulse)

    def update_received(self):
        """Update graph showing received pulse."""
        self.graphs["received"].set_ydata(self.shifted_pulse)

    def update_multiplied(self):
        """Update graph showing product of transmitted and received pulses."""
        self.graphs["multiplied"].set_ydata(self.multiplied_pulses)

    def update_correlated(self):
        """Update correlation graph."""
        x = self.correlation_output
        envelope = self.envelope(x)

        if self.magnitude:
            x = np.abs(x)

        self.graphs["correlated"].set_ydata(x)
        self.graphs["envelope"].set_ydata(envelope)

    def update_referenceline(self):
        """Update line showing shifted pulse time."""
        t0 = self.reference_time * 1e6
        self.graphs["referenceline"].set_xdata([t0, t0])

    def scale_axes(self):
        """Set axes scales to fixed scales."""
        t_pulse_start = self.n_pad / self.sample_rate
        t_pad = 1.0 * self.chirp_duration
        t_min = t_pulse_start - self.chirp_duration
        t_max = t_pulse_start + self.chirp_duration + t_pad
        t_span = t_max - t_min

        tlim_us = np.array([t_min, t_max]) * 1e6

        for name in ["pulse", "received", "multiplied"]:
            self.axes[name].set(
                xlim=tlim_us,
                ylim=[-1.5, 1.5],
            )

        tspan_us = np.array([-t_span, t_span]) / 2 * 1e6
        self.axes["correlated"].set(
            xlim=tspan_us,
            ylim=[-1.5, 1.5],
        )

        return

    def _create_logo(self, ax):
        """
        Load logo file and place in specified axis.

        Parameters
        ----------
        ax : Axis object
            Axis where logo image is shown
        """
        ax.set_axis_off()

        try:
            base_path = Path(__file__).resolve().parent
        except NameError:
            # Running in Jupyter
            base_path = Path.cwd()

        logo_path = (base_path / ".." / "figs" / LOGOFILE).resolve()

        if logo_path.exists():
            img = mpimg.imread(logo_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.5,
                "USN",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def redraw(self, pulse=False):
        if pulse:
            self.update_pulse()

        self.update_received()
        self.update_multiplied()
        self.update_correlated()

        self.fig.canvas.draw_idle()

    def _start_frequency_change_callback(self, change):
        self.start_frequency = change["new"] * 1e3
        self.redraw(pulse=True)

    def _end_frequency_change_callback(self, change):
        self.end_frequency = change["new"] * 1e3
        self.redraw(pulse=True)

    def _noise_change_callback(self, change):
        self.noise_level = change["new"]
        self.redraw(pulse=False)

    def _shift_change_callback(self, change):
        self.reference_time = change["new"] * 1e-6
        self.update_referenceline()
        self.redraw(pulse=False)

    def _magnitude_change_callback(self, change):
        self.magnitude = change["new"]
        self.update_correlated()
        self.fig.canvas.draw_idle()

    # --- Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = "Matched Filter: Correlation of chirps (FM pulses)"
        title_widget = ipywidgets.Label(title, style=dict(font_weight="bold"))

        # Layouts definitions
        text_layout = {
            "continuous_update": False,
            # 'style': {'description_width': '120px'}
        }

        slider_layout = {
            "continuous_update": True,
            # 'style': {'description_width': '120px'}
        }

        checkbox_layout = {"style": {"description_width": "120px"}}

        # Individual widgets
        start_frequency_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            step=1.0,
            value=self.start_frequency / 1e3,
            description="Start freq. [kHz]",
            readout_format=".0f",
            **text_layout
        )

        start_frequency_widget.observe(
            self._start_frequency_change_callback,
            names="value",
        )

        end_frequency_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            step=1.0,
            value=self.end_frequency / 1e3,
            description="End freq. [kHz]",
            readout_format=".0f",
        )

        end_frequency_widget.observe(
            self._end_frequency_change_callback,
            names="value",
        )

        noise_widget = ipywidgets.BoundedFloatText(
            min=0.0,
            max=2.0,
            step=0.05,
            value=self.noise_level,
            description="Noise level",
            readout_format=".2f",
        )

        noise_widget.observe(
            self._noise_change_callback,
            names="value",
        )

        shift_widget = ipywidgets.FloatSlider(
            min=-250,
            max=250,
            step=1.0,
            value=self.reference_time * 1e6,
            description="Ref. position [$\mu$s]",
            readout_format=".0f",
        )

        shift_widget.observe(
            self._shift_change_callback,
            names="value",
        )

        magnitude_widget = ipywidgets.Checkbox(
            value=self.magnitude, description="Magnitude", **checkbox_layout
        )

        magnitude_widget.observe(
            self._magnitude_change_callback,
            names="value",
        )

        # Arrange in columns and lines
        for w in [
            start_frequency_widget,
            end_frequency_widget,
            noise_widget,
            shift_widget,
            magnitude_widget,
        ]:
            w.style.description_width = "120px"

        for w in [
            start_frequency_widget,
            end_frequency_widget,
            noise_widget,
            magnitude_widget,
        ]:
            w.layout.width = "220px"

        text_column = ipywidgets.VBox(
            [
                start_frequency_widget,
                end_frequency_widget,
                noise_widget,
                magnitude_widget,
            ]
        )

        text_column.layout = ipywidgets.Layout(width="300px")
        shift_widget.layout.width = "900px"

        widget_layout = ipywidgets.HBox(
            [text_column, shift_widget],
            layout=ipywidgets.Layout(width="100%", align_items="center"),
        )

        # Export as dictionary
        widget = {
            "start_frequency_widget": start_frequency_widget,
            "end_frequency_widget": end_frequency_widget,
            "shift_widget": shift_widget,
            "noise_widget": noise_widget,
            "magnitude_widget": magnitude_widget,
        }

        return widget_layout, widget
