# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:21:59 2025

@author: larsh
"""

# Illustration of matched filter using cross-correlation of linear chirps
from math import pi
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets
from pathlib import Path

LOGOFILE = "usn-logo-purple.png"
FIGURE_NAME = "Matched Filter Demo"


LINEFORMAT = {
    "pulse": {
        "color": "C0",
        "linestyle": "solid",
        "linewidth": 1.0,
    },
    "received": {
        "color": "C1",
        "linestyle": "solid",
        "linewidth": 1.0,
    },
    "correlated": {
        "color": "C0",
        "linestyle": "solid",
        "linewidth": 1.0,
    },
    "envelope": {
        "color": "C0",
        "linestyle": "solid",
        "linewidth": 1.0,
    },
}


class Chirp:
    """Create and demonstrate linear chirp."""

    def __init__(self, create_widgets=False):

        # Normally fixed parameters
        self.sample_rate = 10e6
        self.chirp_duration = 200e-6
        self.window = "tukey"
        self.window_par = 0.2

        # Changed during runtime
        self.start_frequency = 125e3
        self.end_frequency = 200e3
        self.selected_lag = -200e-6
        self.noise_level = 0.0
        self.magnitude = False

        # No. of points in chirp
        n_chirp = int(self.chirp_duration * self.sample_rate)
        pad = 2
        self.n_pad = int(pad * n_chirp)  # No. of points to pad ends
        self.n_points = int(n_chirp + 2 * self.n_pad)

        self.t = np.arange(0, self.n_points) / self.sample_rate
        self.noise_base = self.create_noise()

        # Initialisation
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.scale_axes()

        self.update_referenceline()
        self.redraw(
            pulse_changed=True,
            correlation_changed=True,
        )

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    @property
    def lag_index(self):
        """Calculate shifted start index."""
        return int(self.selected_lag * self.sample_rate)

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

    def shift_signal(self, x, k):
        """
        Shift signal in time

        Parameters
        ----------
        x : ndarray
            Signal to shift
        k : int
            No. of sample to shift pulse, positive or negative

        Returns
        -------
        ndarray
            Signal x shifted k samples
        """
        y = np.zeros_like(x)

        if abs(k) >= len(y):
            return y

        if k > 0:
            y[k:] = x[:-k]
        elif k < 0:
            y[:k] = x[-k:]
        else:
            y[:] = x

        return y

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

        n_columns = 6
        fig, axes = plt.subplot_mosaic(
            [
                ["."] + ["pulse"] * n_columns,
                ["."] + ["received"] * n_columns,
                ["."] + ["multiplied"] * n_columns,
                ["logo"] + ["correlated"] * n_columns,
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
        (graphs["pulse_def"],) = axes["pulse"].plot(
            t_us,
            zero_pulse,
            **LINEFORMAT["pulse"],
        )
        (graphs["transmitted"],) = axes["received"].plot(
            t_us,
            zero_pulse,
            **LINEFORMAT["pulse"],
        )
        (graphs["received"],) = axes["received"].plot(
            t_us,
            zero_pulse,
            **LINEFORMAT["received"],
        )
        (graphs["multiplied"],) = axes["multiplied"].plot(
            t_us,
            zero_pulse,
            **LINEFORMAT["correlated"],
        )
        (graphs["correlated"],) = axes["correlated"].plot(
            tc_us,
            zero_correlation,
            **LINEFORMAT["correlated"],
        )
        (graphs["envelope"],) = axes["correlated"].plot(
            tc_us,
            zero_correlation,
            **LINEFORMAT["envelope"],
        )
        graphs["referenceline"] = axes["correlated"].axvline(
            x=self.selected_lag * 1e6,
            **LINEFORMAT["received"],
        )

        # Scale and format axes
        for name in ["pulse", "received", "multiplied", "correlated"]:
            axes[name].axhline(y=0, color="gray")
            axes[name].grid(True, axis="x")
            if name != "correlated":
                axes[name].tick_params(labelbottom=False)

        axes["pulse"].set_title("Pulse", loc="left")
        axes["pulse"].set_title("$x(n)$", loc="right")
        axes["received"].set_title("Shifted pulse", loc="left")
        axes["received"].set_title("$y(n+k)$ and $x(n)$", loc="right")
        axes["multiplied"].set_title("Multiplied pulses", loc="left")
        axes["multiplied"].set_title("$x(n) y(n+k)$ ", loc="right")
        axes["correlated"].set_title("Correlated pulses", loc="left")
        axes["correlated"].set_title(r"$\sum x(n) y(n+k)$ ", loc="right")
        axes["correlated"].set_xlabel(r"Time [$\mu$s] ")

        return fig, axes, graphs

    def update_pulse(self, pulse):
        """Update graphs showing the original pulse."""
        self.graphs["pulse_def"].set_ydata(pulse)
        self.graphs["transmitted"].set_ydata(pulse)

    def update_received(self, shifted_pulse):
        """Update graph showing the shifted received pulse."""
        self.graphs["received"].set_ydata(shifted_pulse)

    def update_multiplied(self, multiplied_pulses):
        """Update graph showing the pointwise product."""
        self.graphs["multiplied"].set_ydata(multiplied_pulses)

    def update_correlated(self, pulse, noisy_pulse):
        """Update the correlation graph."""
        corr = signal.correlate(
            pulse,
            noisy_pulse,
            mode="full",
        )
        corr = corr / np.sum(pulse**2)
        envelope = self.envelope(corr)

        plotted_corr = np.abs(corr) if self.magnitude else corr
        self.graphs["correlated"].set_ydata(plotted_corr)
        self.graphs["envelope"].set_ydata(envelope)

    def update_referenceline(self):
        """Update line showing shifted pulse time."""
        t0 = self.selected_lag * 1e6
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

    def redraw(self, pulse_changed=False, correlation_changed=False):
        pulse = self.pulse
        noisy_pulse = pulse + self.noise_level * self.noise_base
        shifted_pulse = self.shift_signal(noisy_pulse, self.lag_index)

        if pulse_changed:
            self.update_pulse(pulse)

        self.update_received(shifted_pulse)
        self.update_multiplied(pulse * shifted_pulse)

        if correlation_changed:
            self.update_correlated(pulse, noisy_pulse)

        self.fig.canvas.draw_idle()

    def _start_frequency_change_callback(self, change):
        self.start_frequency = change["new"] * 1e3
        self.redraw(
            pulse_changed=True,
            correlation_changed=True,
        )

    def _end_frequency_change_callback(self, change):
        self.end_frequency = change["new"] * 1e3
        self.redraw(
            pulse_changed=True,
            correlation_changed=True,
        )

    def _noise_change_callback(self, change):
        self.noise_level = change["new"]
        self.redraw(correlation_changed=True)

    def _shift_change_callback(self, change):
        self.selected_lag = change["new"] * 1e-6
        self.update_referenceline()
        self.redraw()

    def _magnitude_change_callback(self, change):
        self.magnitude = change["new"]
        self.redraw(correlation_changed=True)

    # --- Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = "Matched Filter: Correlation of chirps (FM pulses)"
        title_widget = ipywidgets.Label(title, style=dict(font_weight="bold"))

        # Individual widgets
        start_frequency_widget = ipywidgets.BoundedFloatText(
            min=10,
            max=300,
            step=1.0,
            value=self.start_frequency / 1e3,
            description="Start freq. [kHz]",
            readout_format=".0f",
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
            value=self.selected_lag * 1e6,
            description=r"Ref. position [$\mu$s]",
            readout_format=".0f",
        )

        shift_widget.observe(
            self._shift_change_callback,
            names="value",
        )

        magnitude_widget = ipywidgets.Dropdown(
            options=[("Signed", False), ("Magnitude", True)],
            value=False,
            description="Correlation",
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

        frequency_column = ipywidgets.VBox(
            [
                start_frequency_widget,
                end_frequency_widget,
            ]
        )

        extras_column = ipywidgets.VBox(
            [
                noise_widget,
                magnitude_widget,
            ]
        )

        frequency_column.layout = ipywidgets.Layout(width="230px")
        extras_column.layout = ipywidgets.Layout(width="230px")
        shift_widget.layout.width = "850px"

        widget_layout = ipywidgets.HBox(
            [frequency_column, shift_widget, extras_column],
            layout=ipywidgets.Layout(width="100%", align_items="center"),
        )

        widget_layout = ipywidgets.VBox(
            [title_widget, widget_layout],
            layout=ipywidgets.Layout(
                width="100%",
            ),
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
