"""Illustrate aliasing in the time- and frequency domains."""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.patches import Rectangle

LINEFORMATS = {
    "background": {"linestyle": "-", "linewidth": 0.8},
}

COLORS = {
    "original": "C0",
    "sampled": "C1",
    "reconstructed": "C2",
    "aliased": "C3",
    "nyquist": "C2",
}


class FrequencyAliasSignal:
    """Demonstration of aliasing in the time- and frequency domains.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self):
        """Initialise signal."""
        self.freq = 12.0
        self.phase = 0.0
        self.sample_rate = 20.0
        self.alias_indices = self._create_aliases(10)
        self.t_max = 15 / self.sample_rate  # Length of time axis, fixed
        self.display_original = True
        self.display_reconstructed = False

        self.fig, self.ax_time, self.ax_freq, self.graphs = (
            self._initialise_graphs()
        )
        self.update_titles()

        self.widget_layout = self._create_widgets()

    def _create_aliases(self, n):
        """Create alias indices. Principal alias, 0, is omitted."""
        m = np.arange(1, n)
        return np.sort(np.concatenate((-m, m)))

    # Time vectors
    @property
    def sample_interval(self):
        """Sample time."""
        return 1 / self.sample_rate

    @property
    def t_in(self):
        """Original time vector, set to 100 samples per cycle."""
        return np.linspace(0, self.t_max, 300)

    @property
    def t_sampled(self):
        """Get sampled time vector, including entire time vector."""
        return np.arange(
            0,
            self.t_max + self.sample_interval / 2,
            self.sample_interval,
        )

    # Signals
    @property
    def original(self):
        """Original signal."""
        return np.cos(2 * pi * self.freq * self.t_in + self.phase)

    @property
    def sampled(self):
        """Calculate sampled signal."""
        return np.cos(2 * pi * self.freq * self.t_sampled + self.phase)

    @property
    def reconstructed(self):
        """Reconstructed signal, using principal alias."""
        return np.cos(2 * pi * self.alias_freq * self.t_in + self.phase)

    # Frequencies
    @property
    def alias_freqs(self):
        """Find unique alias frequencies."""
        fp = self.freq + self.alias_indices * self.sample_rate
        fn = -fp  # Add negative aliases
        return np.sort(np.append(fn, fp))

    @property
    def alias_freq(self):
        """Return principal alias, inside the Nyquist interval."""
        fs = self.sample_rate
        return (self.freq + fs / 2) % fs - fs / 2

    # Update plots
    def _update_stem_plot(self, graph, x, y, base=0):
        """Update values on a stem-plot.

        Parameters
        ----------
        graph : StemContainer
            Plot values returned from stem-command
        x : 1D array of float
            x-values
        y : 1D array of float
            y-values
        """
        markerline, stemlines, baseline = graph

        baseline.set_data(x, base * np.ones_like(y))
        markerline.set_data(x, y)
        stemlines.set_segments(
            [[[xi, base], [xi, yi]] for xi, yi in zip(x, y)]
        )

    def update_graphs(self):
        """Update values of continuous plots."""
        for graph in self.graphs["continuous"][:2]:
            graph.set_data(self.t_in, self.original)

        for graph in self.graphs["continuous"][2:]:
            graph.set_data(self.t_in, self.reconstructed)

        self._update_stem_plot(
            self.graphs["sampled"],
            self.t_sampled,
            self.sampled,
        )

        for graph in self.graphs["spectra"][:2]:
            self._update_stem_plot(
                graph,
                [-self.freq, self.freq],
                np.ones(2),
            )

        self._update_stem_plot(
            self.graphs["spectra"][2],
            self.alias_freqs,
            np.ones(len(self.alias_freqs)),
        )

        self._update_stem_plot(
            self.graphs["spectra"][3],
            [-self.alias_freq, self.alias_freq],
            np.ones(2),
        )

        fn = self.sample_rate / 2
        for rect in self.graphs["nyquist"]:
            rect.set_x(-fn)
            rect.set_width(2 * fn)

        self.update_titles()

        self.fig.canvas.draw_idle()

    def update_titles(self):
        """Update graph titles with new values."""
        self.ax_time[0].set_title(f"Frequency = {self.freq:.1f} Hz")
        self.ax_time[1].set_title(
            f"Sampling at {self.sample_rate:.1f} samples/s"
        )
        self.ax_time[2].set_title(
            f"Reconstructed frequency = {self.alias_freq:.1f} Hz"
        )

    # === Widget callbacks ================================================
    def _phase_change_callback(self, change):
        """Update phase from widget."""
        self.phase = np.radians(change["new"])
        self.update_graphs()

    def _sample_rate_change_callback(self, change):
        """Update sample rate from widget."""
        self.sample_rate = change["new"]
        self.update_graphs()

    def _display_original_change_callback(self, change):
        """Show or hide the original signal in the comparison panel."""
        self.display_original = change["new"]
        self.graphs["continuous"][1].set_visible(self.display_original)
        self.fig.canvas.draw_idle()

    def _display_reconstructed_change_callback(self, change):
        """Show or hide the reconstructed signal in the comparison panel."""
        self.display_reconstructed = change["new"]
        self.graphs["continuous"][2].set_visible(self.display_reconstructed)
        self.fig.canvas.draw_idle()

    def _frequency_change_callback(self, change):
        """Update frequency from widget."""
        self.freq = change["new"]
        self.update_graphs()

    # === Create plots ====================================================
    def _create_continuous_plots(self, axes):
        """Create and format graphs for continuous curves."""

        graphs = []
        for ax in axes[:2]:
            (graph,) = ax.plot(
                self.t_in,
                self.original,
                color=COLORS["original"],
                **LINEFORMATS["background"],
            )
            graphs.append(graph)

        for ax in axes[1:]:
            (graph,) = ax.plot(
                self.t_in,
                self.reconstructed,
                color=COLORS["reconstructed"],
                **LINEFORMATS["background"],
            )
            graphs.append(graph)

        return graphs

    def _create_spectral_plots(self, axes):
        """Create frequency-domain plots."""
        graphs = []

        for ax in axes[:2]:
            graphs.append(
                ax.stem(
                    [-self.freq, self.freq],
                    np.ones(2),
                    linefmt=COLORS["original"],
                )
            )

        graphs.append(
            axes[1].stem(
                self.alias_freqs,
                np.ones(len(self.alias_freqs)),
                linefmt=COLORS["sampled"],
            )
        )

        graphs.append(
            axes[2].stem(
                [-self.alias_freq, self.alias_freq],
                np.ones(2),
                linefmt=COLORS["reconstructed"],
            )
        )
        return graphs

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""

        # --- Create figure and axes ---
        plt.close("all")
        fig = plt.figure(
            figsize=(10, 4),
            constrained_layout=True,
            num="Aliasing Demo",
        )

        n_plots = 3

        ax_time = [fig.add_subplot(2, n_plots, k + 1) for k in range(n_plots)]
        ax_freq = [
            fig.add_subplot(2, n_plots, k + 1 + n_plots)
            for k in range(n_plots)
        ]

        # --- Set axis scales ---
        for ax in ax_time:
            ax.set(
                xlim=(0, self.t_max),
                ylim=(-1.1, 1.1),
                xlabel="Time [s]",
            )

        fs_scale = 2 * self.sample_rate
        for ax in ax_freq:
            ax.set(
                xlim=(-fs_scale, fs_scale),
                ylim=(0, 1.1),
                xlabel="Frequency [Hz]",
            )

        graphs = {}
        graphs["continuous"] = self._create_continuous_plots(ax_time)
        graphs["continuous"][1].set_visible(self.display_original)
        graphs["continuous"][2].set_visible(self.display_reconstructed)

        graphs["sampled"] = ax_time[1].stem(
            self.t_sampled,
            self.sampled,
            linefmt=COLORS["sampled"],
        )
        graphs["spectra"] = self._create_spectral_plots(ax_freq)

        # Make box showing Nyquist limits
        fn = self.sample_rate / 2
        graphs["nyquist"] = []

        for ax in ax_freq:
            ymin, ymax = ax.get_ylim()
            rect = Rectangle(
                (-fn, 0),
                2 * fn,
                ymax - ymin,
                facecolor=COLORS["nyquist"],
                edgecolor="none",
                alpha=0.2,
                zorder=0,
            )
            ax.add_patch(rect)
            graphs["nyquist"].append(rect)

        return fig, ax_time, ax_freq, graphs

    # === Interactive widgets ===============================================
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = "Aliasing in the Time and Frequency Domains"
        title_widget = widgets.Label(title, style=dict(font_weight="bold"))

        # Layouts definitions
        text_layout = {
            "continuous_update": False,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "60%"},
        }

        checkbox_layout = {
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "5%"},
        }

        slider_layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="95%"),
            "style": {"description_width": "15%"},
        }

        # Individual widgets
        phase_widget = widgets.BoundedFloatText(
            min=-180,
            max=180,
            value=np.degrees(self.phase),
            step=5,
            description="Phase [deg]",
            **text_layout,
        )
        phase_widget.observe(
            self._phase_change_callback,
            names="value",
        )

        sample_rate_widget = widgets.BoundedFloatText(
            min=1,
            max=100,
            value=self.sample_rate,
            step=1,
            description="Sample rate [samples/s]",
            **text_layout,
        )
        sample_rate_widget.observe(
            self._sample_rate_change_callback,
            names="value",
        )

        display_original_widget = widgets.Checkbox(
            value=self.display_original,
            description="Original signal",
            **checkbox_layout,
        )
        display_original_widget.observe(
            self._display_original_change_callback,
            names="value",
        )

        display_reconstructed_widget = widgets.Checkbox(
            value=self.display_reconstructed,
            description="Reconstructed signal",
            **checkbox_layout,
        )
        display_reconstructed_widget.observe(
            self._display_reconstructed_change_callback,
            names="value",
        )

        frequency_widget = widgets.FloatSlider(
            min=0.5,
            max=40,
            value=self.freq,
            step=0.5,
            description="Frequency [Hz]",
            readout_format=".1f",
            **slider_layout,
        )
        frequency_widget.observe(
            self._frequency_change_callback,
            names="value",
        )

        # Arrange in columns and lines
        left_column = widgets.VBox(
            [phase_widget, sample_rate_widget],
            layout=widgets.Layout(width="20%"),
        )

        checkbox_column = widgets.VBox(
            [display_original_widget, display_reconstructed_widget],
            layout=widgets.Layout(width="15%"),
        )

        slider_column = widgets.VBox(
            [frequency_widget], layout=widgets.Layout(width="60%")
        )

        widget_layout = widgets.HBox(
            [left_column, checkbox_column, slider_column],
            layout=widgets.Layout(width="90%"),
        )

        widget_layout = widgets.VBox([title_widget, widget_layout])

        return widget_layout
