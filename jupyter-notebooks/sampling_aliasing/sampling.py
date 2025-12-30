"""Illustrate aliasing in the time domain.

Created on Wed Oct 22 10:39:03 2025
@author: lah
"""

# Load external modules
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets


# Class definitions
class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class MultipleAliasSignal():
    """Demonstation of multiple aliasing frequencies.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, f=0.2, phase=0, fs=1, create_widgets=True):
        """Initialise signal."""
        self.f = f                          # Original frequency
        self.phase = phase                  # Phase [radians]
        self.fs = fs                        # Sample rate

        self.m = 0                          # Alias frequencies
        self.n_t = 1000                     # Number of point in time vectors
        self.t_max = 20/self.fs             # Length of time axis
        self.show_curve = True              # Display curve

        self.color = {'original': 'C1',
                      'sampled': 'C0',
                      'aliased': 'C3'}

        self.ax = self._initialise_graphs()
        if create_widgets:
            self.widget = self._create_widgets()

    # Time vectors
    def ti(self):
        """Original time vector."""
        return np.arange(0, self.t_max, 1/(40*self.fs))

    def ts(self):
        """Get sampled time vector."""
        return np.arange(0, self.t_max+0.5/self.fs, 1/self.fs)

    # Original and sampled signals
    def original(self):
        """Original signal."""
        return np.cos(2 * pi * self.f * self.ti() + self.phase)

    def sampled(self):
        """Calculate sampled signal."""
        return np.cos(2*pi * self.f * self.ts() + self.phase)

    def alias(self):
        """Reconstructed signal."""
        return np.cos(2 * pi * self.fa() * self.ti() + self.phase)

    # Frequencies
    def fa(self):
        """Find alias no. m of frequency."""
        return (self.f + self.m * self.fs)

    # Display results
    def display(self):
        """Plot all signals and spectra."""
        for art in list(self.ax.lines):
            art.remove()
        for art in list(self.ax.collections):
            art.remove()

        # Plot time traces
        if self.show_samples:
            self.ax.stem(self.ts(), self.sampled(),
                         linefmt=self.color["sampled"],
                         basefmt='gray')

        if self.show_curve:
            self.ax.plot(self.ti(), self.alias(),
                         linestyle="-",
                         color=self.color["original"],
                         label=f"f = {abs(self.fa()):.2f} Hz")

        self.ax.legend(loc="upper right")

        return 0

    # Simple interactive operation
    def interact(self,
                 frequency=None,
                 phase_deg=None,
                 sample_rate=None,
                 alias_no=None,
                 show_curve=None,
                 show_samples=None,
                 ):
        """Scale inputs and  display results.

        For interactive operation with simple inputs.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        frequency: float, optional
            Frequency of original in Hz
        phase_deg: float, optional
            Phase of original in degrees
        sample_rate: float, optional
            Sample rate in 1/s
        alias_no: int, optional
            Alias number, original frequency shifted m times sample rate
        """
        if frequency is not None:
            self.f = frequency
        if phase_deg is not None:
            self.phase = np.radians(phase_deg)
        if sample_rate is not None:
            self.fs = sample_rate
        if alias_no is not None:
            self.m = alias_no
        if show_curve is not None:
            self.show_curve = show_curve
        if show_samples is not None:
            self.show_samples = show_samples

        # Display result in graphs
        self.display()

        return

    # Non-public methods
    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close("all")

        fig = plt.figure(figsize=[14, 6],
                         clear=True,
                         num='Aliasing Demo')

        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=(0, self.t_max),
               ylim=(-1.1, 1.1),
               xlabel="Time [s]")

        ax.axhline(y=0)

        return ax

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = 'Aliasing in the Time Domain'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': widgets.Layout(width='95%'),
            'style': {'description_width': '70%'}}

        checkbox_layout = {
            'layout': widgets.Layout(width='95%'),
            'style': {'description_width': '20%'}}

        # Individual widgats
        phase_widget = widgets.FloatText(
            min=-180, max=180, value=45, step=15,
            description='Phase [deg]',
            readout_format='.0f',
            **text_layout)

        sample_rate_widget = widgets.FloatText(
            min=0.1, max=4, value=1, step=0.1,
            description='Sample rate [1/s]',
            readout_format='.1f',
            **text_layout)

        alias_no_widget = widgets.IntText(
            value=0,
            description='Alias number',
            **text_layout)

        show_curve_widget = widgets.Checkbox(
            value=True,
            description='Display curve',
            **checkbox_layout)

        show_samples_widget = widgets.Checkbox(
            value=False,
            description='Display samples',
            **checkbox_layout)

        frequency_widget = widgets.FloatText(
            min=0.1, max=2, value=0.3, step=0.1,
            description='Frequency [Hz]',
            readout_format='.1f',
            **text_layout)

        # Arrange in columns and lines
        left_column = widgets.VBox([sample_rate_widget,
                                    alias_no_widget],
                                   layout=widgets.Layout(width='20%'))

        checkbox_column = widgets.VBox([show_curve_widget,
                                        show_samples_widget],
                                       layout=widgets.Layout(width='20%'))

        last_column = widgets.VBox([phase_widget,
                                    frequency_widget],
                                   layout=widgets.Layout(width='20%'))

        widget_layout = widgets.HBox([left_column,
                                      last_column,
                                      checkbox_column
                                     ],
                                     layout=widgets.Layout(width='90%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'frequency': frequency_widget,
                  'phase_deg': phase_widget,
                  'sample_rate': sample_rate_widget,
                  'alias_no': alias_no_widget,
                  'show_curve': show_curve_widget,
                  'show_samples': show_samples_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
