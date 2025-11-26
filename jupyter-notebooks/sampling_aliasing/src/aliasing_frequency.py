"""Illustrate aliasing in the time- and frequency domains."""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

# Class definitions


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class FrequencyAliasSignal():
    """Demonstation of aliasing in the time- and frequency domains.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, f=5, phase=0, fs=20):
        """Initialise signal."""
        self.f = f                          # Original frequency
        self.phase = 0.0                    # Phase [radians]
        self.fs = fs                        # Sample rate
        self.n_alias = np.arange(-10, 10)   # Alias frequency number
        self.n_t = 1000                     # Number of points in time vectors
        self.t_max = 15/self.fs             # Length of time axis
        self.display_original = True
        self.display_reconstructed = True

        self.color = {'original': 'C0',
                      'sampled': 'C1',
                      'reconstructed': 'C2',
                      'aliased': 'C3',
                      'nyquist': 'C2'}

        self.ax_time, self.ax_freq = self._initialise_graphs()
        self.widget = self._create_widgets()

    # Time vectors
    def dts(self):
        """Sample time."""
        return 1/self.fs

    def ti(self):
        """Original time vector, set to 100 samples per cycle."""
        return np.arange(0, self.t_max, 1/(100*self.f))

    def ts(self):
        """Get sampled time vector, including entire time vector."""
        return np.arange(0, self.t_max+self.dts()/2, self.dts())

    # Original and sampled signals
    def original(self):
        """Original signal."""
        return np.cos(2 * pi * self.f * self.ti() + self.phase)

    def sampled(self):
        """Calculate sampled signal."""
        return np.cos(2*pi * self.f * self.ts() + self.phase)

    def reconstructed(self):
        """Reconstructed signal, using principal alias."""
        return np.cos(2 * pi * self.fa() * self.ti() + self.phase)

    # Frequencies
    def f_all(self):
        """Find the lowest unique aliasing frequencies."""
        fp = self.f + self.n_alias * self.fs
        fn = -fp      # Add negative aliases
        return np.sort(np.append(fn, fp))

    def fa(self):
        """Principal alias frequency."""
        return np.min(abs(self.f_all()))

    # Display results
    def display(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for ax in self.ax_time + self.ax_freq:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        # Plot time traces
        self.ax_time[0].plot(self.ti(), self.original(),
                             '-',
                             color=self.color['original'])
        self.ax_time[1].stem(self.ts(), self.sampled(),
                             linefmt=self.color['sampled'])

        bacgroundline = {'linestyle': '-', 'linewidth': 0.8}

        if self.display_original:
            self.ax_time[1].plot(self.ti(), self.original(),
                                 color=self.color['original'],
                                 **bacgroundline)

        if self.display_reconstructed:
            self.ax_time[1].plot(self.ti(), self.reconstructed(),
                                 color=self.color['reconstructed'],
                                 **bacgroundline)

        self.ax_time[2].plot(self.ti(), self.reconstructed(),
                             linestyle='-',
                             color=self.color['reconstructed'])

        # Add titles with values
        self.ax_time[0].set_title(f'Frequency = {self.f:.1f} Hz')
        self.ax_time[1].set_title(f'Sampling at {self.fs:.1f} samples/s')
        self.ax_time[2].set_title(f'Reconstructed frequency = {
                                  abs(self.fa()):.1f} Hz')

        # Plot spectra
        self.ax_freq[0].stem([-self.f, self.f], np.ones(2),
                             linefmt=self.color['original'])
        self.ax_freq[1].stem(self.f_all(), np.ones(len(self.f_all())),
                             linefmt=self.color['sampled'])
        self.ax_freq[1].stem([-self.f, self.f], np.ones(2),
                             linefmt=self.color['original'])
        self.ax_freq[2].stem([-self.fa(), self.fa()], np.ones(2),
                             linefmt=self.color['reconstructed'])

        # == Disabled ===
        # Indicate if aliasing occurs
        # aliasing = (self.f > self.fs/2)
        # if aliasing:
        #    nyquistcolor = self.color['aliased']
        # else:
        #    nyquistcolor = self.color['nyquist']
        # ===============

        # Make box showing Nyquist limits
        fn = self.fs/2
        for ax in self.ax_freq:
            ax.fill_betweenx(y=[0, 2], x1=-fn, x2=fn,
                             alpha=0.2, color=self.color['nyquist'])
            ax.plot([-fn, -fn], [0, 2], color=self.color['nyquist'])
            ax.plot([fn, fn], [0, 2], color=self.color['nyquist'])

        return 0

    # Simple interactive operation
    def interact(self,
                 frequency=None,
                 phase_deg=None,
                 sample_rate=None,
                 display_original=None,
                 display_reconstructed=None):
        """Scale inputs and  display results.

        For interactive operation with dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        frequency: float, optional
            Frequency of original in Hz
        phase_deg: float, optional
            Phase of original in degrees
        sample_rate: float, optional
            Sample rate in 1/s
        """
        if frequency is not None:
            self.f = frequency
        if phase_deg is not None:
            self.phase = np.radians(phase_deg)
        if sample_rate is not None:
            self.fs = sample_rate
        if display_original is not None:
            self.display_original = display_original
        if display_original is not None:
            self.display_reconstructed = display_reconstructed

        # Display result in graphs
        self.display()

        return

    # Non-public methods
    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        plt.rc('font', size=10)          # Default text sizes
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num='Aliasing Demo')

        n_plots = 3
        ax_time = [fig.add_subplot(2, n_plots, k+1) for k in range(n_plots)]
        ax_freq = [fig.add_subplot(2, n_plots, k+1+n_plots)
                   for k in range(n_plots)]

        fs_scale = 2*self.fs
        for k in range(n_plots):
            ax_time[k].set(xlim=(0, self.t_max),
                           ylim=(-1.1, 1.1),
                           xlabel='Time [s]')

            ax_freq[k].set(xlim=(-fs_scale, fs_scale),
                           ylim=(0, 1.1),
                           xlabel='Frequency [Hz]')

        return ax_time, ax_freq

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = 'Aliasing in the Time and Frequency Domains'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': widgets.Layout(width='95%'),
            'style': {'description_width': '60%'}}

        checkbox_layout = {
            'layout': widgets.Layout(width='95%'),
            'style': {'description_width': '5%'}}

        slider_layout = {
            'continuous_update': True,
            'layout': widgets.Layout(width='95%'),
            'style': {'description_width': '15%'}}

        # Individual widgats
        phase_widget = widgets.FloatText(
            min=-180, max=180, value=0, step=5,
            description='Phase [deg]',
            readout_format='.0f',
            **text_layout)

        sample_rate_widget = widgets.FloatText(
            min=1, max=100, value=40, step=1,
            description='Sample rate [1/s]',
            readout_format='.1f',
            **text_layout)

        display_original_widget = widgets.Checkbox(
            value=True,
            description='Original signal',
            **checkbox_layout)

        display_reconstructed_widget = widgets.Checkbox(
            value=False,
            description='Reconstructed signal',
            **checkbox_layout)

        frequency_widget = widgets.FloatSlider(
            min=0.5, max=40, value=8, step=0.5,
            description='Frequency [Hz]',
            readout_format='.1f',
            **slider_layout)

        # Arrange in columns and lines
        left_column = widgets.VBox([phase_widget,
                                    sample_rate_widget],
                                   layout=widgets.Layout(width='20%'))

        checkbox_column = widgets.VBox([display_original_widget,
                                        display_reconstructed_widget],
                                       layout=widgets.Layout(width='15%'))

        slider_column = widgets.VBox([frequency_widget],
                                     layout=widgets.Layout(width='60%'))

        widget_layout = widgets.HBox([left_column,
                                      checkbox_column,
                                      slider_column],
                                     layout=widgets.Layout(width='90%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'frequency': frequency_widget,
                  'phase_deg': phase_widget,
                  'sample_rate': sample_rate_widget,
                  'display_original': display_original_widget,
                  'display_reconstructed': display_reconstructed_widget
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
