"""Illustrate aliasing as repeated spectra in the frequency domain."""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class SampledSpectra():
    """Demonstation of aliasing as overlaping spectra in the frequency domain.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, initialise_widgets=True):
        self.f_max = 2.2     # Max. frequency to plot
        self.b = 0.40        # Relative bandwidth
        self.n_samples = 1001

        self.include_ref = False
        self.f_ref = 0.25    # Reference fraquency, for marker

        self.white_noise_level = 0.2
        self.include_anti_alias = False
        self.include_noise = False
        self.f_noise = 1.2      # Width of noise peak
        self.df_noise = 0.02    # Width of noise peak
        self.n_max = 0.7    # Height of noise peak

        if initialise_graphs:
            self.ax = self.initialise_graphs()
        if initialise_widgets:
            self.widget = self._create_widgets()

    def frequency(self):
        """Define frequency vector with interval."""
        f, df = np.linspace(-self.f_max, self.f_max, self.n_samples,
                            retstep=True)
        return f, df

    def f_span(self):
        """Find frequencies where to repeat spectrum."""
        f_int = int(np.ceil(2*self.f_max))
        return np.arange(-f_int, f_int+1)

    def f_marker(self, f_ref, x):
        """Mark a selected frequency and its aliases."""
        fa = f_ref + self.f_span()   # Vector of alias frequencies

        f, df = self.frequency()
        n = np.abs(f - f_ref).argmin()
        x = x[n] * np.ones_like(fa)  # Value at reference frequencies

        return [fa, x]

    def signal(self, f):
        """Define spectrum of continous signal, no noise."""
        if np.isscalar(f):
            f = np.array([f])

        s = np.zeros_like(f)
        n = np.where(abs(f) < self.b)
        s[n] = np.cos(0.5*pi*f[n]/self.b)
        s[n] = 1.05-abs(f[n]/self.b)

        return s

    def white_noise(self):
        """Define white noise to add to spectrum."""
        rng = np.random.default_rng()

        m = int(np.floor(self.n_samples/2))
        r = np.abs(rng.standard_normal(m))

        x = np.zeros(self.n_samples)
        x[:m] = r
        x[-m:] = np.flip(r[-m:])

        return self.white_noise_level * x

    def noise_peak(self):
        """Define single noise peak."""
        f, df = self.frequency()
        n0 = np.abs(f).argmin()  # Index of f=0
        n = np.where(abs(f - self.f_noise) < self.df_noise)
        xn = np.zeros_like(f)

        fn = f[n]-self.f_noise
        xn[n] = self.n_max/2 * (np.cos(pi/self.df_noise*fn) + 1)
        xn[2*n0-n] = np.flip(xn[n])

        return xn

    def anti_alias(self, f):
        """Anti-alias filter, defined as perfect lowpass filter."""
        if np.isscalar(f):
            f = np.array([f])

        s = np.ones_like(f)
        n = np.where(abs(f) >= 0.5)
        s[n] = np.zeros_like(n)

        return s

    def x_continous(self):
        """Spectrum of original, continous signal with noise."""
        f, df = self.frequency()

        x = self.signal(f)
        if self.include_noise:
            x += self.noise_peak()
        if self.white_noise_level > 0:
            x += self.white_noise()

        return x

    def x_repeated(self):
        """Repeat spectra with sample rate."""
        f, df = self.frequency()
        xr = np.zeros_like(f)
        xr = np.tile(xr, (len(self.f_span()), 1))
        ns = 1/df  # Index of sample rate

        x0 = self.x_continous()
        if self.include_anti_alias:
            xc = x0 * self.anti_alias(f)
        else:
            xc = x0

        n = len(x0)
        for k, f_n in enumerate(self.f_span()):
            dn = int(ns*f_n)
            if f_n > 0:
                if dn < n and dn > 0:
                    xr[k][dn:] = xc[:-dn]
            elif f_n < 0:
                if n+dn < n and n+dn > 0:
                    xr[k][:n+dn] = xc[-dn:]

        return xr, xc, x0

    def initialise_graphs(self):
        """Initialise graphs for spectra."""
        # Define figure
        fig = plt.figure(figsize=[12, 6],
                         constrained_layout=True,
                         num='Sampling - Repeated Spectra')

        n_subplots = 3
        ax = fig.subplots(n_subplots, 1, sharex=True)

        for axn in ax:
            for f_n in self.f_span():
                axn.axvline(x=f_n, color='gray', linestyle=':')
                axn.axvspan(-1/2, 1/2, alpha=0.02, color='green')
                # axn.axvspan(-self.f_max, -1/2, alpha=0.02, color='red')
                # axn.axvspan(1/2, self.f_max, alpha=0.02, color='red')

            axn.set(xlim=(-self.f_max, self.f_max),
                    ylim=(0, 1.7))

        ax[0].set_title('Spectrum of Physical Signal - Continous')
        ax[1].set_title('Phyical Spectrum Replicated at Interval $f_s$')
        ax[2].set_title('Interpretation: Spectrum Inside Nyquist Limits')

        ax[n_subplots-1].set(xlabel='Frequency [$f/f_s$]')

        return ax

    def display(self):
        """Plot results."""
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()

        # Lines marking multiples of sampling rate
        for axn in self.ax:
            for f_n in self.f_span():
                axn.axvline(x=f_n, color='gray', linestyle=':')

        f, df = self.frequency()
        xr, xc, x0 = self.x_repeated()

        # Original continous spectrum
        self.ax[0].plot(f, xc, 'C0')
        if self.include_anti_alias:
            self.ax[0].plot(f, x0, 'C0', linestyle='dotted')
        self.ax[1].plot(f, xc, 'C0')

        # Replicated spectra
        for x in xr:
            self.ax[1].plot(f, x, 'C1')

        # Mark reference frequency
        if self.include_ref:
            if self.include_noise:
                f_ref = self.f_noise
            else:
                f_ref = self.f_ref

            f_mark, x_mark = self.f_marker(f_ref, xc)
            self.ax[1].plot(f_mark, x_mark,
                            color='C5', linestyle='dotted', marker='o')

        # Sum of replicated spectra
        print('start')
        ni = np.where(abs(f) < 1/2)
        xs = np.sum(xr, axis=0) + xc
        self.ax[2].plot(f[ni], xs[ni], 'C0')

        return 0

    # Simple interactive operation
    def interact(self,
                 b=None,
                 include_ref=False,
                 f_ref=None,
                 white_noise_level=False,
                 include_anti_alias=False,
                 include_noise=False,
                 f_noise=None):
        """Scale inputs and  display results.

        For interactive operation with dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        b: float, optional
            Bandwidth as maximum frequency relative to sample rate
        """
        if b is not None:
            self.b = b
        if f_ref is not None:
            self.f_ref = f_ref
        if include_ref is not None:
            self.include_ref = include_ref
        if include_anti_alias is not None:
            self.include_anti_alias = include_anti_alias
        if white_noise_level is not None:
            self.white_noise_level = white_noise_level
        if include_noise is not None:
            self.include_noise = include_noise
        if f_noise is not None:
            self.f_noise = f_noise

        # Display result in graphs
        self.display()

        return

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = 'Sampling - Repeated Spectra. Ref. Figs. 2-4 and 2-5 in Lyons, "Understanding Digital Signal Processing", 3rd ed. (2011)'

        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': widgets.Layout(width='90%'),
            'style': {'description_width': '75%'}}

        checkbox_layout = {
            'layout': widgets.Layout(width='90%')}

        # Individual widgets
        empty_widget = widgets.Label(' ', **text_layout)

        bandwidth_widget = widgets.FloatText(
            min=0.0, max=1.0, value=0.30, step=0.05,
            description='Bandwidth rel. sample rate',
            readout_format='.2f',
            **text_layout)

        marker_widget = widgets.FloatText(
            min=-2.0, max=2.0, value=0.25, step=0.05,
            description='Reference frequency',
            readout_format='.2f',
            **text_layout)

        include_marker_widget = widgets.Checkbox(
            value=False,
            description='Include frequency marker',
            **checkbox_layout)

        f_noise_widget = widgets.FloatText(
            min=2.5, max=2.5, value=1.20, step=0.05,
            description='Noise peak frequency',
            readout_format='.2f',
            **text_layout)

        white_noise_widget = widgets.FloatText(
            min=0.0, max=0.2, value=0.0, step=0.01,
            description='White noise level',
            readout_format='.3f',
            **text_layout)

        include_noise_widget = widgets.Checkbox(
            value=False,
            description='Include noise peak',
            **checkbox_layout)

        include_anti_alias_widget = widgets.Checkbox(
            value=False,
            description='Anti-aliasing filter',
            **checkbox_layout)

        # Arrange in columns and lines
        widget_line_1 = widgets.HBox(
            [bandwidth_widget, marker_widget, include_marker_widget])
        widget_line_2 = widgets.HBox(
            [empty_widget, f_noise_widget, include_noise_widget])
        widget_line_3 = widgets.HBox(
            [empty_widget, white_noise_widget, include_anti_alias_widget])
        widget_layout = widgets.VBox([title_widget,
                                      widget_line_1,
                                      widget_line_2,
                                      widget_line_3],
                                     layout=widgets.Layout(width='60%'))

        # Export as dictionary
        widget = {'bandwidth': bandwidth_widget,
                  'marker': marker_widget,
                  'white_noise_level': white_noise_widget,
                  'include_marker': include_marker_widget,
                  'f_noise': f_noise_widget,
                  'include_noise': include_noise_widget,
                  'include_anti_alias': include_anti_alias_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
