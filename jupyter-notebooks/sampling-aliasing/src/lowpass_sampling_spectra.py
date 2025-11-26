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

    def _f_vector(self):
        """Define frequency vector with interval."""
        return np.linspace(-self.f_max, self.f_max,
                           self.n_samples, retstep=True)

    def f(self):
        """Return frequency vector."""
        f, df = self._f_vector()
        return f

    def df(self):
        """Return frequency vector interval."""
        f, df = self._f_vector()
        return df

    def f_span(self):
        """Find frequencies where to repeat spectrum."""
        f_int = int(np.ceil(2*self.f_max))
        return np.arange(-f_int, f_int+1)

    def f_marker(self, f_ref, x):
        """Mark a selected frequency and its replica."""
        f = f_ref + self.f_span()   # Vector of equal frequencies

        n = np.abs(self.f() - f_ref).argmin()
        x = x[n] * np.ones_like(f)  # Value at reference frequencies

        return [f, x]

    def signal(self, f):
        """Define continous signal."""
        if np.isscalar(f):
            f = np.array([f])

        s = np.zeros_like(f)
        n = np.where(abs(f) < self.b)
        s[n] = np.cos(0.5*pi*f[n]/self.b)
        # x[n] = 1-abs(f[n]/self.b)

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

    def noise(self):
        """Define noise peak."""
        n0 = np.abs(self.f()).argmin()  # Index of f=0
        n = np.where(abs(self.f() - self.f_noise) < self.df_noise)
        xn = np.zeros_like(self.f())

        fn = self.f()[n]-self.f_noise
        xn[n] = self.n_max/2 * (np.cos(pi/self.df_noise*fn) + 1)
        xn[2*n0-n] = np.flip(xn[n])

        return xn

    def anti_alias(self, f):
        """Define continous signal."""
        if np.isscalar(f):
            f = np.array([f])

        s = np.ones_like(f)
        n = np.where(abs(f) > 0.49)
        s[n] = np.zeros_like(n)

        return s

    def xc(self):
        """Spectrum of original, continous signal."""
        x = self.signal(self.f())
        if self.include_noise:
            x += self.noise()
        if self.white_noise_level > 0:
            x += self.white_noise()

        return x

    def x_aaf(self):
        """Spectrum before sampling."""
        x = self.x0()
        if self.include_anti_alias:
            x *= self.anti_alias(self.f())
        return x

    def xr(self):
        """Repeat spectra with sample rate."""
        xr = np.zeros_like(self.f())
        xr = np.tile(xr, (len(self.f_span()), 1))
        ns = 1/self.df()  # Index of sample rate

        x0 = self.xc()
        if self.include_anti_alias:
            xc = x0 * self.anti_alias(self.f())
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

    def xs(self):
        """Sum signals."""
        return np.sum(self.xr(), axis=0) + self.xc()

    def initialise_graphs(self):
        """Initialise graphs for spectra."""
        # Define figure
        fig = plt.figure(figsize=[14, 7],
                         constrained_layout=True,
                         num='Sampling - Repeated Spectra')

        n_subplots = 4
        ax = fig.subplots(n_subplots, 1, sharex=True)

        for axn in ax:
            for f_n in self.f_span():
                axn.axvline(x=f_n, color='gray', linestyle=':')
                axn.axvspan(-1/2, 1/2, alpha=0.02, color='green')
                # axn.axvspan(-self.f_max, -1/2, alpha=0.02, color='red')
                # axn.axvspan(1/2, self.f_max, alpha=0.02, color='red')

            axn.set(xlim=(-self.f_max, self.f_max),
                    ylim=(0, 1.7))

        ax[0].set_title('Original spectrum - Continous signal')
        ax[1].set_title('Replicas of original spectrum')
        ax[2].set_title('Spectrum after sampling')
        ax[3].set_title('Interpretation: Spectrum inside Nyquist limits')

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

        xr, xc, x0 = self.xr()

        # Original continous spectrum
        self.ax[0].plot(self.f(), xc, 'C0')
        if self.include_anti_alias:
            self.ax[0].plot(self.f(), x0, 'C0', linestyle='dotted')
        self.ax[1].plot(self.f(), xc, 'C0')

        # Replicated spectra
        for x in xr:
            self.ax[1].plot(self.f(), x, 'C1')

        # Mark refrence frequency
        if self.include_ref:
            if self.include_noise:
                f_ref = self.f_noise
            else:
                f_ref = self.f_ref
            f, x = self.f_marker(f_ref, xc)
            self.ax[1].plot(f, x, color='C5', linestyle='dotted', marker='o')

        # Sum of replicated spectra
        ni = np.where(abs(self.f()) < 1/2)

        xs = np.sum(xr, axis=0) + xc

        self.ax[2].plot(self.f(), xs, 'C1')
        self.ax[2].plot(self.f()[ni], xs[ni], 'C0')

        self.ax[3].plot(self.f()[ni], xs[ni], 'C0')

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
        title = 'Sampling - Repeated Spectra. Ref. Figs. 2-4 and 2-5 in Lyons,'
        ' "Understanding Digital Signal Processing", 3rd ed. (2011)'

        title_widget = widgets.Label(title, style=dict(font_weight='bold'))
        empty_widget = widgets.Label(' ')

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': widgets.Layout(width='100%'),
            'style': {'description_width': '70%'}}

        checkbox_layout = {
            'layout': widgets.Layout(width='95%')}

        # slider_layout = {
        #     'continuous_update': True,
        #     'layout': widgets.Layout(width='60%'),
        #     'style': {'description_width': '30%'}}

        # Individual widgets
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
        widget_items = [bandwidth_widget, include_anti_alias_widget,
                        white_noise_widget, empty_widget,
                        marker_widget, include_marker_widget,
                        f_noise_widget, include_noise_widget ]

        widget_layout = widgets.GridBox(widget_items, layout=widgets.Layout(grid_template_columns='repeat(2, 25em)'))

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
