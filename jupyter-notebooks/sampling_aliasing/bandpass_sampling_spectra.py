"""Illustrate aliasing as repeated spectra in the frequency domain."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

    def __init__(self):
        self.f_max = 3.2     # Max. frequency to plot

        self.fc = 1.4        # Carrier frequency
        self.b = 0.10        # Relative bandwidth
        self.fs = 2.0
        self.m = np.arange(1, 10)
        self.n_samples = 2001
        self.white_noise_level = 0.03

        self.filter_options = {0: 'None',
                               1: 'Anti-alias',
                               2: 'Bandpass'}
        self.filter = self.filter_options[0]

        self.ax, self.ax_text = self.initialise_graphs()
        self.widget = self._create_widgets()

    def frequency(self):
        """Define frequency vector with interval."""
        f, df = np.linspace(-self.f_max, self.f_max, self.n_samples,
                            retstep=True)
        return f, df

    def f_span(self):
        """Find frequencies where to repeat spectrum."""
        n_max = np.ceil(2*self.f_max / self.fs)
        n_int = np.arange(-n_max, n_max+1)
        return self.fs * n_int

    def f_marker(self, f_ref, x):
        """Mark a selected frequency and its aliases."""
        fa = f_ref + self.f_span()   # Vector of alias frequencies

        f, df = self.frequency()
        n = np.abs(f - f_ref).argmin()
        x = x[n] * np.ones_like(fa)  # Value at reference frequencies

        return [fa, x]

    def fs_allowed(self):
        """List allowed sample rates."""
        m = np.arange(0, int(np.ceil(self.fc / self.b - 1/2)))
        with np.errstate(divide='ignore'):
            fs_min = (2*self.fc+self.b)/(m+1)
            fs_max = (2*self.fc-self.b)/m

        k = m
        fs_14 = 4*self.fc/(2*k+1)

        d_snr = 10*np.log10(m+1)

        return m, [fs_min, fs_max, fs_14], d_snr

    def textbox_allowed(self):
        """Crate text-box of allowed frequencies."""
        m, fs, d_snr = self.fs_allowed()

        f_header = ['m',
                    r'$\frac{2f_c+B}{m+1}$',
                    r'$\frac{2f_c-B}{m}$',
                    r'$\frac{4f_c}{2m+1}$',
                    r'$D_{SNR}$ [dB]']
        f_text = (f'{f_header[0]:7} '
                  f'{f_header[1]:24} '
                  f'{f_header[2]:24} '
                  f'{f_header[3]:24} '
                  f'{f_header[4]:2} \n')
        for mk, k in enumerate(m):
            f_text += (f'{mk:3} {fs[0][k]:10.2f} {fs[1][k]:10.2f} '
                       f'{fs[2][k]:10.2f}  {d_snr[k]:10.1f}  \n')

        f_text += '\n'
        f_text += f'     $f_s > 2B $= {2*self.b:.2f}'

        return f_text

    def anti_alias(self, f):
        """Define anti-alias filter."""
        if np.isscalar(f):
            f = np.array([f])

        s = np.ones_like(f)

        if self.filter == self.filter_options[1]:
            n = np.where(abs(f) > (self.fc + self.b/2))
            s[n] = np.zeros_like(n)
        elif self.filter == self.filter_options[2]:
            n = np.where((abs(f) > self.fc + self.b/2) |
                         (abs(f) < self.fc - self.b/2))
            s[n] = np.zeros_like(n)

        return s

    def x_continous(self):
        """Spectrum of continous signal."""
        f, df = self.frequency()
        n0 = np.abs(f).argmin()  # Index of f=0
        nc = np.where(abs(f - self.fc) < self.b/2)

        xn = np.zeros_like(f)
        fn = f[nc] - self.fc
        xn[nc] = 0.5 + 0.25 * fn/self.b

        xn[2*n0-nc] = xn[nc]

        if self.white_noise_level > 0:
            xn += self.white_noise()

        return xn

    def white_noise(self):
        """Define white noise to add to spectrum."""
        rng = np.random.default_rng()

        m = int(np.floor(self.n_samples/2))
        r = np.abs(rng.standard_normal(m))

        x = np.zeros(self.n_samples)
        x[:m] = r
        x[-m:] = np.flip(r[-m:])

        return self.white_noise_level * x

    def x_repeated(self):
        """Repeat spectra with sample rate."""
        f, df = self.frequency()
        xr = np.zeros_like(f)
        xr = np.tile(xr, (len(self.f_span()), 1))
        ns = 1/df  # Index of sample rate

        x0 = self.x_continous()
        xc = x0 * self.anti_alias(f)
        print(x0)

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
        fig = plt.figure(figsize=[12, 5],
                         constrained_layout=True,
                         num='Sampling - Repeated Spectra')

        # Axes for graphs
        n_subplots = 3
        gs = GridSpec(n_subplots, 4, figure=fig)
        ax = [fig.add_subplot(gs[k, 0:-1]) for k in range(n_subplots)]

        # Axis for textbox
        ax_text = fig.add_subplot(gs[:-1, -1])
        ax_text.set_axis_off()

        # Scale and mark axes
        for a in ax:
            a.set(xlim=(-self.f_max, self.f_max),
                  ylim=(0, 1.8))

        ax[0].set_title('Spectrum of Physical Continous Signal')
        ax[1].set_title('Phyical Spectrum Replicated at Interval $f_s$')
        ax[2].set_title('Interpretation: Spectrum Inside Nyquist Limits')

        ax[n_subplots-1].set(xlabel='Frequency')

        return ax, ax_text

    def display(self):
        """Plot results."""
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()
            for art in list(ax.texts):
                art.remove()

        for art in list(self.ax_text.texts):
            art.remove()

        # Mark Nyquist limit and multiples of sample rate
        nyquistcolor = (0.8, 1.0, 0.8, 1.0)
        for ax in self.ax:
            ax.axvspan(-1/2*self.fs, 1/2*self.fs,
                       color=nyquistcolor)
            for f_n in self.f_span():
                ax.axvline(x=f_n, color='0.7', linestyle=':')

        # Spectra
        xr, xc, x0 = self.x_repeated()
        f, df = self.frequency()

        # Original continous spectrum
        self.ax[0].plot(f, xc, 'C0')
        self.ax[1].plot(f, xc, 'C0')

        # Bandwidth markers
        for xm in [-self.fc-self.b/2,
                   -self.fc+self.b/2,
                   self.fc-self.b/2,
                   self.fc+self.b/2]:
            self.ax[0].axvline(x=xm, color='C0', linestyle='dotted')

        # Annotate distence between spectra ltoMark disspan
        x = self.fc + np.array([-self.b/2, self.b/2])
        y = [1.0, 1.5]

        for k in range(len(x)):
            self.ax[0].plot([-x[k], x[k]], [y[k], y[k]],
                            color='C0', linestyle='dotted', marker='|')

        span_marker = {'color': 'C0',
                       'verticalalignment': 'bottom',
                       'horizontalalignment': 'center',
                       'backgroundcolor': (0.5, 0.5, 0.5, 0.0)}

        self.ann = self.ax[0].annotate(f'$2f_c - B$={2*x[0]:.2f}',
                                       (0, y[0]),
                                       **span_marker)

        self.ann = self.ax[0].annotate(f'$2f_c + B$={2*x[1]:.2f}',
                                       (0, y[1]),
                                       **span_marker)

        # Mark sample rate and centre frequency
        text_par = {'verticalalignment': 'top',
                    'horizontalalignment': 'center',
                    'backgroundcolor': '1',
                    'textcoords': 'offset points'}

        self.ax[0].plot(-self.fc, 0, '|')
        self.ax[0].plot(self.fc, 0, '|')
        self.ax[0].annotate('$-f_c$', (-self.fc, 0), (0, -5), **text_par)
        self.ax[0].annotate('$f_c$', (self.fc, 0), (0, -5), **text_par)
        self.ax[1].annotate('$-f_s$', (-self.fs, 1), (0, 0), **text_par)
        self.ax[1].annotate('$f_s$', (self.fs, 1), (0, 0), **text_par)

        # Replicated spectra
        for x in xr:
            self.ax[1].plot(f, x, 'C1')

        # Sum of replicated spectra
        ni = np.where(abs(f) < 1/2*self.fs)
        xs = np.sum(xr, axis=0) + xc
        self.ax[2].plot(f[ni], xs[ni], 'C0')

        # Textbox
        self.ax_text.text(0.1, 0.0, self.textbox_allowed(),
                          backgroundcolor=(0.95, 0.95, 0.95, 1.0))

        return 0

    # Simple interactive operation
    def interact(self,
                 fc=None,
                 b=None,
                 fs=None,
                 white_noise_level=None,
                 anti_alias=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        fc: float, optional
            Carrier frequency
        b: float, optional
            Bandwidth
        fs: float, optional
            Sample rate
        """
        if fc is not None:
            self.fc = fc
        if b is not None:
            self.b = b
        if fs is not None:
            self.fs = fs
        if white_noise_level is not None:
            self.white_noise_level = white_noise_level
        if anti_alias is not None:
            self.filter = anti_alias

        # Display result in graphs
        self.display()

        return

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('Sampling - Repeated Spectra. Ref. Figs. 2-6, 2-7 and 2-8 '
                 'in Lyons, "Understanding Digital Signal Processing", '
                 '3rd ed. (2011)')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': widgets.Layout(width='90%'),
            'style': {'description_width': '70%'}}

        dropdown_layout = {
            'layout': widgets.Layout(width='90%'),
            'style': {'description_width': '40%'}}

        # Individual widgets
        centre_frequency_widget = widgets.FloatText(
            min=0.0, max=1.0, value=self.fc, step=0.1,
            description='Centre frequency',
            readout_format='.2f',
            **text_layout)

        bandwidth_widget = widgets.FloatText(
            min=0.0, max=1.0, value=self.b, step=0.01,
            description='Bandwidth',
            readout_format='.2f',
            **text_layout)

        sample_rate_widget = widgets.FloatText(
            min=0.1, max=3.0, value=self.fs, step=0.01,
            description='Sample rate',
            readout_format='.2f',
            **text_layout)

        white_noise_widget = widgets.FloatText(
            min=0.0, max=0.2, value=0.0, step=0.01,
            description='White noise level',
            readout_format='.3f',
            **text_layout)

        filter_widget = widgets.Dropdown(
            options=[self.filter_options[0],
                     self.filter_options[1],
                     self.filter_options[2]],
            value=self.filter_options[0],
            description='Filter',
            **dropdown_layout)

        # Arrange in columns and lines
        widget_col_1 = widgets.VBox([centre_frequency_widget,
                                     bandwidth_widget,
                                     ],
                                    layout=widgets.Layout(width='90%'))

        widget_col_2 = widgets.VBox([white_noise_widget,
                                     filter_widget,
                                     ],
                                    layout=widgets.Layout(width='90%'))

        widget_layout = widgets.HBox([widget_col_1,
                                      sample_rate_widget,
                                      widget_col_2],
                                     layout=widgets.Layout(width='50%'))

        widget_layout = widgets.VBox([title_widget,
                                      widget_layout])

        # Export as dictionary
        widget = {'centre_frequency': centre_frequency_widget,
                  'bandwidth': bandwidth_widget,
                  'sample_rate': sample_rate_widget,
                  'white_noise_level': white_noise_widget,
                  'filter': filter_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
