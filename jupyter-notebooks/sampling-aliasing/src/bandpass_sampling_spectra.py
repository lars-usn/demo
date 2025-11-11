"""Illustrate aliasing as repeated spectra in the frequency domain."""

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

    def __init__(self):
        self.f_max = 3.2     # Max. frequency to plot

        self.fc = 1.4      # Width of noise peak
        self.b = 0.10        # Relative bandwidth
        self.fs = 2.0
        self.m = np.arange(1, 10)

        self.ax = self.initialise_graphs()
        self.widget = self._create_widgets()

    def _f_vector(self):
        """Define frequency vector with interval."""
        return np.linspace(-self.f_max, self.f_max, 2001, retstep=True)

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
        n_max = np.ceil(2*self.f_max / self.fs)
        n_int = np.arange(-n_max, n_max+1)
        return self.fs * n_int

    def f_marker(self, include_all=True):
        """Create array of equal frequencies."""
        f = self.f_ref
        if include_all:
            f += self.f_span()   # Vector of equal frequencies

        n = np.abs(self.f()-self.f_ref).argmin()
        x = self.xc()[n] * np.ones_like(f)  # Value at reference frequencies

        return [f, x]

    def fs_allowed(self):
        """List allowed sample rates."""
        m = np.arange(1, 6)
        fs_min = (2*self.fc+self.b)/(m+1)
        fs_max = (2*self.fc-self.b)/m

        return m, fs_min, fs_max

    def textbox_allowed(self):
        """Crate text-box of allowed frequencies."""
        m, f_min, f_max = self.fs_allowed()

        f_text = 'm      $(2f_c - B)/m$      $(2f_c+B)/(m+1)$ \n'
        for k, mk in enumerate(m):
            f_text += f'{mk}  {f_min[k]:16.2f}  {f_max[k]:16.2f} \n'

        f_text += '\n'
        f_text += f'     $f_s > 2B $= {2*self.b:.2f}'

        return f_text

    def xc(self):
        """Spectrum of continous signal."""
        n0 = np.abs(self.f()).argmin()  # Index of f=0
        nc = np.where(abs(self.f() - self.fc) < self.b/2)

        xn = np.zeros_like(self.f())
        fn = self.f()[nc] - self.fc
        xn[nc] = 0.5 + 0.25 * fn/self.b

        xn[2*n0-nc] = xn[nc]

        return xn

    def xr(self):
        """Repeat spectra with sample rate."""
        x = np.zeros_like(self.f())
        x = np.tile(x, (len(self.f_span()), 1))
        ns = 1/self.df()  # Index of sample rate

        x0 = self.xc()
        n = len(x0)
        for k, f_n in enumerate(self.f_span()):
            dn = int(ns*f_n)
            if f_n > 0:
                if dn < n and dn > 0:
                    x[k][dn:] = x0[:-dn]
            elif f_n < 0:
                if n+dn < n and n+dn > 0:
                    x[k][:n+dn] = x0[-dn:]

        return x

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
            axn.set(xlim=(-self.f_max, self.f_max),
                    ylim=(0, 1.8))

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
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()
            for art in list(ax.texts):
                art.remove()

        # Mark Nyquist limit
        nyquistcolor = 'honeydew'
        for ax in self.ax:
            for f_n in self.f_span():
                ax.axvline(x=f_n, color='gray', linestyle=':')
                ax.axvspan(-1/2*self.fs, 1/2*self.fs,
                           color=nyquistcolor)

        # Lines marking multiples of sampling rate
        for axn in self.ax:
            for f_n in self.f_span():
                axn.axvline(x=f_n, color='C1', linestyle=':')

        # Original continous spectrum
        self.ax[0].plot(self.f(), self.xc(), 'C0')
        self.ax[1].plot(self.f(), self.xc(), 'C0')

        # Mark span
        x = self.fc + np.array([-self.b/2, self.b/2])
        y = [1.1, 1.6]

        for k in range(len(x)):
            self.ax[0].plot([-x[k], x[k]], [y[k], y[k]],
                            color='C0', linestyle='dotted', marker='|')

        span_marker = {'color': 'C0',
                       'va': 'center',
                       'backgroundcolor': nyquistcolor}

        self.ann = self.ax[0].annotate(f'$2f_c - B$={2*x[0]:.2f}',
                                       (0, y[0]),
                                       **span_marker)

        self.ann = self.ax[0].annotate(f'$2f_c + B$={2*x[1]:.2f}',
                                       (0, y[1]),
                                       **span_marker)

        # Bandwidth markers
        for xm in [-self.fc-self.b/2,
                   -self.fc+self.b/2,
                   self.fc-self.b/2,
                   self.fc+self.b/2]:
            self.ax[0].axvline(x=xm, color='C0', linestyle='dotted')

        text_par = {'verticalalignment': 'top',
                    'horizontalalignment': 'center',
                    'textcoords': 'offset points',
                    'color': 'C1'}

        self.ax[0].plot(-self.fc, 0, '|')
        self.ax[0].plot(self.fc, 0, '|')
        self.ax[0].annotate(f'$-f_c=-{self.fc:.2f}$', (-self.fc, 0),
                            (0, -5), **text_par)
        self.ax[0].annotate(f'$f_c={self.fc:.2f}$', (self.fc, 0),
                            (0, -5), **text_par)
        self.ax[1].annotate(f'$f_s={-self.fs:.2f}$', (-self.fs, 0),
                            (0, -5), **text_par)
        self.ax[1].annotate(f'$f_s={self.fs:.2f}$', (self.fs, 0),
                            (0, -5), **text_par)

        # Replicated spectra
        for x in self.xr():
            self.ax[1].plot(self.f(), x, 'C1')

        # Sum of replicated spectra
        ni = np.where(abs(self.f()) < 1/2*self.fs)

        self.ax[2].plot(self.f(), self.xs(), 'C1')
        self.ax[2].plot(self.f()[ni], self.xs()[ni], 'C0')

        self.ax[3].plot(self.f()[ni], self.xs()[ni], 'C0')

        self.ax[0].text(self.f_max, 0.0, self.textbox_allowed(),
                        backgroundcolor='linen')

        return 0

    # Simple interactive operation
    def interact(self,
                 fc=None,
                 b=None,
                 fs=None):
        """Scale inputs and  display results.

        For interactive operation with dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        b: float, optional
            Bandwidth as maximum frequency relative to sample rate
        """
        if fc is not None:
            self.fc = fc
        if b is not None:
            self.b = b
        if fs is not None:
            self.fs = fs

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
            'layout': widgets.Layout(width='20%'),
            'style': {'description_width': '70%'}}

        # checkbox_layout = {
        #     'layout': widgets.Layout(width='95%'),
        #     'style': {'description_width': '5%'}}

        # slider_layout = {
        #     'continuous_update': True,
        #     'layout': widgets.Layout(width='70%'),
        #     'style': {'description_width': '30%'}}

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

        # Arrange in columns and lines
        widget_layout = widgets.HBox([centre_frequency_widget,
                                      bandwidth_widget,
                                      sample_rate_widget,
                                      ],
                                     layout=widgets.Layout(width='80%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'centre_frequency': centre_frequency_widget,
                  'bandwidth': bandwidth_widget,
                  'sample_rate': sample_rate_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
