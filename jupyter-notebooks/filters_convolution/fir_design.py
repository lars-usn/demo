"""Demonstrate Fourier synthesis: Build waveform from cosine-waves."""
# Import libraries

# from math import pi
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import ipywidgets as widgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Signal():
    """Demonstation of Fourier synthesis.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise signal."""
        # Incoming signal, two frequencies
        self.n_max = 256
        self.k = 20
        self.n_w = 8
        self.n = np.arange(0, self.n_max) - self.n_max/2

        self.window_list = [('Rectangular', 'boxcar'),
                            ('Hamming', 'hamming',),
                            ('Hann', 'hann'),
                            ('Tukey', 'tukey'),
                            ('Blackman', 'blackman'),
                            ('Chebyshev', 'chebwin'),
                            ('Kaiser', 'kaiser')]
        self.winpar = 8
        self.window_name = self.window_list[0][1]

        self.display_samples = False

        if initialise_graphs:
            self.ax = self._initialise_graphs()
            # self.scale_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def H_inf(self):
        H = np.zeros_like(self.n)
        H[(self.n >= -self.k) * (self.n <= self.k)] = 1

        return fftshift(H)

    def h_inf(self):
        h = ifft(self.H_inf())
        h = fftshift(h)
        if np.max(np.imag(h)) < 1e-6:
            h = np.real(h)

        return h

    def h(self):
        h = self.h_inf() * self.window()
        return h

    def H(self):
        h = self.h()
        h = fftshift(h)
        H = fft(h)
        if np.max(np.imag(H)) < 1e-6:
            H = np.real(H)

        return H

    def window(self):
        """Select window function from name."""
        n = self.n
        w = np.zeros_like(n)
        win = self.window_name
        if self.window_name ==  'kaiser':
            win = (self.window_name, self.winpar)
        if self.window_name ==  'chebwin':
            # Chebyshev window parameter from Lyons vs. SciPy
            win = (self.window_name, self.winpar*20)
        wf = signal.get_window(win, 2*self.n_w+1, fftbins=False)

        w[np.abs(self.n) <= self.n_w] = wf
        return w

    def display(self):
        """Plot all signals and spectra."""
        # Clear old lines
        for ax in self.ax.flatten():
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()
            for art in list(ax.texts):
                art.remove()

        n = self.n

        col = ['C0', 'C1', 'C0']
        for ax in self.ax.flatten():
            ax.axhline(y=0, color='gray')

        for k, x in enumerate([self.h_inf(), self.window(), self.h()]):
            self.ax[k, 0].stem(n, x, linefmt=col[k], basefmt = 'gray')

        W = fft(fftshift(self.window()))
        H = fft(fftshift(self.h()))
        for k, X in enumerate([self.H_inf(), W, H]):
            X = fftshift(X)
            if self.display_samples:
                self.ax[k, 1].stem(n, abs(x), linefmt=col[k], basefmt = 'gray')
            else:
                self.ax[k, 1].plot(n, X.real, color=col[k])

            Xr = abs(X)/max(abs(X))
            X_db = 20*np.log10(Xr, out=-100*np.ones_like(Xr), where=(Xr!=0))
            self.ax[k, 2].plot(n, X_db, color=col[k])

        wscale = np.max(np.abs(self.h_inf()))
        w = 1.2*wscale*self.window()
        self.ax[0, 0].plot(n, w, 'C1')

        self.ax[0, 0].fill_between(n, w,
                                   facecolor = 'C1',
                                   alpha = 0.2)

        self.ax[0, 0].axvline(x=-self.n_w, color='C1')
        self.ax[0, 0].axvline(x=self.n_w, color='C1')

        self.scale_graphs()
        return 0

    def scale_graphs(self):
        """Scale graph axes."""

        xmax = self.n_max/4
        for ax in self.ax.flatten():
            ax.set_xlim(-xmax, xmax)

        wmax = np.sum(self.window())
        self.ax[1,1].set(ylim=(-0.4*wmax, 1.1*wmax))

        for ax in self.ax[:,2]:
            ax.set(ylabel='Magnitude [dB]',
                   ylim=(-30, 3),
                   yticks=np.arange(-48, 3, 6))
            ax.grid(True)

        arrow_pos = {'ha':'right', 'va':'baseline'}
        box_pos = {'ha':'center', 'va':'top'}
        boxcol = {'fc':'bisque', 'ec':'C1'}

        dfttext = [' IDFT ', ' DFT ']
        arrowstyle = ['larrow, pad=0.7', 'rarrow, pad=0.7']
        for k, ax in enumerate([self.ax[0, 1], self.ax[2, 1]]):
            ax.text(0.0, 0.5, dfttext[k],
            **arrow_pos, transform=ax.transAxes,
             bbox=dict(boxstyle=arrowstyle[k], **boxcol))

        optext = [' Multiplication ', ' Convolution ']
        for k, ax in enumerate([self.ax[1, 0], self.ax[1, 1]]):
            ax.text(0.5, -0.25, optext[k],
            **box_pos, transform=ax.transAxes,
            bbox=dict(boxstyle='square', **boxcol))

        return 0

    # Simple interactive operation

    def interact(self, n_w=None, window=None, winpar=None):
        """Scale inputs and  display results.

        For interactive operation.
        Existing values are used if a parameter is omitted.
        """
        if n_w is not None:
            self.n_w = n_w
        if window is not None:
            self.window_name = window
        if winpar is not None:
            self.winpar = winpar

        self.display()

        return

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         layout='constrained',
                         num='FIR Filter Design')
        ax = fig.subplots(3, 3)

        ax[0, 0].set_title(r'Ideal coefficients. $h^\infty(k)$')
        ax[1, 0].set_title(r'Truncation window $w(k)$')
        ax[2, 0].set_title(r'Truncated coeficients. $h(k) = h^\infty(k) w(k)$')
        ax[0, 1].set_title(r'Ideal frequency response. $|H^\infty(m)|$')
        ax[1, 1].set_title(r'Spectrum of window. $|W(m)|$')
        ax[2, 1].set_title(r'Actual frequency response. $|H(m)|=|H^\infty(m) * W(m)|$')
        ax[0, 2].set_title('Magnitude Reponse in dB')

        for a in ax[:, 0]:
            a.set(xlabel='k')

        for a in ax[:, 1]:
            a.set(xlabel='m')

        return ax

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('FIR lowpass filter design')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': widgets.Layout(width='15%'),
                       'style': {'description_width': '70%'}}

        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='70%'),
                         'style': {'description_width': '15%'}}

        dropdown_layout = {
            'layout': widgets.Layout(width='20%'),
            'style': {'description_width': '50%'}}

        # Individual widgets
        n_widget = widgets.IntSlider(
            min=1, max=self.n_max/4, value=self.n_w,
            description='Truncation [samples]',
            **slider_layout)

        window_widget = widgets.Dropdown(
            options=self.window_list,
            value=self.window_name,
            description='Window',
            **dropdown_layout)

        window_parameter_widget = widgets.FloatText(
            min = 0.5, max=100, value=self.winpar, step=0.5,
            description='Window Parameter',
            **text_layout)

        # Widget layout
        widget_line = widgets.HBox([window_widget, window_parameter_widget])
        widget_layout = widgets.VBox([title_widget, n_widget, widget_line])

        # Export as dictionary
        widget = {'n_w': n_widget,
                  'window': window_widget,
                  'window_parameter':window_parameter_widget
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
