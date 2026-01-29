import numpy as np
from math import pi
import matplotlib.pyplot as plt
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class FrequencyResponse():
    """Demonstation of first order system frequency response.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.tau = 0.1        # Time constant
        self.n_f = 300        # Number of points in frequency vectors
        self.flim = [-1, 3]   # Frequency limits, logarithmic

        if initialise_graphs:
            self.ax = self.initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def initialise_graphs(self):
        """Initialise result graph ."""
        plt.close('all')
        plt.rc('font', size=10)          # Default text sizes
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='First Order System - Frequency Response')
        ax = fig.subplots(2, 1, sharex=True)

        ax[1].set(xlabel='Frequency [Hz]')
        for a in ax:    # Common for both plots
            a.set(xlim=(1e-1, 1e3))
            a.grid(True, which='major', axis='both')
            a.grid(True, which='minor', axis='x')

        db_min = -24
        ax[0].set(ylabel='Magnitude',
                         ylim=(db_min, 2),
                         yticks=np.arange(db_min, 2, 3))

        phi_min = -90
        ax[1].set(ylabel='Phase [Degrees]',
                  ylim=(phi_min, 0),
                  yticks=np.arange(phi_min, 1, 15))

        return ax

    def f(self):
        """Create frequency vector."""
        return np.logspace(min(self.flim), max(self.flim), self.n_f)

    def fc(self):
        """Calculate cut-off frequency."""
        return 1/(2*pi*self.tau)

    def H(self):
        """Calculate frequency response."""
        return 1/(1 + 1j*self.f()/self.fc())

    def display(self):
        """Plot result in graph."""
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.texts):
                art.remove()
            for art in list(ax.patches):
                art.remove()

        # Plot curves
        h_db = 20*np.log10(abs(self.H()))
        self.ax[0].semilogx(self.f(), h_db, '-', color='C0')

        self.ax[1].semilogx(self.f(), np.degrees(
            np.angle(self.H())), '-', color='C0')

        # Indicators
        indicator_color = 'gray'

        # Cut-off frequency
        for ax in self.ax:
            ax.axvline(x=self.fc(), color='C1', linestyle='--')
        self.ax[0].text(
            self.fc(), -20, f' $ f_c$={self.fc():.3g} Hz', color='black')

        # -3 dB limits
        y_lim = [-3, 0]
        for y in y_lim:
            self.ax[0].axhline(y=y, color=indicator_color, linestyle='-')
        self.ax[0].axhspan(y_lim[0], y_lim[1], color='green', alpha=0.1)

        # -45 degree phase
        self.ax[1].axhline(y=-45, color=indicator_color, linestyle='-')

        # Text box with values
        indicator_text = ('\n'
                          r'Time constant  $\tau$='
                          f'{self.tau:.2g} s \n'
                          r'Cut-off frequency $f_c=\frac{1}{2 \pi \tau}$='
                          f'{self.fc():.3g} Hz')

        box_style = dict(boxstyle='round',
                         facecolor='aliceblue',
                         pad=1.0)

        self.ax[1].text(30, -15, indicator_text, bbox=box_style, va='top')

        return

    def interact(self, tau=None):
        """Set values and call plotting function."""
        if tau is not None:
            self.tau = tau

        self.display()
        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title_widget = ipywidgets.Label(
            'First Order System Frequency Response',
            style=dict(font_weight='bold'))

        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='60%'),
                         'style': {'description_width': '15%'}}

        tau_widget = ipywidgets.FloatLogSlider(
            min=-4, max=1, value=1, step=0.01,
            description='Time constant [s]',
            readout_format='.2g',
            **slider_layout)

        widget_layout = ipywidgets.VBox([title_widget, tau_widget])

        # Export as dictionary
        widget = {'tau': tau_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
