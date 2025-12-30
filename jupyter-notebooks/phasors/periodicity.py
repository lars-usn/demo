"""Demonstrate periodic signals. Create signal as a sum of sine-waves."""


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget

        return


class Periodicity():
    """Demonstration of periodic signals."""

    def __init__(self, initialise_graphs=True, create_widgets=False):
        self.fs = 4000            # Sample rate [1/s]
        self.a = [1, 0.5, 0.2]    # Amplitudes
        self.f = [50, 100, 150]   # Frequencies [Hz]
        self.phi = [0, 90, 180]   # Phases [degrees]
        self.duration = 0.1       # Signal duration [s]

        if initialise_graphs:
            self.ax = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num='Periodicity Demo')

        ax = fig.subplots(2, 1, sharex=True)
        ax[1].set(xlabel='Time [s]')
        for a in ax:
            a.set(ylabel='Amplitude',
                  xlim=[0, self.duration],
                  ylim=3*np.array([-1, 1]),
                  )
            a.minorticks_on()
            a.grid(True, which='both', axis='x')
            a.grid(True, which='major', axis='y')

        return ax

    def f0(self):
        """Find fundamental frequency."""
        f_int = np.array(self.f).astype(int)  # Integer array needed by gcd
        return np.gcd.reduce(f_int)

    def T0(self):
        """Find fundamental period."""
        return 1/self.f0()

    def t(self):
        """Create time vector."""
        return np.arange(0, self.duration, 1/self.fs)

    def phase(self):
        """Calculate phase in radians."""
        return np.radians(self.phi)

    def n_frequencies(self):
        """Find number of frequencies to add."""
        return len(self.f)

    def signal(self):
        """Create three cos-waves and their sum."""
        # Individual cosine-signals
        for k in range(self.n_frequencies()):
            s = [self.a[k] * np.cos(2*pi*self.f[k]*self.t() + self.phase()[k])
                 for k in range(self.n_frequencies())]

        # Last element is the sum of the individual signals
        s.append(s[0]+s[1]+s[2])

        return s

    def display(self):
        """Plot all signals and spectra."""
        # Clear old lines and texts
        for k in range(len(self.ax)):
            for art in list(self.ax[k].lines):
                art.remove()
            for art in list(self.ax[k].texts):
                art.remove()

        # Get and plot signals
        s = self.signal()

        col = ['C1', 'C2', 'C3', 'C4', 'C5']
        # Individual signals
        for k in range(self.n_frequencies()):
            self.ax[0].plot(self.t(), s[k], color=col[k])

        # Summed signals
        self.ax[1].plot(self.t(), s[self.n_frequencies()], color='C0')

        # Add text-box with results
        textstr = '\n'.join([rf'$f_0= {self.f0():.0f}$ Hz',
                             rf'$T_0= {1/self.f0():.3f}$ s'])

        self.ax[1].text(0.90, 0.95, textstr,
                        transform=self.ax[1].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round',
                                  facecolor='aliceblue',
                                  alpha=0.50))

        return

    def interact(self,
                 a_0=None, a_1=None, a_2=None,
                 f_0=None, f_1=None, f_2=None,
                 phi_0=None, phi_1=None, phi_2=None):
        """Interactive operation."""
        for k, a in enumerate([a_0, a_1, a_2]):
            if a is not None:
                self.a[k] = a

        for k, f in enumerate([f_0, f_1, f_2]):
            if f is not None:
                self.f[k] = f

        for k, phi in enumerate([phi_0, phi_1, phi_2]):
            if phi is not None:
                self.phi[k] = phi

        self.display()

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title_widget = ipywidgets.Label('Periodic Signals',
                                        style=dict(font_weight='bold'))

        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='95%'),
                         'style': {'description_width': '10%'}}

        text_layout = {'continuous_update': True,
                       'layout': ipywidgets.Layout(width='95%'),
                       'style': {'description_width': '30%'}}

        frequency_widget = [ipywidgets.FloatText(
            min=10, max=500, value=f, step=10,
            description=rf' $f_{k} \:$ [Hz]',
            **text_layout)
            for k, f in enumerate(self.f)]

        amplitude_widget = [ipywidgets.FloatSlider(
            min=0, max=1.5, value=a, step=0.1,
            description=fr'$A_{k}$',
            **slider_layout)
            for k, a in enumerate(self.a)]

        phase_widget = [ipywidgets.FloatSlider(
            min=-360, max=360, value=phi, step=15,
            description=rf'$\phi_{k} \:$ [$\degree]$',
            **slider_layout)
            for k, phi in enumerate(self.phi)]

        frequency_col = ipywidgets.VBox(frequency_widget,
                                        layout=ipywidgets.Layout(width='15%'))
        amplitude_col = ipywidgets.VBox(amplitude_widget,
                                        layout=ipywidgets.Layout(width='35%'))
        phase_col = ipywidgets.VBox(phase_widget,
                                    layout=ipywidgets.Layout(width='35%'))

        widget_layout = ipywidgets.HBox([frequency_col,
                                         amplitude_col,
                                         phase_col],
                                        layout=ipywidgets.Layout(width='100%'))

        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'a_0': amplitude_widget[0],
                  'a_1': amplitude_widget[1],
                  'a_2': amplitude_widget[2],
                  'f_0': frequency_widget[0],
                  'f_1': frequency_widget[1],
                  'f_2': frequency_widget[2],
                  'phi_0': phase_widget[0],
                  'phi_1': phase_widget[1],
                  'phi_2': phase_widget[2]}

        w = WidgetLayout(widget_layout, widget)

        return w
