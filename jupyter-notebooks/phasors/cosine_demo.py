"""
Created on Mon Dec 29 14:09:37 2025

@author: larsh
Demonstrate a cosine-wave

Illustrate the parameteres amplitude, frequency and phase, and relate them to
period and delay. Include sampling and normalized frequency
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class CosineDemo:
    """Demonstation of cosine-wave.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.a = 1.0          # Amplitude
        self.f = 1.0          # Frequency
        self.fs = 10          # sample rate
        self.phase = 0.0      # Phase [degrees]
        self.t_min = -0.5     # Start time
        self.t_max = 2.0      # End time
        self.a_max = 2.0      # Max. amplitude on plot
        self.show_samples = True  # Show or hide samples, as stem-plot

        if initialise_graphs:
            self.ax = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def _initialise_graphs(self):
        """Initialise result graph ."""
        plt.close('all')
        plt.rc('font', size=10)
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num='Cosine wave demo')
        ax = fig.add_subplot(1, 1, 1)

        ax.set(xlabel=r'Time $t$ [s]',
               ylabel=r'$s(t)$',
               title=r'$s(t)=Acos(2\pi f t + \phi)$',
               xlim=(self.t_min, self.t_max),
               ylim=self.a_max*np.array([-1, 1]))
        ax.grid(False)

        return ax

    def T(self):
        """Period of wave."""
        return 1/self.f

    def phi(self):
        """Phase in radians."""
        return np.radians(self.phase)

    def t(self):
        """Create time vector for 'continous' signal."""
        return np.linspace(self.t_min, self.t_max, 300)

    def ts(self):
        """Create time vector for samples."""
        return np.arange(self.t_min, self.t_max, 1/self.fs)

    def _cos_wave(self, a, f, t, phi):
        """Calculate cosine-wave, internal use."""
        s = a * np.cos(2*pi*f*t + phi)
        return s

    def s(self):
        """Get continous cosine-wave."""
        s = self._cos_wave(self.a, self.f, self.t(), self.phi())
        return s

    def ss(self):
        """Get sampled cosine-wave."""
        ss = self._cos_wave(self.a, self.f, self.ts(), self.phi())
        return ss

    def display(self):
        """Plot result in graph."""
        # Remove old plots and markers
        for art in list(self.ax.lines):
            art.remove()
        for art in list(self.ax.collections):   # Stem plots are collections
            art.remove()
        for art in list(self.ax.texts):
            art.remove()

        # Plot signal
        self.ax.plot(self.t(), self.s(), '-', color='C0')
        if self.show_samples:
            self.ax.stem(self.ts(), self.ss(), '-', linefmt='C0')

        # Markers and annotations
        self.ax.axvline(0, color='gray')  # Line at t=0
        self.ax.axhline(0, color='gray')  # Line at s=0

        y_text = 1.1*self.a  # y-position of markers and texts
        id = np.ones(2)  # Array [1, 1] to draw lines
        rot = 5  # Rotation of text

        # Amplitude marker
        col = 'C3'
        t0 = (1-self.phi()/(2*pi))/self.f  # Position of wave-crest, phase=2*pi
        self.ax.stem(t0, self.a, linefmt=col)
        self.ax.text(t0, self.a,
                     f' Amplitude= {self.a:.2f}',
                     color=col,
                     horizontalalignment='left',
                     rotation=rot,
                     verticalalignment='bottom')

        # Period marker
        t0 = (1-self.phi()/pi)/(2*self.f)  # Position of wave-through, phase=pi
        self.ax.plot([t0, t0+self.T()], -self.a*id, '-|', color=col)
        self.ax.text(t0+self.T()/2, -y_text,
                     f' Period $T= 1/f=${self.T():.2f} s',
                     color=col,
                     horizontalalignment='center',
                     verticalalignment='top')

        # Delay marker
        t_d = -self.phi()/(2*pi*self.f)  # Position of wave-crest, phase=0
        self.ax.plot([0, t_d], self.a*id, '-|', color=col)
        self.ax.text(t_d, self.a,
                     fr' Delay $t_d=-\phi/(2\pi f)=${t_d:.2f} s',
                     color=col,
                     horizontalalignment='left',
                     rotation=rot,
                     verticalalignment='bottom')

        # Sample rate text
        if self.show_samples:
            textstr = '\n'.join([rf'$f_s= {self.fs/self.f:.1f} f_0$',
                                 rf'$T_s= {1/self.fs:.2f} $ s',
                                 rf'$\hat \omega= {2*self.f/self.fs:.2f} \pi $ radians'])

            self.ax.text(0.85, 0.95, textstr,
                         transform=self.ax.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='aliceblue',
                                   alpha=0.50))

        return 0

    def interact(self, a=None, f=None, phase=None, fs=None, show_samples=None):
        """Set values and call plotting function."""
        if a is not None:
            self.a = a
        if f is not None:
            self.f = f
        if phase is not None:
            self.phase = phase
        if fs is not None:
            self.fs = fs
        if show_samples is not None:
            self.show_samples = show_samples

        self.display()

        return

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title_widget = ipywidgets.Label('Cosine wave demo',
                                        style=dict(font_weight='bold'))

        # Layouts definitions
        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='95%'),
                         'style': {'description_width': '20%'}}

        # Individual widgets
        a_widget = ipywidgets.FloatSlider(
            min=0, max=2, value=1.0, step=0.01,
            description='Amplitude ',
            readout_format='.2f',
            **slider_layout)

        f_widget = ipywidgets.FloatSlider(
            min=0.1, max=5, value=1, step=0.01,
            description='Frequency [Hz]',
            readout_format='.2f',
            **slider_layout)

        phase_widget = ipywidgets.FloatSlider(
            min=-360, max=360, value=0, step=5,
            description='Phase [deg]',
            readout_format='.0f',
            **slider_layout)

        fs_widget = ipywidgets.FloatSlider(
            min=0.5, max=50, value=10, step=0.5,
            description='Sample Rate [samples/s]',
            readout_format='.1f',
            **slider_layout)

        show_samples_widget = ipywidgets.ToggleButton(
            value=False,
            description='Show samples',
            button_style='info',
            layout=ipywidgets.Layout(width='10%'))

        slider_lines = ipywidgets.VBox(
            [fs_widget, a_widget, f_widget, phase_widget], layout=ipywidgets.Layout(width='60%'))

        widget_layout = ipywidgets.HBox([show_samples_widget, slider_lines])
        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'a': a_widget,
                  'f': f_widget,
                  'phase': phase_widget,
                  'fs': fs_widget,
                  'show_samples': show_samples_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
