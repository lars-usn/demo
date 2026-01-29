"""
Created on Mon Dec 29 15:04:34 2025

@author: larsh
"""

import zplot     # Phasor plotting module for DSP First, USN-course TSE2280
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from cmath import phase, rect


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget

        return


class PhasorDemo:
    """Demonstation of phasor plots."""

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise system parameters."""
        self.a = [0.5, 1.0]          # Amplitude
        self.phi = [0.0, 0.5]

        if initialise_graphs:
            self.ax = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def interact(self, a_0=None, a_1=None, phi_0=None, phi_1=None):
        """Set values and call plotting function."""
        if a_0 is not None:
            self.a[0] = a_0
        if a_1 is not None:
            self.a[1] = a_1
        if phi_0 is not None:
            self.phi[0] = phi_0
        if phi_1 is not None:
            self.phi[1] = phi_1

        self.display()

        return

    def display(self):
        """Set values and call plotting function."""
        z = [rect(self.a[k], pi*self.phi[k]) for k in range(2)]

        # Remove old plot
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.texts):
                art.remove()

        labels = [' $z_1$ ', ' $z_2$ ']
        zplot.plot_phasor(z, labels=labels, include_sum=True, ax=self.ax[0])
        zplot.plot_signal(z, labels=labels, include_sum=True,
                          frequency=1, ax=self.ax[1])

        self.ax[0].set(xlim=[-2, 2], ylim=[-2, 2])
        self.ax[1].set(ylim=[-3, 3])

        zs = sum(z)
        textstr = '\n'.join([r'Summed signal $\Sigma z$',
                             rf'$A_s= {abs(zs):.2f}$',
                             rf'$\phi_s = {phase(zs)/pi:.2f}$ $\pi$'])

        textbox_style = dict(boxstyle='round', facecolor='white', alpha=0.90)
        self.ax[1].text(0.75, 0.98, textstr,
                        transform=self.ax[1].transAxes,
                        verticalalignment='top',
                        color='C1',
                        bbox=textbox_style)

        return

    def _initialise_graphs(self):
        """Initialise result graph."""
        plt.close('all')
        plt.rc('font', size=10)
        fig = plt.figure(figsize=[12, 5],
                         constrained_layout=True,
                         num='Phasor Demo')

        ax = [fig.add_subplot(1, 2, k+1) for k in range(2)]

        return ax

    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title_widget = ipywidgets.Label('Phasor Demo',
                                        style=dict(font_weight='bold'))

        label = [r'$z_1$', r'$z_2$']
        label_widget = [ipywidgets.Label(labeltext,
                                         layout=ipywidgets.Layout(width='10%'))
                        for labeltext in label]

        slider_layout = {'continuous_update': True,
                         'layout': ipywidgets.Layout(width='30%'),
                         'style': {'description_width': '20%'}}

        amplitude_widget = [ipywidgets.FloatSlider(
            min=0.0, max=2.0, step=0.01, value=a0,
            description=f'Amplitude {k+1}',
            **slider_layout)
            for k, a0 in enumerate(self.a)]

        phase_widget = [ipywidgets.FloatSlider(
            min=-2, max=2, step=0.1, value=phi0,
            description=rf'Phase {k+1} [$\pi$]',
            readout_format='.2f',
            **slider_layout)
            for k, phi0 in enumerate(self.phi)]

        # Build list of widgets, display in a grid
        widget_line = [ipywidgets.HBox([amplitude_widget[k], phase_widget[k]])
                       for k in range(2)]

        widget_layout = ipywidgets.VBox(widget_line)
        widget_layout = ipywidgets.VBox([title_widget, widget_layout])

        # Export as dictionary
        widget = {'a_0': amplitude_widget[0],
                  'a_1': amplitude_widget[1],
                  'phi_0': phase_widget[0],
                  'phi_1': phase_widget[1]}

        w = WidgetLayout(widget_layout, widget)

        return w
