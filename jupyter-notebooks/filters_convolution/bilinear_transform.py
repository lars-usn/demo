"""Demonstrate Bilienear Transform. s-plane to z-plane"""
# Import libraries

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Transform():
    """Demonstation bilinear tans form.

    All calculations and plotting routines are contained in this class
    """

    def __init__(self, initialise_graphs=True, create_widgets=False):
        """Initialise signal."""
        # Incoming signal, two frequencies
        self.s = -1 + 0.5j
        self.bilinear = True

        if initialise_graphs:
            self.ax = self._initialise_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

    def z(self):
        if self.bilinear:
            z = (1 + self.s/2)/(1 - self.s/2)
        else:
            z = np.exp(self.s)

        return z


    def display(self):
        """Plot result."""

        for k, x in enumerate([self.s, self.z()]):
            self.marker[k][0].remove()
            self.marker[k] = self.ax[k].plot(x.real, x.imag,
                                             marker='o', color='C0')
        return

    # Simple interactive operation
    def interact(self, sigma=None, omega=None, bilinear=None):
        """For interactive operation.

        Existing values are used if a parameter is omitted.
        """
        if sigma is not None:
            self.s = sigma + 1j*self.s.imag
        if omega is not None:
            self.s = self.s.real + 1j*omega
        if bilinear is not None:
            self.bilinear = bilinear

        self.display()

        return

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra."""
        plt.close('all')
        fig = plt.figure(figsize=[8, 4],
                         layout='constrained',
                         num='Bilinear Transform')
        ax = fig.subplots(1, 2)

        sigma_max = 4
        omega_max = 8
        ax[0].set(title = r'$s$-domain (Continous, Laplace)',
                  xlim = (-sigma_max, sigma_max),
                  ylim = (-omega_max, omega_max),
                  xlabel = r'$\sigma = Re\{s\}$',
                  ylabel = r'$\omega = Im\{s\}$')

        ax[1].set(title =r'$z$-domain (Discrete)',
                  xlim = (-3, 3),
                  ylim = (-3, 3),
                  xlabel = r'$Re\{z\}$',
                  ylabel = r'$Im\{z\}$')

        for k in range(2):
            ax[k].axhline(y=0, color='grey')
            ax[k].axvline(x=0, color='grey')

        # Mark regions
        for m in [-1, 1]:
            ax[0].axhline(y=m*pi, color=('green', 0.5), linestyle='dashed')

        ax[0].text(0, -pi, ' $-\pi$', va='top', color='green')
        ax[0].text(0, pi, ' $\pi$', va='baseline', color='green')


        unstablecolor = ('red', 0.2)
        # Unit circle in z-plane
        circ = plt.Circle((0, 0), radius=1,
                              edgecolor='gray',
                              facecolor='white')

        ax[0].set_aspect(1)
        ax[1].set_aspect(1)
        ax[1].set_facecolor(unstablecolor)
        ax[1].add_artist(circ)

        # Right half plane in s-plane
        ax[0].fill_betweenx([-10, 10], x1=10, color=unstablecolor)


        # Initialise with single point in origos
        self.marker=[]
        for k in range(2):
            self.marker.append(ax[k].plot(0 ,0))

        return ax

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = ('Bilinear Transform. s-plane to z-plane ')
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        slider_layout = {'continuous_update': True,
                         'layout': widgets.Layout(width='50%'),
                         'style': {'description_width': '10%'}}

        checkbox_layout = {
            'layout': widgets.Layout(width='20%'),
            'style': {'description_width': '40%'}}

        # Individual widgets
        sigma_widget = widgets.FloatSlider(
            min=-8, max=8, value=self.s.real,
            description='$\sigma = $Re{$s$}',
            **slider_layout)

        omega_widget = widgets.FloatSlider(
            min=-8, max=8, value=self.s.imag,
            description='$\omega = $Im{$s$}',
            **slider_layout)

        bilinear_widget = widgets.Checkbox(
            value=self.bilinear,
            description='Use Bilinear',
            **checkbox_layout)


        # Widget layout
        widget_header = widgets.HBox([title_widget,
                                      bilinear_widget])

        widget_layout = widgets.VBox([widget_header,
                                      sigma_widget,
                                      omega_widget])

        # Export as dictionary
        widget = {'bilinear': bilinear_widget,
                  'sigma': sigma_widget,
                  'omega': omega_widget
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
