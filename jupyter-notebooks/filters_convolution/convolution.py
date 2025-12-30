# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:21:59 2025

@author: larsh
"""

# Illustration of the convolution operation, applied to FIR filtering
from math import pi
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import ipywidgets


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class ConvolutionPlot():
    """Plot all resouts of convolution."""

    def __init__(self):
        self.n = 0
        self.f = 0.05
        self.h_length = 5
        noise_level = 0.2

        self.xspan = 80
        self.ymax = 1.5

        n_samples = 500
        n = np.arange(n_samples)

        rng = np.random.default_rng()
        noise = noise_level * rng.standard_normal(n_samples)

        self.s = np.cos(2*pi*self.f*n) + noise

        self.ax = self.initialise_graphs()
        self.color = self.initialise_colors()
        self.widget = self._create_widgets()


    def h(self):
        return 1/self.h_length * np.ones(self.h_length)

    def initialise_colors(self):
        """Define standard colors for graphs."""
        color = {}
        color["signal"] = "C0"
        color["filter"] = "C3"
        color["indicator"] = "C2"
        color["result"] = "C1"
        color["baseline"] = " "

        return color

    def initialise_graphs(self):
        """Initialise graphs for signals and spectra.

        Returns
        -------
        ax : List of axis objects
            Axes where results are plotted
        """
        fig = plt.figure(figsize=[14, 6],
                         constrained_layout=True,
                         num="Convolution Demo")

        gs = fig.add_gridspec(3, 3)
        ax = [fig.add_subplot(gs[k, :]) for k in range(3)]

        ax[0].set_title("Input $x(n)$")
        ax[1].set_title("Filter $h(k)$")
        ax[2].set_title("Output $y(n) = \sum h(k) x(n-k)$")

        flip = [True, False, True]

        x_start = -0.4 * self.xspan
        x_end = 0.6 * self.xspan
        for k, a in enumerate(ax):
            if flip[k]:
                xlim = np.array([-x_start, -x_end])
                xticks = np.arange(-x_start, -x_end, -2)
            else:
                xlim = np.array([x_start, x_end])
                xticks = np.arange(x_start, x_end, 2)

            a.set(xlabel='Sample [n]',
                  xlim=xlim,
                  xticks=xticks)

            a.axvline(x=0, color='gray')
            a.grid(True)

        ax[0].text(15,self.ymax, "Future",
            ha='center', va='center',
            bbox=dict(boxstyle="larrow",
                      fc="lightblue", ec="steelblue"))

        ax[0].text(-15, self.ymax, "Past",
            ha='center', va='baseline',
            bbox=dict(boxstyle="rarrow",
                      fc="lightblue", ec="steelblue"))

        ax[0].text(0, self.ymax, "Now",
            ha='center', va='baseline',
            bbox=dict(boxstyle="square",
                      fc="lightblue", ec="steelblue"))

# =============================================================================
#         for k in [0, 2]:
#             ax[k].xaxis.set_inverted(True)
# =============================================================================

        return ax

    def display(self):
        """Plot all signals and spectra.

        Arguments
        ---------
        s : Signal object
            Input signal
        h : Signal object
            Filter impulse response
        n : int
            Sample position to show in plot
        """
        # Clear old graphs, add indicators
        for ax in self.ax:
            for art in list(ax.lines):
                art.remove()
            for art in list(ax.collections):
                art.remove()
            for art in list(ax.patches):
                art.remove()

            ax.axvline(x=0, color='gray')
            ax.axhline(y=0, color='gray')

        # Calculate result of convolution
        n = self.n
        s = self.s
        h = self.h()
        y = convolve(s, h)

        # Plot input and filter response
        n_max = len(s)
        n_plot = np.arange(0, n_max) - n
        self.ax[0].stem(n_plot, s,
                        linefmt=self.color["signal"],
                        basefmt=self.color["baseline"])
        self.ax[1].stem(h,
                        linefmt=self.color["filter"],
                        basefmt=self.color["baseline"])

        y = y[0:n_max]
        y[n+1:] = 0
        self.ax[2].stem(n_plot, y,
                        linefmt=self.color["result"],
                        basefmt=self.color["baseline"])

        self.ax[2].stem(n_plot[n], y[n],
                        linefmt=self.color["indicator"],
                        basefmt=self.color["baseline"])

        # Mark filter support
        x_lim = len(h)-1
        for k in [0, 1]:
            if self.ax[k].xaxis_inverted():
                xmin = -x_lim
                xmax = 0
            else:
                xmin = 0
                xmax = x_lim

            self.ax[k].axvspan(xmin=xmin, xmax=xmax,
                               facecolor=self.color["indicator"], alpha=0.20)

         # Scale y-axes
        for k in [0, 2]:
            self.ax[k].set_ylim(-self.ymax, self.ymax)

        self.ax[1].set_ylim(0, 0.5)


        return 0


    def interact(self, n=None):
        if n is not None:
            self.n = n

        self.display()

        return

       # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        # Title
        title = 'FIR-filter as Convolution'
        title_widget = ipywidgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {
            'continuous_update': False,
            'layout': ipywidgets.Layout(width='15%'),
            'style': {'description_width': '50%'}}

        slider_layout = {
            'continuous_update': True,
            'layout': ipywidgets.Layout(width='95%'),
            'style': {'description_width': '15%'}}

        # Individual widgats
        n_widget = ipywidgets.IntText(min=-self.xspan,
                                      max=self.xspan,
                                      value=10,
                                      description='Time [n]',
                                      readout_format='.1f',
                                      **text_layout)

        # Arrange in columns and lines
        widget_layout = ipywidgets.HBox([title_widget, n_widget])

        # Export as dictionary
        widget = {'n': n_widget}

        w = WidgetLayout(widget_layout, widget)

        return w
