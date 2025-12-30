# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 18:00:12 2025

@author: larsh
"""

import ipywidgets as widgets
from math import pi
import numpy as np


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


def interact(filt, L, k0):
    """Scale inputs and  display results.

    For interactive operation.
    """
    filt.b, filt.a, w0 = coefficients(L, k0)

    filt.plot(print_coefficients=False,
              textstr=rf'$\hat \omega_0$={w0/pi:.2f}$\pi$')

    return


def coefficients(L, k0):
    """Calculate coefficients for bandpass FIR filter."""
    w0 = 2*pi*k0/L
    k = np.arange(0, L)
    km = 1/2 * (L-1)
    b = np.cos(w0*(k-km))
    a = [1]

    return b, a, w0


def create_widgets():
    """Create widgets for interactive operation."""
    # Title
    title = ('Simple Bandpass FIR-filter')
    title_widget = widgets.Label(title, style=dict(font_weight='bold'))

    text_layout = {'continuous_update': False,
                   'layout': widgets.Layout(width='10%'),
                   'style': {'description_width': '50%'}}

    L_widget = widgets.IntText(min=1, max=50,  value=15,
                               description=" $L$ ",
                               **text_layout)

    k0_widget = widgets.FloatText(min=0.0, max=50.0,  value=4.0, step=0.1,
                                  description=" $k_0$ ",
                                  **text_layout)

    widget_layout = widgets.HBox([title_widget, L_widget, k0_widget])

    # Export as dictionary
    widget = {'L': L_widget,
              'k0': k0_widget}

    w = WidgetLayout(widget_layout, widget)

    return w
