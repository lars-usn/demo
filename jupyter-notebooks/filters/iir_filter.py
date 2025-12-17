# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 18:00:12 2025

@author: larsh
"""

import filter_response
import ipywidgets as widgets
# from math import pi
import numpy as np


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget

# Interactive widgets


def interact(filt, b_text=None, a_text=None):
    """Scale inputs and  display results.

    For interactive operation.
    Existing values are used if a parameter is omitted.
    """
    if b_text is not None:
        filt.b = np.fromstring(b_text, sep=' ')
    if a_text is not None:
        filt.a = np.fromstring(a_text, sep=' ')

    filt.plot(print_coefficients=False)

    return


def create_widgets():
    """Create widgets for interactive operation."""
    # Title
    title = ('IIR-filter')
    title_widget = widgets.Label(title, style=dict(font_weight='bold'))

    label_layout = {'layout': widgets.Layout(width='10%')}

    text_layout = {'continuous_update': False,
                   'layout': widgets.Layout(width='25%'),
                   'style': {'description_width': '30%'}}

    label = ["Forward coefficients", "Backward coefficients"]
    label_widget = [widgets.Label(labeltext,
                                  **label_layout)
                    for labeltext in label]

    coefficient_placeholder = '0000000000'
    b_widget = widgets.Text(value='1 1 1 1 1 1 1 1',
                            placeholder=coefficient_placeholder,
                            description=' $b$ ',
                            **text_layout)

    a_widget = widgets.Text(value='1',
                            placeholder=coefficient_placeholder,
                            description=' $a$ ',
                            **text_layout)

    b_widget_line = widgets.HBox([label_widget[0], b_widget])
    a_widget_line = widgets.HBox([label_widget[1], a_widget])

    widget_layout = widgets.VBox(
        [title_widget, b_widget_line, a_widget_line])

    # Export as dictionary
    widget = {'b': b_widget,
              'a': a_widget
              }
    w = WidgetLayout(widget_layout, widget)

    return w
