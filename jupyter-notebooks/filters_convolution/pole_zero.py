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


def interact(filt, n_zp,
             include_0, r_0, theta_0,
             include_1, r_1, theta_1,
             include_2, r_2, theta_2,
             include_3, r_3, theta_3,
             include_4, r_4, theta_4,
             include_5, r_5, theta_5,
             gain):
    """For interactive operation."""
    zero_list = [False for k in range(n_zp)]
    pole_list = [True for k in range(n_zp)]
    pole = zero_list + pole_list    # First values are zeros, last are poles

    # Hard code single values, poor support for lists in Jupyter widgets
    include = [include_0, include_1, include_2,
               include_3, include_4, include_5]
    r = [r_0, r_1, r_2, r_3, r_4, r_5]
    theta = [theta_0, theta_1, theta_2, theta_3, theta_4, theta_5]

    # Use only zeros/poles marked with 'include'
    r = np.extract(include, r)
    theta = np.extract(include, theta)
    pole = np.extract(include, pole)

    # Set values
    filt.set_polar_roots(r, theta*pi, pole, gain)
    filt.plot()

    return


def create_widgets():
    """Create widgets for interactive operation."""
    # Title
    title = ('Pole-zero Demo')
    title_widget = widgets.Label(title, style=dict(font_weight='bold'))

    label_layout = {'layout': widgets.Layout(width='5%')}

    r_theta_layout = {'layout': widgets.Layout(width='30%'),
                      'style': {'description_width': '50%'}}

    include_layout = {'layout': widgets.Layout(width='5%'),
                      'style': {'description_width': '5%'}}

    gain_layout = {'layout': widgets.Layout(width='10%'),
                   'style': {'description_width': '60%'}}

    label = ['Zeros', 'Poles', 'Gain']
    label_widget = [widgets.Label(labeltext,
                                  **label_layout)
                    for labeltext in label]

    n_zp = 3    # Number of zeros (First n_z cvalues interpreted as zeros)

    width = '25%'

    # Magnitude widgets
    r = np.concatenate((np.ones(n_zp), 0.9*np.ones(n_zp)))
    r_widget = [widgets.FloatText(min=0, max=4.0,  step=0.05, value=val,
                                  description=" $r$ ",
                                  format='.2f',
                                  **r_theta_layout)
                for val in r]

    theta = np.concatenate((np.zeros(n_zp), np.zeros(n_zp)))
    theta_widget = [widgets.FloatText(min=-2.0, max=2.0, step=0.01, value=val,
                                      description=r" $\theta$ [rad/$\pi$]",
                                      format='.2f',
                                      **r_theta_layout)
                    for val in theta]

    include = [False for k in range(2*n_zp)]
    include[0] = include[n_zp] = True
    include_widget = [widgets.Checkbox(value=val,
                                       description=' ',
                                       **include_layout)
                      for val in include]

    gain_widget = widgets.FloatText(min=0.0, max=1000.0, step=0.01, value=1,
                                    description='Gain ',
                                    readout_format='.3f',
                                    **gain_layout)

    # Create lines of parameter widgets
    zeros_line = [widgets.HBox([include_widget[k],
                                r_widget[k],
                                theta_widget[k]])
                  for k in range(n_zp)]

    zeros_grid = widgets.VBox([line for line in zeros_line])

    poles_line = [widgets.HBox([include_widget[k],
                                r_widget[k],
                                theta_widget[k]])
                  for k in range(n_zp, 2*n_zp)]

    poles_grid = widgets.VBox([line for line in poles_line])
    full_grid = widgets.HBox([label_widget[0],
                              zeros_grid,
                              label_widget[1],
                              poles_grid])

    widget_layout = widgets.VBox(
        [title_widget, full_grid, gain_widget])

    # Export as dictionary
    widget = {'include': include_widget,
              'r': r_widget,
              'theta': theta_widget,
              'gain': gain_widget
              }
    w = WidgetLayout(widget_layout, widget)

    return w, n_zp
