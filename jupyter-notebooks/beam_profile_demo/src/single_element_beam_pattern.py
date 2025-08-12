"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

import ipywidgets as widgets

import curve_analysis as ca


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


# Calculate and display lateral bem profile
class Transducer():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self,
                 create_lateral_graphs=False,
                 create_axial_graphs=False,
                 create_widgets=False):

        self.circular = False   # Circular or rectangular element
        self.width = 100e-3     # m   Element width (azimuth, x) or diameter
        self.height = 100e-3    # m   Element height (elevation, y)

        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.c = 1500           # m/s Speed of sound

        self.theta_max = 90    # deg  Max. angle to calculate
        self.d_max = 200e-3    # m    Max. dimension on element display
        self.x_max = 2.0       # m    Max. lateral dimension to calculate
        self.z_max = 20.0      # m    Max. depth to calculate

        self.z_ref = 10.0       # m    Reference depth
        self.db_min = -30      # dB   Min. on dB-scales

        self.x_sidelobe = np.nan

        self.y_lim = 0.5         # Relative limit for markers
        self.lim_text = '-6 dB'  # Text for markers

        self.element_color = 'crimson'
        self.element_background = 'whitesmoke'
        self.text_face = 'whitesmoke'

        self.lateral = False
        self.axial = False

        if create_lateral_graphs:
            self.lateral = True
            ax, fig = self._initialise_lateral_graphs()
            self.axes = ax
            self.fig = fig

            if create_widgets:
                self.widget = self._create_widgets()

        if create_axial_graphs:
            self.axial = True
            ax, fig = self._initialise_axial_graphs()
            self.axes = ax
            self.fig = fig

            if create_widgets:
                self.widget = self._create_widgets()

    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def d_theta(self):
        """Calculate opening angle from theory, two-sided, -6 dB."""
        if self.circular:
            x_6 = 0.705   # 6 dB limit, circular aperture
        else:
            x_6 = 0.603   # 6 dB limit, line (rectangular) aperture

        return 2*np.arcsin(x_6 * self.wavelength()/self.width)

    def z_r(self):
        """Rayleigh distance, far-field limit."""
        return self.width**2/(2*self.wavelength())

    def x(self):
        """Lateral dimension for axial plot (azimuth, x)."""
        return self._xz_points()[1]

    def z(self):
        """Axial dimension (depth) for axial plot (z)."""
        return self._xz_points()[0]

    def r(self):
        """Distance from aperture centre for axial plot."""
        return np.sqrt(self.z()**2 + self.x()**2)

    def theta(self):
        """Azimuth(x) angle to point(z, x)."""
        return np.arctan2(self.x(), self.z())

    def angle(self):
        """Angle argument, used in both azimuth and elevation."""
        theta_m = np.radians(self.theta_max)
        return np.linspace(-theta_m, theta_m, 301)

    def xy(self):
        """Lateral region to plot, plane at fixed axial distance."""
        pts = np.linspace(-self.x_max, self.x_max, 201)
        return np.meshgrid(pts, pts)

    def theta_xy(self):
        """Azimuth angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[0], self.z_ref)

    def phi_xy(self):
        """Elevation angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[1], self.z_ref)

    def theta_r(self):
        """Angle with z-axis for (xyz)-positions."""
        return np.arctan2(np.sqrt(self.xy()[0]**2+self.xy()[1]**2), self.z_ref)

    def w_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength()

    def h_lambda(self):
        """Aperture height relative to wavelength."""
        return self.height / self.wavelength()

    def p_axial(self):
        """Calculate axial pressure field (zx)."""
        if self.circular:
            p = 1/self.r() * ca.jinc(self.w_lambda() * np.sin(self.theta()))
        else:
            p = 1/self.r() * np.sinc(self.w_lambda() * np.sin(self.theta()))
        return p

    def set_dbmin(self, dbmin):
        """Set minimum on db-axes."""
        self.db_min = dbmin

        if self.lateral:
            self.axes['azimuth'].set_ylim(ymin=dbmin)
            self.axes['elevation'].set_ylim(ymin=dbmin)
            self.display_lateral()
        elif self.axial:
            self.axes['azimuth'].set_ylim(ymin=dbmin)
            self.display_axial()

        return

    def display_lateral(self):
        """Display lateral beam pattern in graphs."""
        for ax in self.axes.values():
            ca.remove_artists(ax)

        try:
            self.cbar.remove()
        except Exception:
            pass

        ax = self.axes

        # Element image
        self._draw_element(ax["element"])

        db_marker = ca.db(self.y_lim, p_ref=1)

        # Beam profiles
        if self.circular:
            p_x = self._p_circ(theta=self.angle())
            p_y = None
            p_xy = self._p_circ(theta=self.theta_r())
        else:
            p_x = self._p_rect(theta=self.angle(), phi=0)
            p_y = self._p_rect(theta=0, phi=self.angle())
            p_xy = self._p_rect(theta=self.theta_xy(), phi=self.phi_xy())

        self._plot_lateral_beamprofile(ax['azimuth'],
                                       np.degrees(self.angle()),
                                       ca.db(p_x, p_ref=0),
                                       y_marker=db_marker)

        if not self.circular:
            self._plot_lateral_beamprofile(ax['elevation'],
                                           np.degrees(self.angle()),
                                           ca.db(p_y, p_ref=0),
                                           y_marker=db_marker)

        # Lateral plane
        contour_format = {'colors': 'white',
                          'linestyles': 'dotted',
                          'alpha': 0.9}

        x_m = self.xy()[0][0, :]
        y_m = self.xy()[1][:, 0]
        im = ax['intensity'].pcolormesh(x_m, y_m,
                                        ca.db(p_xy, p_ref=0),
                                        vmin=self.db_min,
                                        cmap='magma')

        ax['intensity'].contour(x_m, y_m,
                                ca.db(p_xy, p_ref=0),
                                levels=[db_marker],
                                **contour_format)

        self.cbar = self.fig.colorbar(im, ax=ax['intensity'])
        ca.db_colorbar(self.cbar, db_sep=6)

        # Find -6 dB limits
        ref = ca.Refpoints(x=self.angle(), y=p_x)
        self.d_theta, _ = ref.lobe_width(y_rel=self.y_lim)

        if not (self.circular):
            ref = ca.Refpoints(x=self.angle(), y=p_y)
            self.d_phi, _ = ref.lobe_width(y_rel=self.y_lim)

        # Text box with results
        self._resulttext()

        return

    def display_axial(self):
        """Display axial beam pattern in graphs."""
        for ax in self.axes.values():
            ca.remove_artists(ax)

        try:
            self.cbar.remove()
        except Exception:
            pass

        marker_color = 'white'

        ax = self.axes

        # Intensity image
        x = self.x()[:, 0]
        z = self.z()[0, :]
        im = ax['intensity'].pcolormesh(z, x, ca.db(self.p_axial(), p_ref=0),
                                        vmin=self.db_min,
                                        cmap='magma')

        ax['intensity'].axvline(x=self.z_ref,
                                color=marker_color,
                                linestyle='dotted')

        y_element = self.width/2
        ax['intensity'].axhspan(-y_element, y_element, xmax=0.01,
                                color=self.element_color)

        for w in [-self.width, self.width]:
            ax['intensity'].plot([0.0, self.z_r()],
                                 w/2*np.array([1, 1]),
                                 color=self.element_color,
                                 linestyle='dotted')

        self.cbar = self.fig.colorbar(im, ax=ax['intensity'])
        ca.db_colorbar(self.cbar, db_sep=6)

        x_line = np.array([0, self.z_max])
        y_line = np.array([0, self.z_max*np.tan(self.d_theta()/2)])
        ax['intensity'].plot(x_line, y_line, x_line, -y_line,
                             color=marker_color,
                             linestyle='dotted')

        # Lateral beam profile
        k_ref = np.argmin(abs(z-self.z_ref))
        p = self.p_axial()[:, k_ref]    # Find pressure at reference distance
        p_db = ca.db(p, p_ref=self.p_axial().max())
        ax['azimuth'].plot(x, p_db, color='C0')

        indicator_line = {'color': 'C1',
                          'linestyle': 'solid'}

        ax['azimuth'].axhline(y=p_db.max()+ca.db(self.y_lim, p_ref=1),
                              **indicator_line)

        # Find reference values
        ref = ca.Refpoints(x=x, y=p)
        xl, _ = ref.ref_values(y_rel=self.y_lim)   # Beam width limits
        self.dx = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = ref.sidelobe()   # Highest sidelobe
        self.db_sidelobe = ca.db(self.y_sidelobe, p_ref=p.max())

        for x in xl:
            ax['azimuth'].axvline(x=x, **indicator_line)

        # Text box with results
        self._resulttext()

        return

    def interact(self,
                 circular=None,
                 frequency=None,
                 width=None,
                 height=None,
                 z_ref=None,
                 db_min=None
                 ):
        """Scale inputs and  display results.

        For interactive operation with  dimensions in mm and frequency in kHz.
        Existing values are used if a parameter is omitted.

        Parameters
        ----------
        circular: boolean, optional
            Circular (True) or rectangular (False) aperture
        frequency: float, optional
            Frequency in kHz
        widh: float, optional
            Element width (azimuth, x) in mm
        height: float, optional
            Element height (elevation, y) in mm
        z_ref: float, optional
            Reference depth in m
        db_min: float
            Minimum on dB-axes
        """
        if circular is not None:
            self.circular = circular
        if frequency is not None:
            self.frequency = 1e3*frequency
        if width is not None:
            self.width = 1e-3*width
        if height is not None:
            self.height = 1e-3*height
        if z_ref is not None:
            self.z_ref = float(z_ref)
        if db_min is not None:
            self.set_dbmin(db_min)

        # Reference distance cannot be smaller than Rayleigh distance
        self.z_ref = np.max([self.z_ref, self.z_r()])

        # Display result in graphs
        if self.lateral:
            self.display_lateral()
        if self.axial:
            self.display_axial()

        return

    ###################################################################
    # Non-public methods

    # Calculations, internal
    def _xz_points(self):
        """Mesh of points(z, x)."""
        return np.meshgrid(np.linspace(self.z_r(), self.z_max, 400),
                           np.linspace(-self.x_max, self.x_max, 201))

    def _p_circ(self, theta):
        """Calculate pressure field from circular element, 1D or 2D.

        Parameters
        ----------
        theta: array of float, 1D or 2D
            Angles to calculate pressure at
        """
        p = ca.jinc(self.w_lambda() * np.sin(theta))
        return p

    def _p_rect(self, theta=0, phi=0):
        """Calculate pressure field from rectangular element, 1D or 2D.

        Parameters
        ----------
        theta: array of float, 1D or 2D
            Azimuth angles
        phi: array of float, 1D or 2D
            Elevation angles
        """
        p = np.sinc(self.w_lambda() * np.sin(theta)) \
            * np.sinc(self.h_lambda() * np.sin(phi))

        return p

    # Graphs and results
    def _plot_lateral_beamprofile(self, ax, angle, y, y_marker=None,
                                  line_color='C0', marker_color="C1"):
        """Plot beam profile graph.

        Parameters
        ----------
        ax: axis object
            Axis to plot into
        angle: array of float
            Angle vector (x-axis)
        y: array of float
            Result vector (y-axis)
        y_marker: float, optional
            Position of marker line
        line_color: color string
            Graph color
        marker_color: color string
            Marker line color
        """
        ax.plot(angle, y, color=line_color)
        if not (y_marker is None):
            ax.axhline(y=y_marker, color=marker_color)

        return 0

    def _draw_element(self, ax):
        """Draw image of aperture in specified axis."""
        w_mm = self.width*1e3
        h_mm = self.height*1e3

        element_layout = {'fill': True,
                          'color': self.element_color}
        if self.circular:
            illustration = patches.Circle((0, 0), w_mm/2, **element_layout)
        else:
            illustration = patches.Rectangle((-w_mm/2, -h_mm/2), w_mm, h_mm,
                                             **element_layout)
        ax.add_patch(illustration)

        return 0

    def _resulttext(self):
        """Text box for lateral profile results."""
        header = (fr'Frequency  $f$ = '
                  fr'{self.frequency/1e3:.0f} kHz'
                  '\n'
                  r'Wavelength  $\lambda$ = '
                  f'{self.wavelength()*1e3:.0f} mm'
                  '\n')

        if not self.lateral:
            header += (fr'Rayleigh distance $z_r$ = '
                       fr'{self.z_r():.1f} m'
                       '\n'
                       fr'Reference distance {self.z_ref:.2} m'
                       '\n')

        header += '\n'

        if self.lateral:
            theta_0 = self.d_theta  # Lateral plot finds angles from results
        elif self.axial:            # Axial plot uses theoretical values
            theta_0 = self.d_theta()

        if self.circular or self.axial:    # Height dimension omitted
            text_1 = (f'Diameter $D$ = '
                      f'{self.width*1e3:.0f} mm = '
                      fr'{self.w_lambda():.1f} $\lambda$'
                      '\n'
                      fr'Opening angle ({self.lim_text})'
                      fr' $\theta_0$ = '
                      fr'{np.degrees(theta_0):.1f}$^\circ$'
                      )
        else:
            phi_0 = self.d_phi
            text_1 = ('Width $w$ = '
                      fr'{self.width*1e3:.0f} mm = '
                      fr'{self.w_lambda():.1f} $\lambda$'
                      '\n'
                      'Heigth $h$ = '
                      fr'{self.height*1e3:.0f} mm = '
                      fr'{self.h_lambda():.1f} $\lambda$'
                      '\n'
                      f'Opening angles ({self.lim_text})\n'
                      fr'  Azimuth $\theta_0$ = '
                      fr'{np.degrees(theta_0):.1f}$^\circ$'
                      '\n'
                      fr'  Elevation $\phi_0$ = '
                      fr'{np.degrees(phi_0):.1f}$^\circ$'
                      )

        if self.axial:
            beamwidth_text = ('\n\n'
                              fr'Beam width ({self.lim_text}): '
                              fr'{1e3*self.dx:.0f} mm'
                              )
        else:
            beamwidth_text = ' '

        if np.isnan(self.x_sidelobe):
            sidelobe_text = ''
        else:
            sidelobe_text = ('\n'
                             'Highest sidelobe: '
                             fr'$x$ = {self.x_sidelobe:.2f} m, '
                             fr'{self.db_sidelobe:.1f} dB'
                             )

        result_text = header + text_1 + beamwidth_text + sidelobe_text
        ca.remove_fig_text(self.fig)
        ca.set_fig_text(self.fig, result_text,
                        xpos=0.02, ypos=0.15)

        return

    def _initialise_lateral_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        ca.add_logo(fig)

        gs = GridSpec(2, 4, figure=fig)
        ax = {'element': fig.add_subplot(gs[0, 0]),
              'intensity': fig.add_subplot(gs[0:4, 1:3]),
              'azimuth': fig.add_subplot(gs[0, 3]),
              'elevation': fig.add_subplot(gs[1, 3]),
              }

        # Lateral figures: Element drawing and intensity plot
        for a in [ax['element'], ax['intensity']]:
            a.set(box_aspect=1,
                  xlabel='Azimuth [mm]',
                  ylabel='Elevation [mm]')

        element_max = self.d_max * 1e3 * np.array([-1, 1])
        ax["element"].set(title='Element shape',
                          xlim=element_max,
                          ylim=element_max,
                          facecolor=self.element_background)

        intensity_max = self.x_max*np.array([-1, 1])
        intensity_title = f'Intensity at {self.z_ref:.1f} m'
        ax["intensity"].set(title=intensity_title,
                            xlim=intensity_max,
                            ylim=intensity_max,
                            facecolor=self.element_background)

        # Directivity graphs
        angle_lim = self.theta_max * np.array([-1, 1])
        for a in [ax['azimuth'], ax['elevation']]:
            a.set(xlim=angle_lim,
                  xlabel='Angle [Deg]',
                  ylabel='Power [dB re. max]')

            ca.db_axis(a, db_min=self.db_min, db_max=0, db_sep=6)

            a.xaxis.set_major_locator(MultipleLocator(30))
            a.xaxis.set_minor_locator(MultipleLocator(10))
            a.grid(visible=True, which='both', axis='x')

        ax['azimuth'].set_title('Azimuth beam profile')
        ax['elevation'].set_title('Elevation beam profile')

        return ax, fig

    def _initialise_axial_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='Single Element Beamprofile')

        ca.add_logo(fig)

        gs = GridSpec(2, 2, figure=fig)
        ax = {'intensity': fig.add_subplot(gs[0, :]),
              'azimuth': fig.add_subplot(gs[1, 1])}

        # Lateral figures: Element drawing and intensity plot
        ax['intensity'].set(aspect='equal',
                            xlabel='Distance (z) [m]',
                            ylabel='Azimuth (x) [m]',
                            facecolor='lightgray')

        ax["intensity"].set(ylim=self.x_max*np.array([-1, 1]),
                            xlim=[0, self.z_max])

        # Directivity graphs
        ax['azimuth'].set(xlabel='Azimuth (x) [m]',
                          ylabel='Power [dB re. max]',
                          xlim=self.x_max*np.array([-1, 1]))

        ca.db_axis(ax['azimuth'], db_min=self.db_min, db_max=0, db_sep=6)

        ax['azimuth'].grid(visible=True, which='both', axis='x')

        return ax, fig

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        widget_style = {'description_width': 'initial'}

        title = 'Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        layout_settings = {'continuous_update': True,
                           'layout': widgets.Layout(width='80%'),
                           'style': widget_style}

        shape_widget = widgets.Dropdown(options=[('Rectangular', False),
                                                 ('Circular', True)],
                                        value=False,
                                        description='Shape',
                                        **layout_settings)

        db_widget = widgets.BoundedFloatText(
            value=-42, min=-120, max=-6, step=6,
            description='Dynamic range [dB]',
            **layout_settings)

        # Parameter widgets
        if self.lateral:
            label = ['Frequency', 'Width (Diameter)', 'Height']
            n_dim = 2

        elif self.axial:
            label = ['Frequency', 'Width (Diameter)', 'Ref. distance']
            n_dim = 1

        label_widget = [widgets.Label(labeltext,
                                      layout=widgets.Layout(width='20%'))
                        for labeltext in label]

        layout_settings = {'continuous_update': True,
                           'layout': widgets.Layout(width='80%'),
                           'style': widget_style}

        frequency_widget = widgets.FloatSlider(min=1, max=400,
                                               value=100, step=1,
                                               readout_format='.0f',
                                               description='[kHz]',
                                               **layout_settings)

        size_widget = [widgets.FloatSlider(min=1, max=400,
                                           value=100, step=1,
                                           readout_format='.0f',
                                           description='[mm]',
                                           **layout_settings)
                       for k in range(n_dim)]

        distance_widget = widgets.FloatSlider(min=1, max=10,
                                              value=2.0, step=0.1,
                                              readout_format='.1f',
                                              description='[m]',
                                              **layout_settings)

        if self.lateral:
            parameter_widget = [frequency_widget] + size_widget
        elif self.axial:
            parameter_widget = [frequency_widget] + \
                size_widget + [distance_widget]

        parameter_line = [widgets.HBox([label_widget[k], parameter_widget[k]])
                          for k in range(len(label))]

        col = [widgets.VBox([shape_widget, db_widget],
                            layout=widgets.Layout(width='30%')),
               widgets.VBox(parameter_line,
                            layout=widgets.Layout(width='70%'))]

        grid = widgets.HBox(col, layout=widgets.Layout(width='90%'))
        widget_layout = widgets.VBox([title_widget, grid],
                                     layout=widgets.Layout(width='90%'))

        if self.lateral:
            widget = {'circular': shape_widget,
                      'db_min': db_widget,
                      'frequency': frequency_widget,
                      'width': size_widget[0],
                      'height': size_widget[1],
                      }

        elif self.axial:
            widget = {'circular': shape_widget,
                      'db_min': db_widget,
                      'frequency': frequency_widget,
                      'width': size_widget[0],
                      'distance': distance_widget,
                      }

        w = WidgetLayout(widget_layout, widget)

        return w
