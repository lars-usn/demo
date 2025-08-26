"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

import curve_analysis as ca


class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class Transducer():
    """Define, calculate, and display transducer beam profile."""

    def __init__(self, create_widgets=False):

        # Transducer definition
        self.circular = False   # Circular or rectangular element
        self.width = 100e-3     # m   Element width (azimuth, x) or diameter
        self.height = 200e-3    # m   Element height (elevation, y)
        self.frequency = 100e3  # Hz  Ultrasound frequency
        self.c = 1500           # m/s Speed of soundin load medium

        # Calculation settings
        self.z_ref = 20.0        # m    Reference depth
        self.y_lim = 0.5         # Relative limit for beamwidth
        self.lim_text = '-6 dB'  # Text for markers
        self.x_sidelobe = np.nan

        self.azimuth = True    # Show azimuth (x) or elevation (y) profile

        # Display settings
        self.theta_max = 90    # deg  Max. angle to calculate
        self.d_max = 200e-3    # m    Max. dimension on element display
        self.x_max = 15.0       # m    Max. lateral dimension to calculate
        self.z_max = 100.0      # m    Max. depth to calculate
        self.db_min = -42      # dB   Min. on dB-scales

        # Colors and markers
        self.element_color = 'crimson'
        self.element_background = 'silver'
        self.text_face = 'whitesmoke'

        contour_color = 'white'
        self.contour_line = {'colors': contour_color,
                             'linestyles': 'dotted',
                             'alpha': 0.7}

        self.angle_line = {'color': contour_color,
                           'linestyle': 'dotted',
                           'alpha': 0.7}

        self.orientation_line = {'color': 'cyan',
                                 'linestyle': 'dotted',
                                 'alpha': 1.0}

        self.indicator_line = {'color': 'C1',
                               'linestyle': 'solid'}

        self.main_line = {'color': 'C0',
                          'linestyle': 'solid'}

        self.colormap = 'inferno'
        self.intensity_background = 'black'

        # Initialisation
        ax, fig = self._initialise_graphs()
        self.axes = ax
        self.fig = fig
        self.scale_axes()

        if create_widgets:
            self.widget = self._create_widgets()

    # === Calculated parameters ===========================
    def wavelength(self):
        """Calculate acoustic wavelenght."""
        return self.c/self.frequency

    def w_lambda(self):
        """Aperture width relative to wavelength."""
        return self.width / self.wavelength()

    def h_lambda(self):
        """Aperture height relative to wavelength."""
        return self.height / self.wavelength()

    def theta_0(self):
        """Calculate opening angle from theory, two-sided, -6 dB."""
        if self.circular:
            x_6 = 0.705   # 6 dB limit, circular aperture
        else:
            x_6 = 0.603   # 6 dB limit, line (rectangular) aperture

        if self.azimuth or self.circular:
            d = self.width
        else:
            d = self.height

        return 2*np.arcsin(x_6 * self.wavelength()/d)

    def z_r(self):
        """Rayleigh distance, far-field limit."""
        if self.circular:
            d = self.width
        else:
            d = np.max([self.width, self.height])

        return d**2/(2*self.wavelength())

    def z_c(self):
        """Limit reference distance to outside far-field limit."""
        return np.max([self.z_ref, self.z_r()])

    # === Axial plane ===================
    def x(self):
        """Lateral dimension for axial plot (x or y)."""
        return np.linspace(-self.x_max, self.x_max, 201)

    def z(self):
        """Axial dimension (depth) for axial plot (z)."""
        return np.linspace(self.z_r(), self.z_max, 400)

    def zx(self):
        """Axial plane (zx or zy) to plot."""
        pts = np.meshgrid(self.z(), self.x())
        return pts

    def r(self):
        """Distance from aperture centre for axial plot."""
        return np.sqrt(self.zx()[0]**2 + self.zx()[1]**2)

    def theta(self):
        """Azimuth(x) angle to point(z, x)."""
        return np.arctan2(self.zx()[1], self.zx()[0])

    def p_azimuth(self):
        """Calculate axial pressure field in the azimuth plane (zx)."""
        if self.circular:
            p = 1/self.r() * ca.jinc(self.w_lambda() * np.sin(self.theta()))
        else:
            p = 1/self.r() * np.sinc(self.w_lambda() * np.sin(self.theta()))
        return p

    def p_elevation(self):
        """Calculate axial pressure field in the lateral plane (zy)."""
        if self.circular:
            p = self.p_azimuth()
        else:
            p = 1/self.r() * np.sinc(self.h_lambda() * np.sin(self.theta()))
        return p

    # === Lateral plane ===================
    def xy(self):
        """Lateral region to plot, plane at fixed axial distance."""
        pts = np.linspace(-self.x_max, self.x_max, 201)
        return np.meshgrid(pts, pts)

    def theta_xy(self):
        """Azimuth angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[0], self.z_c())

    def phi_xy(self):
        """Elevation angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[1], self.z_c())

    def theta_r(self):
        """Angle with z-axis for (xyz)-positions."""
        r = np.sqrt(self.xy()[0]**2+self.xy()[1]**2)   # Radial distance
        return np.arctan2(r, self.z_c())

    def r_xy(self):
        """Radial distances at lateral plane."""
        r = np.sqrt(self.xy()[0]**2+self.xy()[1]**2)   # Radial distance
        return np.sqrt(r**2 + self.z_c()**2)

    def p_lateral(self):
        """Calculate lateral intensity at reference distance."""
        if self.circular:
            p = 1/self.r_xy() \
                * ca.jinc(self.w_lambda() * np.sin(self.theta_r()))
        else:
            p = 1/self.r_xy()  \
                * np.sinc(self.w_lambda() * np.sin(self.theta_xy())) \
                * np.sinc(self.h_lambda() * np.sin(self.phi_xy()))

        return p

    # === Commands =============================
    def display(self):
        """Display beam pattern in graphs."""
        self._remove_old_artists()
        ax = self.axes

        # Element image
        self._draw_element(ax["element"])
        self._draw_orientationline(ax["element"])

        # Axial intensity
        x = self.x()
        z = self.z()
        p_max = np.max([np.max(abs(self.p_azimuth())),
                        np.max(abs(self.p_elevation()))])

        if self.azimuth:
            p_axial = self.p_azimuth()
        else:
            p_axial = self.p_elevation()

        p_db = ca.db(p_axial, p_ref=p_max)

        im = ax['axial'].pcolormesh(z, x, p_db,
                                    vmin=self.db_min, cmap=self.colormap)
        if self.azimuth:
            ax['axial'].set(ylabel='Azimuth (x) [m]',
                            title='Beam intensity, azimuth plane (zx)')
        else:
            ax['axial'].set(ylabel='Elevation (y) [m]',
                            title='Beam intensity, elevation plane (zy)')

        ax['axial'].axvline(x=self.z_c(), **self.orientation_line)

        self.cbar = self.fig.colorbar(im, ax=ax['axial'])
        ca.db_colorbar(self.cbar, db_sep=6)

        # Element with lines extending to Rayleigh distance
        y_element = self.width/2
        ax['axial'].axhspan(-y_element, y_element, xmax=0.005,
                            color=self.element_color)

        for w in [-self.width, self.width]:
            ax['axial'].plot([0.0, self.z_r()],
                             w/2*np.array([1, 1]),
                             color=self.element_color,
                             linestyle='dotted')

        illustration = patches.Rectangle((0, -y_element),
                                         self.z_r(), self.width,
                                         fill=True,
                                         color=self.element_color,
                                         alpha=0.7)

        ax['axial'].add_patch(illustration)

        # Lines showing opening angle
        x_line = np.array([0, self.z_max])
        y_line = np.array([0, self.z_max*np.tan(self.theta_0()/2)])
        ax['axial'].plot(x_line, y_line, x_line, -y_line,
                         **self.angle_line)

        # Lateral intensity at reference distance
        x_m = self.xy()[0][0, :]
        y_m = self.xy()[1][:, 0]
        p_db = ca.db(self.p_lateral(), p_ref=p_max)
        im = ax['lateral'].pcolormesh(x_m, y_m, p_db,
                                      vmin=self.db_min,
                                      vmax=0,
                                      cmap=self.colormap)

        db_marker = ca.db(self.y_lim, p_ref=1) + np.max(p_db)
        ax['lateral'].contour(x_m, y_m, p_db,
                              levels=[db_marker],
                              **self.contour_line)

        self._draw_orientationline(ax["lateral"])

        ax["lateral"].set_title(f'Intensity at {self.z_c():.1f} m')

        # Lateral beam profile at reference distance
        k_ref = np.argmin(abs(z-self.z_c()))
        p = p_axial[:, k_ref]
        p_db = ca.db(p, p_ref=p_max)

        ax['beamprofile'].plot(x, p_db, **self.main_line)
        ax['beamprofile'].axhline(y=p_db.max()+ca.db(self.y_lim, p_ref=1),
                                  **self.indicator_line)

        # Find reference values
        ref = ca.Refpoints(x=x, y=p)
        xl, _ = ref.ref_values(y_rel=self.y_lim)   # Beam width limits
        self.dx = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = ref.sidelobe()
        self.db_sidelobe = ca.db(self.y_sidelobe, p_ref=p.max())

        for x in xl:
            ax['beamprofile'].axvline(x=x, **self.indicator_line)

        self.scale_axes()
        self._resulttext()

        return

    def scale_axes(self):
        """Change scales of all graphs."""
        ax = self.axes

        element_max = self.d_max * 1e3 * np.array([-1, 1])
        ax["element"].set(xlim=element_max, ylim=element_max)

        ax["axial"].set(ylim=self.x_max*np.array([-1, 1]),
                        xlim=[0, self.z_max])

        lateral_max = self.x_max * np.array([-1, 1])
        ax["lateral"].set(xlim=lateral_max, ylim=lateral_max)

        ax['beamprofile'].set(xlim=self.x_max*np.array([-1, 1]))

        ca.db_axis(ax['beamprofile'],
                   db_min=self.db_min, db_max=0, db_sep=6)

        return 0

    def interact(self,
                 circular=None,
                 azimuth=None,
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
        if azimuth is not None:
            self.azimuth = azimuth
        if frequency is not None:
            self.frequency = 1e3*frequency
        if width is not None:
            self.width = 1e-3*width
        if height is not None:
            self.height = 1e-3*height
        if z_ref is not None:
            self.z_ref = float(z_ref)
        if db_min is not None:
            self.db_min = db_min

        # Display result in graphs
        self.display()

        return

    #=== Non-public methods ==========================================

    # Graphs and results
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
        header = (f'Frequency  $f$ = {self.frequency/1e3:.0f} kHz\n'
                  fr'Wavelength  $\lambda$ = {self.wavelength()*1e3:.1f} mm')

        if self.circular:    # Height dimension omitted
            size_text = (f'Diameter $D$ = {self.width*1e3:.0f} mm = '
                         fr'{self.w_lambda():.1f} $\lambda$')
        else:
            size_text = (f'Width $w$ = {self.width*1e3:.0f} mm = '
                         fr'{self.w_lambda():.1f} $\lambda$'
                         '\n'
                         f'Heigth $h$ = {self.height*1e3:.0f} mm = '
                         fr'{self.h_lambda():.1f} $\lambda$')

        angle_text = (f'Opening angle ({self.lim_text})'
                      r' $\theta_0$ = '
                      fr'{np.degrees(self.theta_0()):.1f}$^\circ$')

        distance_text = (r'Rayleigh distance $z_R$ = '
                         f'{self.z_r():.2f} m')
        beamwidth_text = (f'Beam width ({self.lim_text}): '
                          f'{self.dx:.2f} m')

        if np.isnan(self.x_sidelobe):
            sidelobe_text = ''
        else:
            sidelobe_text = ('Highest sidelobe '
                             fr'$x$ = {self.x_sidelobe:.2f} m, '
                             fr'{self.db_sidelobe:.1f} dB')

        result_text = header + '\n' + size_text + '\n' +  \
            '\n' + distance_text + '\n' + angle_text + '\n' + \
            beamwidth_text + '\n' + sidelobe_text

        ca.remove_fig_text(self.fig)
        ca.set_fig_text(self.fig, result_text, xpos=0.02, ypos=0.15)

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='Single Element Beamprofile')
        ca.add_logo(fig)

        gs = GridSpec(2, 6, figure=fig)
        ax = {'element': fig.add_subplot(gs[0, 0:2]),
              'axial': fig.add_subplot(gs[0, 2:7]),
              'lateral': fig.add_subplot(gs[1, 2:4]),
              'beamprofile': fig.add_subplot(gs[1, 4:7])}

        # Axial intensity plot
        ax['axial'].set(aspect='equal',
                        xlabel='Depth (z) [m]',
                        ylabel='Azimuth (x) [m]',
                        facecolor=self.intensity_background)

        # Element drawing
        ax["element"].set(title='Transducer',
                          facecolor=self.element_background,
                          box_aspect=1,
                          xlabel='Azimuth [mm]',
                          ylabel='Elevation [mm]')

        # Lateral intensity plot
        ax['lateral'].set(box_aspect=1,
                          xlabel='Azimuth [m]',
                          ylabel='Elevation [m]')

        # Beam profile  graphs
        ax['beamprofile'].set(xlabel='Distance [m]',
                              ylabel='Power [dB re. max]')

        ax['beamprofile'].grid(visible=True, which='major', axis='x')

        return ax, fig

    def _remove_old_artists(self):
        for ax in self.axes.values():
            ca.remove_artists(ax)

        try:
            self.cbar.remove()
        except Exception:
            pass

        return 0

    def _draw_orientationline(self, ax):
        if self.azimuth:
            ax.axhline(y=0, **self.orientation_line)
        else:
            ax.axvline(x=0, **self.orientation_line)

        return

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
                                        value=True,
                                        description='Shape',
                                        **layout_settings)

        orientation_widget = widgets.Dropdown(
            options=[('Azimuth (width)', True),
                     ('Elevation (height)', False)],
            value=True,
            description='Orientation',
            **layout_settings)

        db_widget = widgets.BoundedFloatText(
            value=-60, min=-120, max=-6, step=6,
            description='Dynamic range [dB]',
            **layout_settings)

        # Parameter widgets
        label = ['Frequency', 'Width (Diameter)', 'Height', 'Distance']
        n_dim = 2

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

        size_widget = [widgets.FloatSlider(min=10, max=400,
                                           value=100, step=10,
                                           readout_format='.0f',
                                           description='[mm]',
                                           **layout_settings)
                       for k in range(n_dim)]

        distance_widget = widgets.FloatSlider(min=1.0, max=self.z_max,
                                              value=20.0, step=1.0,
                                              readout_format='.1f',
                                              description='[m]',
                                              **layout_settings)

        parameter_widget = [frequency_widget] + size_widget + [distance_widget]

        parameter_line = [widgets.HBox([label_widget[k], parameter_widget[k]])
                          for k in range(len(label))]

        col = [widgets.VBox([shape_widget, orientation_widget, db_widget],
                            layout=widgets.Layout(width='30%')),
               widgets.VBox(parameter_line,
                            layout=widgets.Layout(width='70%'))]

        grid = widgets.HBox(col, layout=widgets.Layout(width='90%'))
        widget_layout = widgets.VBox([title_widget, grid],
                                     layout=widgets.Layout(width='90%'))

        widget = {'circular': shape_widget,
                  'azimuth': orientation_widget,
                  'db_min': db_widget,
                  'frequency': frequency_widget,
                  'width': size_widget[0],
                  'height': size_widget[1],
                  'distance': distance_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
