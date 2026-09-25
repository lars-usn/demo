"""Run this to import libraries and define internal functions."""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets

# Internal libraries
import beamplot_utilities as bpu

# Colours and lines
COLOR = {
    'element': 'crimson',
    'element_background': 'whitesmoke',
    'text_face': 'whitesmoke',
    'contour': 'white',
    'orientation_line': 'cornflowerblue',
    'intensity_background': 'black',
}

LINE = {
    'contour': {'colors': COLOR['contour'],
                'linestyles': 'dotted',
                'alpha': 0.7},
    'angle': {'color': COLOR['contour'],
              'linestyle': 'dotted',
              'alpha': 0.7},
    'orientation': {'color': COLOR['orientation_line'],
                    'linestyle': 'dotted',
                    'alpha': 1.0},
    'indicator': {'color': 'C1',
                  'linestyle': 'solid'},
    'main': {'color': 'C0',
             'linestyle': 'solid'},
}


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
        self.x_max = 15.0      # m    Max. lateral dimension to calculate
        self.z_max = 100.0     # m    Max. depth to calculate
        self.db_range = 60     # dB   Dynamic gange on dB-scales
        self.db_gain = 6       # dB   Max. on dB-scales

        self.colormap = 'inferno'

        # Initialisation
        self.fig, self.axes, self.graphs = self._initialise_graphs()
        self.scale_axes()

        if create_widgets:
            self.widget = self._create_widgets()

    # === Calculated parameters ===========================
    def wavelength(self):
        """Calculate acoustic wavelenghth."""
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

    def z_rayleigh(self):
        """Rayleigh distance, far-field limit."""
        if self.circular:
            d = self.width
        else:
            d = np.max([self.width, self.height])

        return d**2/(2*self.wavelength())

    def z_reference(self):
        """Limit reference distance to outside far-field limit."""
        return np.max([self.z_ref, 1.1*self.z_rayleigh()])

    # === Axial plane ===================
    def x(self):
        """Lateral dimension for axial plot (x or y)."""
        return np.linspace(-self.x_max, self.x_max, 201)

    def z(self):
        """Axial dimension (depth) for axial plot (z)."""
        return np.linspace(self.z_rayleigh(), self.z_max, 400)

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
            p = 1/self.r() * bpu.jinc(self.w_lambda() * np.sin(self.theta()))
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
        return np.arctan2(self.xy()[0], self.z_reference())

    def phi_xy(self):
        """Elevation angles for xy-positions at distance z."""
        return np.arctan2(self.xy()[1], self.z_reference())

    def theta_r(self):
        """Angle with z-axis for (xyz)-positions."""
        r = np.sqrt(self.xy()[0]**2+self.xy()[1]**2)   # Radial distance
        return np.arctan2(r, self.z_reference())

    def r_xy(self):
        """Radial distances at lateral plane."""
        r = np.sqrt(self.xy()[0]**2+self.xy()[1]**2)   # Radial distance
        return np.sqrt(r**2 + self.z_reference()**2)

    def p_lateral(self):
        """Calculate lateral intensity at reference distance."""
        if self.circular:
            p = 1/self.r_xy() \
                * bpu.jinc(self.w_lambda() * np.sin(self.theta_r()))
        else:
            p = 1/self.r_xy()  \
                * np.sinc(self.w_lambda() * np.sin(self.theta_xy())) \
                * np.sinc(self.h_lambda() * np.sin(self.phi_xy()))

        return p

    # === Commands =============================
    def update_values(self):
        """Update graph values, no scale or other changes."""

        # Element image
        self._update_element()

        # Intensity plots
        p_max = np.max([np.max(abs(self.p_azimuth())),
                        np.max(abs(self.p_elevation()))])

        # Axial
        p_axial = self.p_azimuth() if self.azimuth else self.p_elevation()

        p_db = bpu.db(p_axial, p_ref=p_max)
        self.graphs['axial'].set_array(p_db.ravel())

        self.graphs['refline'].set_xdata([self.z_reference(),
                                          self.z_reference()])

        # Lateral
        p_db = bpu.db(self.p_lateral(), p_ref=p_max)
        self.graphs['lateral'].set_array(p_db.ravel())
        self.axes['lateral'].set_title(
            f'Lateral plane at {self.z_reference():.1f} m')

        # Lateral beam profile
        x = self.x()
        z = self.z()
        k_ref = np.argmin(abs(z-self.z_reference()))
        p = p_axial[:, k_ref]
        p_db = bpu.db(p, p_ref=p_max)
        self.graphs['beamprofile'].set_data(x, p_db)

        # Find reference values
        ref = bpu.Refpoints(x=x, y=p)
        xl, _ = ref.ref_values(y_rel=self.y_lim)   # Beam width limits
        self.dx = xl[1] - xl[0]

        self.x_sidelobe, self.y_sidelobe = ref.sidelobe()
        self.db_sidelobe = bpu.db(self.y_sidelobe, p_ref=p.max())

        # Update scales and messages
        self.update_intensity()
        self._resulttext()

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

        bpu.db_axis(ax['beamprofile'], db_scale=self._db_scale(), db_sep=6)

    def update_intensity(self):
        """Update intensity graph levels."""
        for g in (self.graphs['axial'], self.graphs['lateral']):
            g.set_clim(self._db_scale())

    def interact(self,
                 circular=None,
                 azimuth=None,
                 frequency=None,
                 width=None,
                 height=None,
                 z_ref=None,
                 db_range=None,
                 db_gain=None,
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
            width: float, optional
                Element width (azimuth, x) in mm
            height: float, optional
                Element height (elevation, y) in mm
            z_ref: float, optional
                Reference depth in m
            db_range: float
                Rangeon dB-axes
            db_gain: float
                Maximum on dB-axes
            """
        updates = {
            "circular": circular,
            "azimuth": azimuth,
            "frequency": None if frequency is None else 1e3 * frequency,
            "width": None if width is None else 1e-3 * width,
            "height": None if height is None else 1e-3 * height,
            "z_ref": None if z_ref is None else float(z_ref),
            "db_range": db_range,
            "db_gain": db_gain,
        }

        for name, value in updates.items():
            if value is not None:
                setattr(self, name, value)

        self.update_values()

    # === Non-public methods ==========================================
    # Graphs and results
    def _db_scale(self):
        db_lim = np.array([-self.db_range, 0]) - self.db_gain
        return db_lim

    def _create_element(self, ax):
        """Create aperture patch and return handle."""

        ax.set(title='Transducer shape',
               facecolor=COLOR['element_background'],
               box_aspect=1,
               xlabel='Azimuth [mm]',
               ylabel='Elevation [mm]')

        if self.circular:
            patch = patches.Circle(
                (0, 0),
                radius=0,
                fill=True,
                color=COLOR["element"],
            )
        else:
            patch = patches.Rectangle(
                (0, 0),
                width=0,
                height=0,
                fill=True,
                color=COLOR["element"],
            )

        ax.add_patch(patch)
        return patch

    def _update_element(self):
        """Update aperture patch with current dimensions."""
        w_mm = self.width * 1e3
        h_mm = self.height * 1e3

        patch_type = patches.Circle if self.circular else patches.Rectangle

        # Change shape if necessary
        if not isinstance(self.graphs['element'], patch_type):
            self.graphs['element'].remove()

            if self.circular:
                self.graphs['element'] = patches.Circle(
                    (0, 0), w_mm / 2,
                    fill=True,
                    color=COLOR["element"]
                )
            else:
                self.graphs['element'] = patches.Rectangle(
                    (-w_mm / 2, -h_mm / 2), w_mm, h_mm,
                    fill=True,
                    color=COLOR["element"]
                )

            self.axes['element'].add_patch(self.graphs['element'])
            return

        # Update geometry
        if self.circular:
            self.graphs['element'].set_radius(w_mm / 2)
        else:
            self.graphs['element'].set_xy((-w_mm / 2, -h_mm / 2))
            self.graphs['element'].set_width(w_mm)
            self.graphs['element'].set_height(h_mm)

        # Turn on and off orientation lines
        for g in self.graphs['azimuth']:
            g.set_visible(self.azimuth)
        for g in self.graphs['elevation']:
            g.set_visible(not self.azimuth)

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
                         f'{self.z_rayleigh():.2f} m')
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

        bpu.set_fig_text(self.fig, result_text, xpos=0.02, ypos=0.15)

        return

    def _initialise_graphs(self):
        """Initialise result graphs."""
        plt.close('all')
        fig = plt.figure(figsize=[10, 5],
                         constrained_layout=True,
                         num='Single Element Beamprofile')
        bpu.add_logo(fig)

        gs = GridSpec(2, 6, figure=fig)
        axes = {'element': fig.add_subplot(gs[0, 0:2]),
                'axial': fig.add_subplot(gs[0, 2:7]),
                'lateral': fig.add_subplot(gs[1, 2:4]),
                'beamprofile': fig.add_subplot(gs[1, 4:7])}

        graphs = {}

        # Transducer element illustration
        graphs['element'] = self._create_element(axes['element'])

        # Axial intensity plot
        axes['axial'].set(aspect='equal',
                          xlabel='Depth (z) [m]',
                          ylabel='Lateral position (Azimuth or Elevation) [m]',
                          facecolor=COLOR['intensity_background'])

        x_coords = self.z()
        y_coords = self.x()
        dummy_data = np.full((len(y_coords), len(x_coords)), np.nan)
        graphs['axial'] = axes['axial'].pcolormesh(x_coords,
                                                   y_coords,
                                                   dummy_data,
                                                   clim=self._db_scale(),
                                                   cmap=self.colormap)

        graphs['refline'] = axes['axial'].axvline(x=self.z_reference(),
                                                  **LINE['orientation'])

        graphs['colorbar'] = fig.colorbar(graphs['axial'], ax=axes['axial'])
        graphs['colorbar'].set_ticks(np.arange(-96, 30, 6))

        # Lateral intensity plot
        axes['lateral'].set(box_aspect=1,
                            xlabel='Azimuth [m]',
                            ylabel='Elevation [m]')

        x_coords = self.xy()[0][0, :]
        y_coords = self.xy()[1][:, 0]
        dummy_data = np.full((len(y_coords), len(x_coords)), np.nan)

        graphs['lateral'] = axes['lateral'].pcolormesh(
            x_coords,
            y_coords,
            dummy_data,
            clim=self._db_scale(),
            cmap=self.colormap,
            shading='auto')

        # Beam profile  graphs
        axes['beamprofile'].set(xlabel='Distance [m]',
                                ylabel='Power [dB re. max]')
        axes['beamprofile'].grid(visible=True,
                                 which='major',
                                 axis='x')
        graphs['beamprofile'], = axes['beamprofile'].plot([], [],
                                                          **LINE['main'])
        # Orientation indicator lines
        graphs['azimuth'] = []
        graphs['elevation'] = []
        for ax in (axes['element'], axes['lateral']):
            graphs['azimuth'].append(ax.axhline(y=0, **LINE['orientation']))
            graphs['elevation'].append(ax.axvline(x=0, **LINE['orientation']))

        return fig, axes, graphs

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""
        title = 'Beam-profile from Single Element Transducer'
        title_widget = widgets.Label(title, style=dict(font_weight='bold'))

        left_layout = {'continuous_update': True,
                       'layout': widgets.Layout(width='95%'),
                       'style': {'description_width': '50%'}}

        right_layout = {'continuous_update': True,
                        'layout': widgets.Layout(width='95%'),
                        'style': {'description_width': '30%'}}

        left_width = '25%'
        right_width = '75%'

        # Left column widgets (Dropboxes, number boxes)
        shape_widget = widgets.Dropdown(options=[('Rectangular', False),
                                                 ('Circular', True)],
                                        value=True,
                                        description='Shape',
                                        **left_layout)

        orientation_widget = widgets.Dropdown(
            options=[('Azimuth (width)', True),
                     ('Elevation (height)', False)],
            value=True,
            description='Orientation',
            **left_layout)

        db_range_widget = widgets.BoundedFloatText(
            value=60, min=6, max=120, step=6,
            description='Range [dB]',
            **left_layout)

        db_gain_widget = widgets.BoundedFloatText(
            value=12, min=-120, max=120, step=6,
            description='Gain [dB]',
            **left_layout)

        left_col = widgets.VBox([shape_widget,
                                 orientation_widget,
                                 db_range_widget,
                                 db_gain_widget],
                                layout=widgets.Layout(width=left_width))

        # Right column widgets (Sliders)
        frequency_widget = widgets.FloatSlider(min=1, max=400,
                                               value=100, step=1,
                                               readout_format='.0f',
                                               description='Frequency [kHz]',
                                               **right_layout)

        width_widget = widgets.FloatSlider(min=10, max=400,
                                           value=100, step=10,
                                           readout_format='.0f',
                                           description='Width (Diameter) [mm]',
                                           **right_layout)

        height_widget = widgets.FloatSlider(min=10, max=400,
                                            value=150, step=10,
                                            readout_format='.0f',
                                            description='Height [mm]',
                                            **right_layout)

        distance_widget = widgets.FloatSlider(min=1.0, max=self.z_max,
                                              value=20.0, step=1.0,
                                              readout_format='.0f',
                                              description='Distance [m]',
                                              **right_layout)

        right_col = widgets.VBox([frequency_widget,
                                  width_widget,
                                  height_widget,
                                  distance_widget],
                                 layout=widgets.Layout(width=right_width))

        widget_layout = widgets.HBox([left_col, right_col],
                                     layout=widgets.Layout(width='80%'))

        widget_layout = widgets.VBox([title_widget, widget_layout])

        widget = {'circular': shape_widget,
                  'azimuth': orientation_widget,
                  'db_range': db_range_widget,
                  'db_gain': db_gain_widget,
                  'frequency': frequency_widget,
                  'width': width_widget,
                  'height': height_widget,
                  'distance': distance_widget,
                  }

        w = WidgetLayout(widget_layout, widget)

        return w
