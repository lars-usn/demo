# Interactive illustration of the convolution operation, applied to FIR filtering
# Use Matplotlib backend that allows interactive operation

# from math import pi
import numpy as np
from scipy.signal import convolve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets

class WidgetLayout():
    """Container for widgets and layout."""

    def __init__(self, layout, widget):
        self.layout = layout
        self.widget = widget


class ConvolutionPlot():
    """Plot all resouts of convolution."""

    def __init__(self, initialise_graphs=True, create_widgets=False):

        self.n = 0
        self.length = [10, 3]
        self.amplitude = [1.0, 0.5]
        self.decay = [1.0, 0.8]

        self.color = self._initialise_colors()

        if initialise_graphs:
            self.ax = self._initialise_graphs()
            # self.scale_graphs()

        if create_widgets:
            self.widget = self._create_widgets()

        return

    def signal(self, amplitude=1, length=4, decay=1.0):
        """Create signal or response."""
        return amplitude * decay**np.arange(length)

    def s(self):
        return self.signal(amplitude=self.amplitude[0],
                          length=self.length[0],
                          decay=self.decay[0])

    def h(self):
        return self.signal(amplitude=self.amplitude[1],
                           length=self.length[1],
                           decay=self.decay[1])

    def _initialise_colors(self):
        color = {}
        color['signal'] = 'C0'
        color['filter']= 'C3'
        color['indicator'] = 'C2'
        color['result'] = 'C1'
        color['baseline'] = ' ' # mcolors.TABLEAU_COLORS['tab:gray']

        return color

    def _initialise_graphs(self):
        """Initialise graphs for signals and spectra.

        Returns
        ------
        ax : List of axis objects
            Axes where results are plotted
        """

        plt.close('all')
        plt.rc('font', size=8)          # Default text sizes
        fig = plt.figure(figsize=[12, 6],
                         constrained_layout=True,
                         num='Convolution Demo')

        ax = fig.subplot_mosaic([['signal', 'combined',   'combined'  ],
                                 ['filter', 'multiplied', 'multiplied'],
                                 ['.',     'result',     'result']])

        ax['signal'].set_title('Input signal', loc='left')
        ax['combined'].set_title('Signal and impulse response', loc='left')
        ax['filter'].set_title('Filter', loc='left')
        ax['multiplied'].set_title('Multiplied', loc='left')
        ax['result'].set_title('Output', loc='left')

        for k in ax:
            ax[k].set_xlabel('Sample [n]')
            ax[k].grid(True)

        return ax

    def display(self):
        """Plot all signals and spectra."""

        # Calculate result of convolution
        s = self.s()
        h = self.h()
        y = convolve(s, h)
        y_max = 1.2 * max(np.max(y), np.max(s), np.max(h))  # Max. y scale
        y_min = 1.2 * min(np.min(y), np.min(s), np.min(h), 0)  # Max. y scale

        # Clear old graphs
        for k in self.ax:
            for art in list(self.ax[k].lines):
                art.remove()
            for art in list(self.ax[k].collections):
                art.remove()
            for art in list(self.ax[k].patches):
                art.remove()

        # Plot input and filter response
        self.ax['signal'].stem(s, linefmt=self.color['signal'], basefmt=self.color['baseline'])
        self.ax['filter'].stem(h, linefmt=self.color['filter'], basefmt=self.color['baseline'])

        # Plot signal and flipped filter in same graph
        h_pos = self.n + np.arange(0, -len(h), -1)  # Shifted response
        s_pos = np.arange(len(s))
        self.ax['combined'].stem(s_pos, s, linefmt=self.color['signal'], basefmt=self.color['baseline'])
        self.ax['combined'].stem(h_pos, h, linefmt=self.color['filter'], basefmt=self.color['baseline'])

        # Plot multiplication inside filter support
        ni = self.n + 1 + np.arange(-len(h), 0)    # Indices overlapping with h
        ni = ni[(ni >= 0) & (ni < len(s))]
        nh = len(ni)

        h_flip = np.flip(h)
        if len(ni) > 0:
            s_h = s[ni] * h_flip[0:nh]
            self.ax['multiplied'].stem(ni, s_h, linefmt=self.color['indicator'], basefmt=self.color['baseline'])

        # Plot output of convolution
        self.ax['result'].stem(y, linefmt=self.color['result'], basefmt=self.color['baseline'])

        # Show active point with indicator line
        for k in ['combined', 'multiplied', 'result']:
            self.ax[k].axvline(x=self.n, linestyle='-', color=self.color['indicator'] )

        # Mark filter support with patch
        for k in ['combined', 'multiplied']:
            self.ax[k].add_patch(patches.Rectangle((self.n-len(h)+1, y_min),  # (x,y)
                                 len(h)-1,
                                 y_max-y_min,
                                 alpha=0.20,
                                 color=self.color['indicator']))

        # Scale axes
        n_max = len(y) + 2
        for k in self.ax:
            self.ax[k].axhline(y=0, color='gray', linestyle='-')
            self.ax[k].set_ylim(y_min, y_max)

        for k in ['signal', 'filter']:
            self.ax[k].set_xlim(-0.5, len(s))

        for k in ['combined', 'multiplied', 'result']:
            self.ax[k].set_xlim(-2, n_max)

        return 0

    # Simple interactive operation
    def interact(self, indicator=None,
                 signal_length=None, signal_amplitude=None, signal_decay=None,
                 filter_length=None, filter_amplitude=None, filter_decay=None):
        """For interactive operation."""

        if indicator is not None:
            self.n = indicator
        if signal_length is not None:
            self.length[0] = signal_length
        if signal_amplitude is not None:
            self.amplitude[0] = signal_amplitude
        if signal_decay is not None:
            self.decay[0] = signal_decay

        if filter_length is not None:
            self.length[1] = filter_length
        if filter_amplitude is not None:
            self.amplitude[1] = filter_amplitude
        if filter_decay is not None:
            self.decay[1] = filter_decay

        self.display()

        return

    # Interactive widgets
    def _create_widgets(self):
        """Create widgets for interactive operation."""

        # Title
        title = ('Colvolution Demo')
        title_widget = ipywidgets.Label(title, style=dict(font_weight='bold'))

        # Layouts definitions
        text_layout = {'continuous_update': False,
                       'layout': ipywidgets.Layout(width='10%'),
                       'style': {'description_width': '60%'}}

        label_layout = {'continuous_update': False,
                               'layout': ipywidgets.Layout(width='5%')}

        slider_layout = {'continuous_update': True,
                        'layout': ipywidgets.Layout(width='70%'),
                        'style': {'description_width': '5%'}}

        # Individual widgets
        label = ['Signal', 'Filter']
        label_widget = [ipywidgets.Label(labeltext,
                                         **label_layout)
                        for labeltext in label]

        length_widget = [ipywidgets.IntText(min=1, max=20, value=n,
                                          description='Length',
                                          **text_layout)
                          for n in self.length]

        amplitude_widget = [ipywidgets.FloatText(min=0.1, max=2.0, step=0.1, value=k,
                                              description='Amplitude',
                                              **text_layout)
                            for k in self.amplitude]

        decay_widget = [ipywidgets.FloatText(min=0.1, max=1.0, step=0.1, value=k,
                                          description='Decay',
                                          **text_layout)
                        for k in self.decay]

        # Create lines of parameter widgets
        widget_line = [ipywidgets.HBox([label_widget[k],
                                     length_widget[k],
                                     amplitude_widget[k],
                                     decay_widget[k]]) for k in range(2)]

         # Add slider for position indicator
        indicator_widget = ipywidgets.IntSlider(
            min= -1, max= 20.0, value= 0,
            description='Indicator',
            **slider_layout)

        widget_layout = ipywidgets.VBox([grid_line for grid_line in widget_line] )
        widget_layout = ipywidgets.VBox([title_widget,
                                         widget_layout,
                                         indicator_widget])

       # Export as dictionary
        widget = {'indicator': indicator_widget,
                  'length': length_widget,
                  'amplitude': amplitude_widget,
                  'decay': decay_widget,
                  }
        w = WidgetLayout(widget_layout, widget)

        return w
