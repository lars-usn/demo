# Demo Programs for Courses at the University of South-Eastern Norway

## Ultrasound and Signal Processing

This repository contains small applications written to illustrate phenomena taught in courses, either for classroom demonstration or for use by students on their own. 
The programs were written over several years using different tools, to the courses presently called  'TSE2280 Measurement Technology and Signal Processing' and 'SPE590 Specialisation Topic (Acoustics and Ultrasound Technology)'.

All new developments are done in Python and Jupyter Notebook. The Matlab apps are still in use, while the LabVIEW files are no longer maintained.

The Jupyter Notebook *.ipynb*-files can be run interactively in JupyterLab or Jupyter Notebook.

The Matlab-apps (.mlapp)  and LabVIEW VIs (.vi) are stand-alone programs with a graphical user interface.
The Matlab functions (.m) are called from the Matlab console or from antoher Matlab-function.
The Matlab and LabVIEW programs will need access to all files in their folders to function.


#  Ultrasound
These demo program are written for courses in acoustics  ultrasound technology at at USN. They are presently used in courses at MSc and PhD-level, such as the course *SPE5950 Specialisation Topic - Acoustics and Ultrasound Technology*.

| Description  | Function Name | 
| -- | -- |
| **Jupyter Notebook widgets** | | 
| Radiated field from a single-element transducer             |  `radiated-field-single-element-transducer.ipynb` |
| Simple illustration of pulse length and bandwidth           | `pulse-length-and-bandwidth.ipynb` |
|  **Matlab** | |
| Beam profile from a single-element ultrasound transducer    | `element_beamprofile.mlapp` |
| Beam profile from an ultrasound transducer array            | `array_beamprofile.mlapp`  |
| Simple example files to start simulations in Field II       |  Folder `field-ii-examples`   |
  
# Signal Processing
These demo program are written for use in introductory courses in signal processing at USN. They are presently used in the course [*TSE2280 Måleteknikk og signalbehandling*](https://www.usn.no/studier/studie-og-emneplaner/#/emne/TSE2280_1_2024_V%C3%85R) usnig the textbook  McClellan et al., "DSP First", 2nd ed., Pearson Education Limited, 2016.

### Dynamic Systems
| Description  | Function Name | 
| -- | -- |
| **Jupyter Notebook widgets** | | 
| Step response of a first order dynamic system        |  `first_order_step_response.ipynb` |
| Frequency response of a first order dynamic system   |  `first_order_frequency_response.ipynb` |
| Step response of a second order dynamic system       |  `second_order_step_response.ipynb` |
| Frequency response of a second order dynamic system  |  `second_order_frequency_response.ipynb` |
| **Matlab** | | |
| Step response of a first order dynamic system         | `firstorder_step_demo.mlapp` |
| Frequency response of a first order dynamic system    | `firstorder_frequency_demo.mlapp` |
| Step response of a second order dynamic system        | `secondorder_step_demo.mlapp` |
| **LabVIEW** | |
| Animation of step response for a second order dynamic system  | `Second Order System RT Demo.vi` |

### Signals 
Programs written to illustrate phenomena in chapters 2, 3 and 4 in McClellan et al., "DSP First", 2nd ed., Pearson Education Limited, 2016.
| Description  | Function Name | 
| -- | -- |
| **Python and Jupyter Notebook widgets** | | 
| Illustration of sine- and cosine-functions as complex phasors  | `phasor_demo.ipynb` |
| Periodic signals - Sum of cosine-waves with different frequencies  | `periodicity_demo.ipynb` |
| Fourier series - Arbitrary signals constructed from cosine-waves| `fourier_synthesis_demo.ipynb` |
| Aliasing - Sampled signals in the time and frequency domains  | `aliasing_frequency_demo.ipynb` |
| Aliasing - Multiple frequencies fitting the same sample points   | `multiple_alias_demo.ipynb` |
| Summing of complex phasors and signals in the time domain | `zplot.py` |
| **Matlab** | | 
| Illustration of sine- and cosine-functions as complex phasors | `phasor_demo.mlapp` |
| Illustration of signals constructed from sine-waves: Fourier coefficients   | `fouriersynthesis.mlapp` |
| Aliasing: Several frequencies fitting the same sample points   | `aliasing_demo.mlapp` |
| Simple demonstration of aliasing                               | `aliasing_frequencies_demo.mlapp` |
| Sampling and aliasing illustrated in the frequency domain      | `aliasing_frequencydomain_demo.mlapp` |
| Sampling of an image, Moire-pattern                            | `aliasing_images_demo.mlapp` |
| Stripes in image, Moire-pattern                                | `moirepattern_demo.mlapp` |
| **LabVIEW** | | 
| Spectrogram of sound, real-time                                | `Sound Spectrogram Advanced.vi`
| Spectrogram of sound, real-time. Simplified code               | `Sound Spectrogram Simple.vi`
| Periodicity, sine-waves with different frequencies             | `Periodicity.vi`|
| Beat, two signals close in frequency            | `Beat Demo.vi` |
| Beat as slowly moving phasors                   |  `Beat Blink.vi`|
| Beat signals played as sound                    |  `Beat Sound.vi`|
| Amplitude modulation (AM)                       |  `AM Demo.vi` |

### Filters
Programs written to illustrate phenomena in chapters 5, 6, 9, and 10 in McClellan et al., "DSP First", 2nd ed., Pearson Education Limited, 2016.

| Description  | Function Name | 
| -- | -- |
| **Jupyter Notebook widgets** | | 
| FIR-filter as convolution                         | `convolution_demo.ipynb` |
| **Matlab** | |
| Illustration of convolution                       | `convolution_demo.mlapp` |
| Running-average FIR-filter                        | `running_average_demo.mlapp` |
| Illustration of the Dirichlet-kernel              |  `dirichlet_kernel_demo.mlapp` |
| Frequency response of FIR-filter                  | `fir_response_demo.mlapp` |
| Simple bandpass FIR-filter                        | `fir_bandpass_demo.mlapp` |
| Response of IIR-filter, interactive               | `iir_response_demo.mlapp`  |
| Response of IIR-filter, static function           | `iir_response.m`  |
| Comparison of common lowpass-filter topologies    | `lowpass_response_demo.mlapp` |
| 3D illustration of pole-zero plot in the z-plane  |  `polezero3d.m` |
| Scale x- and y-axes in units of &pi;              | `piaxis.m` |
| **LabVIEW** | |
| FIR and IIR filter for noiose removal           | `FIR and IIR Filter.vi`|

