# Demo Programs for Courses at the University of South-Eastern Norway

This repository contains a  of small applications written to illustrate phenomena taught in courses, either for classroom demonstration or for use by students on their own. 
The programs were written over several years using different tools, to the courses presently called  'TSE2280 Measurement Technology and Signal Processing' and 'SPE590 Specialisation Topic (Acoustics and Ultrasound Technology)'.

Note that the LabVIEW and Matlab programs will need access to all files in their folders to function.
The Matlab functions are called from the Matlab console or from antoher Matlab-function.
The LabVIEW VIs and Matlab-apps are stand-alone programs with a graphical user interface.

#  Ultrasound
## SPE590 Specialisation Topic (Acoustics and Ultrasound Technology)
| Description  | Type | Function Name | 
| -- | -- | -- | 
| Beam profile from a single-element ultrasound transducer    | Matlab-app | `element_beamprofile.mlapp` |
| Beam profile from ultrasound transducer array               | Matlab-app | `array_beamprofile.mlapp`  |
| Radiated field from a single-element transducer             | JupyterLab widget |  `radiated-field-single-element-transducer.ipynb` |
| Simple illustration of pulse length and bandwidth           | JupyterLab widget | `pulse-length-and-bandwidth.ipynb` |
| Simple example files to start simulations in Field II       | Matlab functions  | Folder `field-ii-examples`   |
  
# Signal Processing
## TSE2280 Measurement Technology and Signal Processing

### Dynamic Systems
| Description  | Type | Function Name | 
| -- | -- | -- | 
| Step response of a first-order dynamic system   | Matlab-app | `firstorder_step_demo.mlapp` |
| Frequency response of a first-order dynamic system   | Matlab-app | `firstorder_frequency_demo.mlapp` |
| Step response of a second-order dynamic system  | Matlab-app | `secondorder_step_demo.mlapp` |
| Step response of a second-order dynamic system  | LabVIEW VI | `Second Order System RT Demo.vi` |

### Signals 
These programs were written to illustrate phenomena in chapters 2 and 3 in McClellan et al., "DSP First", 2nd ed., Pearson Education Limited, 2016.
| Description  | Type | Function Name | 
| -- | -- | -- | 
| Illustration of sine- and cosine-functions as complex phasors | Matlab-app| `phasor_demo.mlapp` |
| Periodicity, sine-waves with different frequencies                    | LabVIEW VI       | `Periodicity.vi`|
| Illustration of signals constructed from sine-waves: Fourier coefficients   | Matlab-app | `fouriersynthesis.mlapp` |
| Spectrogram of sound, real-time                                             | LabVIEW VI | `Sound Spectrogram Advanced.vi`
| Spectrogram of sound, real-time. Simplified code                            | LabVIEW VI | `Sound Spectrogram Simple.vi`
| Beat, two signals close in frequency            | LabVIEW VI       | `Beat Demo.vi` |
| Beat as slowly moving phasors                   | LabVIEW VI       | `Beat Blink.vi`|
| Beat signals played as sound                    | LabVIEW VI       | `Beat Sound.vi`|
| Amplitude modulation (AM)                       | LabVIEW VI       | `AM Demo.vi` |
| Aliasing: Several frequencies fitting the same sample points    | Matlab-app | `aliasing_demo.mlapp` |
| Simple demonstration of aliasing                                | Matlab-app | `aliasing_frequencies_demo.mlapp` |
| Sampling and aliasing illustrated in the frequency domain       | Matlab-app | `aliasing_frequencydomain_demo.mlapp` |
| Sampling of an image, Moire-pattern                             | Matlab-app | `aliasing_images_demo.mlapp` |
| Stripes in image, Moire-pattern                                 | Matlab-app | `moirepattern_demo.mlapp` |

### Filters
These programs were written to illustrate phenomena in chapters 5, 6, 9, and 10 in McClellan et al., "DSP First", 2nd ed., Pearson Education Limited, 2016.
| Description  | Type | Function Name | 
| -- | -- | -- | 
| Illustration of convolution                    | Matlab-app      | `convolution_demo.mlapp` |
| Running-average FIR-filter                     | Matlab-function | `running_average_demo.mlapp` |
| Illustration of the Dirichlet-kernel           | Matlab-app      |  `dirichlet_kernel_demo.mlapp` |
| Frequency response of FIR-filter               | Matlab-app | `fir_response_demo.mlapp` |
| Simple bandpass FIR-filter                     | Matlab-app | `fir_bandpass_demo.mlapp` |
| Response of IIR-filter                         | Matlab-app | `iir_response_demo.mlapp`  |
| Response of IIR-filter                         | Matlab-function | `iir_response.m`  |
| Comparison of common lowpass-filter topologies | Matlab-app      | `lowpass_response_demo.mlapp` |
| 3D illustration of pole-zero plot in the z-plane| Matlab-function |  `polezero3d.m` |
| FIR and IIR filter for noiose removal           | LabVIEW VI       | `FIR and IIR Filter.vi`|
| Scale x- and y-axes in units of &pi; | Matlab-function | `piaxis.m` |

