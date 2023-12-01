function Th=define_transducer(TR)
% function Th=define_transducer(TR)
%
% Define transducer for use in Field II
%
%   TR   Struct containing transducer specification
%   Th   Pointer to transducer aperture, from Field II
%

% Lars Hoff, HiVe, Nov 2013
% Corrected Nov 2023

mathelement= 0.2e-3;   % m  Transducer mathematical element size

%--- Geometry ---
%    Rectangular aperture following Field II specification
w= TR.W;
h= TR.H;
rect=[1 [-w/2 -h/2 0] [-w/2 h/2 0] [w/2 h/2 0] [w/2 -h/2 0], 1, w, h, [0 0 0] ];
center=[0 0 0];
focus= [0 0 TR.F];

Th = xdc_rectangles(rect, center, focus);
xdc_show(Th, 'all')

%--- Pulse ---
TR.T  = TR.Nc/TR.f0;         % s      Excitation pulse length
excitation=sin(2*pi*TR.f0*(0:1/TR.fs:TR.T));
xdc_excitation (Th, excitation);

%--- Transmit aperture ---
%    Impulse response approximated as Hanning-windowed sine-pulse 
ti = (0:1/TR.fs:2/(TR.B*TR.f0));    % s  Time points where impulse response is specified 
impulse_response=sin(2*pi*TR.f0*ti);
impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
xdc_impulse(Th, impulse_response);

% plot(ti,impulse_response) % Uncomment for viewing impulse response
% pause

%--- Receive aperture ---
xdc_impulse(Th, impulse_response);

return