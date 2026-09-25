function [Th,TR]=define_array_transducer(TR)
% function Th=define_array_transducer(TR)
%
% Define rectangular array transducer for use in Field II
%
%   TR   Struct containing transducer specification
%   Th   Pointer to transducer aperture, from Field II
%

% Lars Hoff, HiVe, Nov 2013

%--- Geometry ---
TR.nx = 4;   %      Sub-division azimuth (mathematical) in x-direction
TR.ny = 10;  %      Sub-division elevation (mathematical) in y-direction

TR.d  = TR.w+TR.k;             % m    Pitch: Distance between elements
TR.W = TR.N*(TR.w+TR.k)-TR.k;  % m    Width of full array

Th = xdc_linear_array (TR.N, TR.w, TR.h, TR.k, TR.nx, TR.ny, TR.F); % Defina array

%--- Impulse response ---
ti= (0:1/TR.fs:2/(TR.B*TR.f0));
ht= sin(2*pi*TR.f0*ti);
ht= ht.*hanning(max(size(ht)))';
xdc_impulse(Th, ht);

%--- Electrical excitation ---
TR.T  = TR.Nc/TR.f0; % s   Excitation pulse length
excitation=sin(2*pi*TR.f0*(0:1/TR.fs:TR.T));  % Define electrical excitation of array
xdc_excitation (Th, excitation);

%=== Steering angle and focus ============================================
%    Sets up the electrical focusing distance and angle
theta_s = TR.dir*pi/180;    % rad  Steering angle
xdc_focus(Th, 0, TR.F(3)*[sin(theta_s) 0 cos(theta_s)]);


return