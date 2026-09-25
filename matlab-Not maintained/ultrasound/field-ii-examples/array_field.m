% Script: array_field
%
% Simple script to calculate radiation field from a rectangular 
% array transducer.
% Using the 'Field II' toolbox from J.A.Jensen et al, DTU
%
% Start Up Instructions 
%   1) Ensure that the Field II librry is available to Matlab by adding 
%      the folder to the Matlab path. On newer versions of MAtlab, this is 
%      most easily done by right-clicking the folder, and selecting
%      'Add to path - Selected Folders and Subfolders'.
%
%   2) Start Field by typing 'field_init.m'

% Lars Hoff, HiVe, Nov 2013
% Modified LH, April 2019

%=== Constants ==========================================================
c   = 1540;                  % m/s Speed of sound in fluid

%=== Transducer Definition ==============================================
%    'TR' is a struct containing transducer specification
clear TR

TR.fs = 100e6;        % Hz   Sample rate
TR.f0 = 4e6;          % Hz   Transducer center frequency
TR.B  = 0.80;         %      Transducer rel. bandwidth 
TR.Nc = 30;            % cycles Excitation pulse length

TR.N  = 64;           %      No. of elements in array
TR.w  = 220e-6;       % m    Element width (x-direction, azimuth)
TR.k  = 20e-6;        % m    Element kerf: Distance between elements
TR.h  = 10.0e-3;      % m    Element height (y-direction, elevation)

TR.F = [0 0 70e+3];   % m    Geometrical focus of aperture
TR.dir = 30;          % deg  Steering angle

[Th,TR] = define_array_transducer(TR);  % Call function to set up define transducer  
set_sampling(TR.fs);         % Define sampling rate in Field II

%=== Calculate and plot tranducer field ==================================
xrange = [-200 200]*1e-3;  % [m]  x-range to calculate and plot
zrange = [0 1000]*1e-3;   % [m]  z-range to calculate and plot
E =calculate_one_way_array(TR, Th, xrange, zrange, 150, 150);

plot_one_way_power(E,xrange,zrange);
%title(sprintf('Frequency %.1f MHz. \n%d element linear array. Total width %.1f mm',...
%    TR.f0/1e6, TR.N, TR.W*1e3));

%=== Clean up and quit ===================================================
xdc_free(Th)

return