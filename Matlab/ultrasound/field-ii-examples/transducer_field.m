% Script: transducer_field
%
% Simple script to calculate radiation field from a rectangular transducer
% Using the 'Field II' toolbox from J.A.Jensen et al, DTU
%
% Start Up Instructions 
%   1) Ensure that the Field II librry is available to Matlab by adding 
%      the folder to the Matlab path. On newer versions of Matlab, this is 
%      most easily done by right-clicking the folder, and selecting
%      'Add to path - Selected Folders and Subfolders'.
%
%   2) Start Field by typing 'field_init.m'
%

% Lars Hoff, HiVe, Nov 2013
% Modified LH, April 2019

%=== Constants ==========================================================
c   = 1540;                  % m/s Speed of sound in fluid

%=== Transducer Definition ==============================================
%    'TR' is a struct containing transducer specification
clear TR
TR.fs = 100e6;            % Hz     Sample rate
TR.f0 = 4e6;              % Hz     Transducer center frequency
TR.W  = 15e-3;             % m      Transducer width
TR.H  = 10e-3;            % m      Transducer height
TR.F  = 1000e-3;          % m      Transducer focal length. Large value: Unfocused
TR.B  = 0.8;              %        Transducer relative bandwidth
TR.Nc = 30;               % cycles Excitation pulse length

Th = define_transducer(TR);  % Call function to set up define transducer  
set_sampling(TR.fs);         % Define sampling rate in Field II

%=== Calculate and plot tranducer field ==================================
xrange = [-40 40]*1e-3;  % [m]  x-range to calculate and plot
zrange = [0 200]*1e-3;   % [m]  z-range to calculate and plot
E =calculate_transducer_field(TR, Th, xrange, zrange, 300, 300);

plot_one_way_power(E,xrange,zrange);
title( sprintf('Frequency %.1f MHz. Width %.0f mm', TR.f0/1e6, TR.W*1e3) )

%=== Clean up and quit ===================================================
xdc_free(Th)

return