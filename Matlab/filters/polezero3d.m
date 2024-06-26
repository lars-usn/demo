function polezero3d( b, a, zmax, dispFormat )
% function polezero3d( b, a, zmax, dispFormat )
%
%  3D illustration of pole-zero plot in the z-plane, with frequency response
%
%   a   Forward filter coefficients.   
%   b   Backward filter coefficients.  
% 
%  zmax Maximunm on z-scale
%
% dispformat (optional)  "dB" for dB-scale display, otherwise linear
%
% Example 1 (Chapter 10-13 in McClellan et al., DSP first)
%              [b,a]= ellip( 3, 1, 30, 0.3 );  
%              pole_zero_3d( b, a )
%
% Example 2  (Similar to Ex. 1, illustrating different filter topology)  
%              [b,a]= butter( 3, 0.3 );  
%              pole_zero_3d( b, a )

% Lars Hoff, USN, Feb 2023 (Updated May 2024)

if nargin<4, dispFormat='abs';  end
if nargin<3, zmax = 2;          end


%% Create grid, calculate and plot result
xmax = 1.5; % Span of real and imaginary axes
x = linspace( -xmax, xmax, 100);
[ zr, zim ] = meshgrid( x, x );
z = zr + 1i*zim;

H = Hz( b, a, z, dispFormat );      % Calculate system function H
surf( zr, zim, H, 'EdgeColor', 'none' );

%% Create plot markers
nPoints = 100;
phi     = linspace( 0, 2*pi, nPoints );  % Angle vector
uc      = exp( 1i*phi );                 % z on unit circle

% Axes and unit circle
Huc = Hz( b, a, uc,   dispFormat);  % Unit circle
Hre = Hz( b, a, x,    dispFormat);  % Real axis
Him = Hz( b, a, 1i*x, dispFormat);  % Imaginary axis

hold on
plot3( x,               zeros(size(x)), Hre, 'k-', 'linewidth', 1 ) % Real axis
plot3( zeros(size(x) ), x,              Him, 'k-', 'linewidth', 1 ) % Imaginary axis
plot3( real(uc),        imag(uc),       Huc, 'k:', 'linewidth', 2 ) % Unit circle
hold off

if zmax==0  % Autoscale z-axis
    zmax= max( max( abs(Huc) ) );    % Max magnitude on unit circle
end

axis equal
zlim( [ 0 zmax ] );
clim( [ 0 zmax ] );
colorbar

xlabel( 'Re\{z\}' )
ylabel( 'Im\{z\}' )
zlabel( '|H(z)|' )

end

%% Internal functions

%--- Calculate system function H(z) directly from coefficients a and b
function H = Hz ( b, a, z, dispformat )

% Denominator A(z)
N = length(a) - 1;
A = zeros( size(z) );
for n=0:N
    A = A + a(n+1)*z.^(-n); 
end

% Numerator B(z)
M = length(b) - 1;
B = zeros( size(z) );
for k=0:M
    B = B+ b(k+1)*z.^(-k);
end

H = B./A;

%--- Select linear or dB-scale
switch lower(dispformat)
    case "db",  H = 20*log10(abs(H));
    otherwise,  H = abs(H);
end
end