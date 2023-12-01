function plot_one_way_power(E,xrange,zrange)
%function plot_one_way_power(E,xrange,zrange)
% 
% Intensity plot of one-way transducer field 
%
%   E     Acoustic power in each point 
%

[Nz Nx]= size(E);

%-- Convert to dB, normalized to max value ---
n1= ceil(Nx/10);
Emax= max(max(E(n1:end,:)));   % Maximum, neglecting nearest points
EdB = 10*log10(E/Emax);   % Convert to dB, normalized to max 

%--- Plot field energy in intensity plot ---
imagesc( xrange, zrange, EdB, [-40 0] ) 
colormap(hot)
colorbar
xlabel('Lateral distance [m]')
ylabel('Axial distance [m]')

axis('equal')
axis([min(xrange) max(xrange) min(zrange) max(zrange)])

return    