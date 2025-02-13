function TR=calculate_and_plot(TR, Th, xzsize)
% function TR=calculate_and_plot(TR, Th, xzsize)
% 
% Calculate transducer transmit field, one-way
% Use Field II
%
%   TR      Transducer definition
%   Th      Handle to transducer, given by Field II
%   xzsize  Max x- and z-dimension to plot

%--- Initialize ---
xmax= xzsize(1); 
zmax= xzsize(2);

Nx = 200;  % No of points to plot in x- and z dimensions
Nz = 300;  % Increase for better resolution, decreass for speed

%--- Define xz-points to include in plot ---
x= linspace(-xmax, xmax, Nx)';
y= 0;
z= linspace(0, zmax, Nz)';

%--- Loop over all points. Calcluate impulse response ---
fprintf('\nCalculating beam profile ...')
for kz=1:Nz
    for kx=1:Nx
        pt(kx,:)= [ x(kx) y z(kz)];    % Coordinates of point.
                                      % All x-values for one z-value 
    end
    [hp,t0]= calc_hp(Th,pt);   % Time response, all points at distance z(kz)
    Php(kz,:)= sum(hp.^2,1);   % Total energy, all points at distance z(kz)
end
clear hp                       % hp can be big, remove to save memory

%-- Convert to dB, normalized to max value ---
TR.PhpdB = 10*log10(Php/max(max(Php(Nz/3:end,:)))); 

fprintf('Finished\n')
%--- Plot field energy in intensity plot ---
imagesc( xmax*[-1 1], [0 zmax], TR.PhpdB, [-40 0] )  
colormap(hot)
colorbar
xlabel('Lateral distance [m]')
ylabel('Axial distance [m]')

axis('equal')
axis([xmax*[-1 1] [0 zmax] ])

return    