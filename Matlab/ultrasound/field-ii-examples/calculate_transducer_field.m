function E=calculate_and_plot_zx(TR, Th, xrange, zrange, Nx, Nz)
% function E=calculate_and_plot_zx(TR, Th, xrange, zrange, Nx, Nz)
%
% Calculate transducer transmit field, one-way
% Use Field II
%
%   TR      Transducer definition
%   Th      Handle to transducer, given by Field II
%   xrange  Range to plot in x-dimension
%   zrange  Range to plot in z-dimension
%

if nargin < 6, Nz=300; end
if nargin < 5, Nx=300; end

%--- Define xz-points to include in plot ---
x= linspace(min(xrange), max(xrange), Nx)';
y= 0;
z= linspace(min(zrange), max(zrange), Nz)';

%--- Loop over all points and calcluate impulse response ---
%    For efficient calculation and not using too much memory,
%    results are calculated at all x-positions in one call, while 
%    looping over z.

fprintf('\nCalculating beam profile ')
pt= zeros(Nx,3);  % Define empty matrix of points, for efficient execution
E = zeros(Nz,Nx); % z as first dimansion, the 'vertical'
n0= 0;            % Used for display of progress 

for kz=1:Nz
    n1=floor(10*kz/Nz);
    if n1>n0
        n0=n1;
        fprintf(' %d', 10*n1)
    end
    for kx=1:Nx
        pt(kx,:)= [ x(kx) y z(kz)];  % Coordinates of point.
                                     % Vector of x-values for one z-value 
    end
    [hp,~]= calc_hp(Th,pt);   % Impulse response, all points at distance z(kz)
    E(kz,:)= sum(hp.^2,1);     % Total energy, all points at distance z(kz)
end
clear hp                       % hp can be big, remove to save memory

fprintf('  Finished\n')
return    