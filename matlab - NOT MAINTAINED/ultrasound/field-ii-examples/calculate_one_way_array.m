function E =calculate_one_way_array(TR, Th, xrange, zrange, Nx, Nz);
%function E =calculate_one_way_array(TR, Th, xrange, zrange);
% 
% Calculate transducer array transmit field, one-way
% Use Field II
%
%   prange  Range of points [xmin xmax ; ymin ymax ; zmin zmax]
%   PhP     Acoustic power in each point 
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

%--- Loop over all points. Calcluate impulse response ---
fprintf('\nCalculating beam profile ...')
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
        pt(kx,:)= [ x(kx) y z(kz)];   % Coordinates of point.
                                      % All x-values for one z-value 
    end
    [hp,~]=  calc_hp(Th,pt);   % Time response, all points at distance z(kz)
    E(kz,:)= sum(hp.^2,1);   % Total energy, all points at distance z(kz)
end
clear hp                       % hp can be big, remove to save memory

fprintf('  Finished\n')
       
return    