function pole_zero_3d(b,a,dispformat)
% function PoleZero3D(b,a,dispformat)
%  3D illustration of pole-zero plot in the z-plane and frequency response
%
%   a  Forward filter coefficients.   
%   b  Backward filter coefficients.  
%
% dispformat (optional)  "dB" for dB-scale display, otherwise linear
%
% Example 1 (Chapter 10-13 in McClellan et sal., DSP first)
%              [b,a]= ellip( 3, 1, 30, 0.3 );  
%              PoleZero3D(b,a)
%
% Example 2  (Similar to Ex. 1, illustrating different filter topology)  
%              [b,a]= butter( 3, 0.3 );  
%              PoleZero3D(b,a)

% Lars Hoff, USN, Feb 2023 (Updated April 2023)

if nargin<3, dispformat='abs'; end

xmax = 1.5;
zmax = 3;

%% Create grid
x=linspace(-xmax, xmax, 100);
[zr,zim]=meshgrid(x,x);
z=zr+1i*zim;

%% Calculate response
H=Hz(b,a,z, dispformat);
surf(zr, zim, H, 'EdgeColor', 'none' );
axis equal
zlim([ 0 zmax ]);
clim([ 0 zmax ]);
colorbar

%% Create markers
Np = 100;
phi= linspace(0,2*pi,Np);
zu = exp(1i*phi);

Hu= Hz(b,a,zu,   dispformat);  % Unit circle
Hr= Hz(b,a,x,    dispformat);  % Real axis
Hi= Hz(b,a,1i*x, dispformat);  % Imaginary axis

hold on
plot3( x, zeros(size(x)),  Hr,'k-', 'linewidth', 1)
plot3( zeros(size(x)), x,  Hi,'k-', 'linewidth', 1)
plot3( real(zu), imag(zu), Hu,'k-', 'linewidth', 1)
hold off

xlabel('Re\{z\}')
ylabel('Im\{z\}')
zlabel('|H(z)|')

end

%% Internal functions
function H=Hz(b,a,z, dispformat)
N=length(a)-1;
A=zeros(size(z));
for n=0:N
    A=A+ a(n+1)*z.^(-n);
end

M=length(b)-1;
B=zeros(size(z));
for k=0:M
    B=B+ b(k+1)*z.^(-k);
end
H=B./A;

switch lower(dispformat)
    case "db",  H=20*log10(abs(H));
    otherwise,  H=abs(H);
end
end