function PoleZero3D(b,a,dispformat)
% function PoleZero3D(b,a,dispformat)

if nargin<3, dispformat='abs'; end

%% Create grid
xmax=1.4;
x=linspace(-xmax, xmax, 70);
[zr,zim]=meshgrid(x,x);
z=zr+1i*zim;

%% Calculate response
H=Hz(b,a,z, dispformat);
mesh(zr, zim, H );
axis equal
zlim([0 10]);

%% Create markers
Np = 100;
phi= linspace(0,2*pi,Np);
zu = exp(1i*phi);

Hu= Hz(b,a,zu,   dispformat);  % Unit circle
Hr= Hz(b,a,x,    dispformat);  % Real axis
Hi= Hz(b,a,1i*x, dispformat);  % Imaginary axis

hold on
plot3( x, zeros(size(x)),  Hr,'k-', 'linewidth', 3)
plot3( zeros(size(x)), x,  Hi,'k-', 'linewidth', 3)
plot3( real(zu), imag(zu), Hu,'k-', 'linewidth', 3)
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