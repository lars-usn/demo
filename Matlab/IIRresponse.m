function IIRresponse(b,a)
% function FIRresponse(b)
% 
% Plot response of FIR filter

% Lars Hoff, March 2020

b=b(:)'; % Ensure b is a row vector, as required by 'zplane'
a=a(:)';    

N= length(b);
w= linspace(-pi,pi,2001);   % Normalized frequency
z= exp(1i*w);               % z on unit circle

%--- System function H(z) calculated from definition ---
H=freqz(b,a,w);

%=== Plot response in four graphs ===
%--- Pole-zero plot ---
subplot(2,2,1)
zplane(b,a)

%--- Frequency response ---
subplot(2,2,2)
plot(w,abs(H))
PiScaledAxis(gca,'x','Normalized frequency',3)
ylabel('Magnitude |H|')
Hmin= min([0,min(abs(H))]);
Hmax= max([0,max(abs(H))]);
ylim([Hmin,Hmax]);
grid on

subplot(2,2,4)
plot(w,angle(H) )
PiScaledAxis(gca,'x','Normalized frequency',3)
PiScaledAxis(gca,'y','Phase',2)
grid on

%--- Impulse response ---
N = 20;
nz =2;
n = (-nz:N);          % Plot a few zero-points before and after the filter length
x= zeros(size(n));
x(nz+1)=1;
h = filter(b,a,x);
ni= (n>=0 & n<N);      % Part of n covered by the filter

subplot(2,2,3)
stem(n,h,'filled')
xlim([min(n) max(n)])
ylim(1.2*[min(h) max(h)])
grid on
xlabel ('n')
ylabel ('h[n]')

