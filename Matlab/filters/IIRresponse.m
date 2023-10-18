function IIRresponse( b, a, displayformat)
% function IIRresponse( b, a, displayformat)
% 
% Plot response of FIR filter

% Lars Hoff, March 2020

if nargin < 3
    displayformat = "linear";
end

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
[hz, hp, ht] = zplane(b,a);
z0= hz.XData+1i*hz.YData;    % Zeros
p = hp.XData+1i*hp.YData;    % Poles
stable=all(abs(p)<1);
            
title('')

%--- Frequency response ---
subplot(2,2,2)

Hmin= min([0,min(abs(H))]);
Hmax= max([0,max(abs(H))]);

if not(stable)
    H=NaN*H;
end

if lower(displayformat) =="db"
    plot(w, 20*log10( abs(H)) )
    %    ylim ( 20*log10( Hmax ) + [-40 0] )
    ylim (  [-40 0] )
    ylabel('Magnitude |H| [dB]')

else
    plot(w, abs(H) )
    ylim( [ Hmin, Hmax ] );
    ylabel('Magnitude |H|')
end

PiScaledAxis(gca,'x','Normalized frequency',3)
grid on
if not(stable)
    text( 0, mean([Hmin,Hmax]), 'Unstable', HorizontalAlignment='center' )
end

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

