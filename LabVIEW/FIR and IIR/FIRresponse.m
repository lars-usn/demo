function FIRresponse(b)
% function FIRresponse(b)
% 
% Plot response of FIR filter

% Lars Hoff, March 2020

b=b(:)'; % Ensure b is a row vector, as required by 'zplane'
a=1;     % Feedback coefficients, not used for FIR-filter (always one)

N= length(b);
w= linspace(-pi,pi,2001);   % Normalized frequency
z= exp(1i*w);               % z on unit circle

%--- System function H(z) calculated from definition ---
H= zeros(size(z));
for k=1:N
    H=H+b(k)*z.^(-(k-1));  
end

%=== Plot response in four graphs ===
%--- Pole-zero plot ---
subplot(2,2,1)
zplane(b,a)

%--- Frequency response ---
subplot(2,2,2)
plot(w,abs(H))
PiScaledAxis(gca,'x','Normalized frequency',3)
ylabel('Magnitude |H|')
grid on

subplot(2,2,4)
plot(w,angle(H) )
PiScaledAxis(gca,'x','Normalized frequency',3)
PiScaledAxis(gca,'y','Phase',2)
grid on

%--- Impulse response ---
n = (-2:N+4);          % Plot a few zero-points before and after the filter length
h = zeros(size(n));
ni= (n>=0 & n<N);      % Part of n covered by the filter
h(ni)=b;               % Impulse response = Filter coefficients

subplot(2,2,3)
stem(n,h,'filled')
xlim([min(n) max(n)])
ylim(1.2*[min(h) max(h)])
grid on
xlabel ('n')
ylabel ('h[n]')

