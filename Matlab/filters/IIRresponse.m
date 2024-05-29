function iirresponse( b, a, displayformat)
% function iirresponse( b, a, displayformat)
% 
% Plot response of IIR filter

% Lars Hoff, March 2020
%   Updated Nov 2023, LH
%           May 2024, LH

%% Organise and initialise
if nargin < 3
    displayformat = "linear";
end

% Ensure a and b are row vectors as required by 'zplane'
b=b(:)'; 
a=a(:)';    

N = length( b );
w = linspace( -pi, pi, 2001 );   % Normalized frequency vector

% calculate system function H(z) from definition 
H = freqz( b, a, w );

%% Pole-zero plot
subplot(2,2,1)
[ hz, hp, ht ]=zplane( b,a );
xlabel( 'Re\{z\}' )
ylabel( 'Im\{z\}' )

z0 = hz.XData+1i*hz.YData;    % Zeros
zp = hp.XData+1i*hp.YData;    % Poles
stable = all( abs(zp)<1 );
title('')

%% Frequency response
subplot(2,2,2)
Hmin= min( [ 0, min(abs(H)) ] );
Hmax= max( [ 0, max(abs(H)) ] );

% Remove frequency response if the system is unstable
if not(stable)  
    H = NaN*ones( size(H) );
end

if lower(displayformat) =="db"
    plot( w, 20*log10( abs(H)) )
    ylim ( 20*log10( Hmax ) + [-40 0] )
    ylabel( '|H| [dB]')
else
    plot( w, abs(H) )
    ylim( [ Hmin, Hmax ] );
    ylabel( '|H|' )
end

wAxisLabel = '$\hat \omega$';
piaxis(gca,'x', wAxisLabel, 3 )
grid on

subplot(2,2,4)
plot(w,angle(H) )
piaxis( gca, 'x', wAxisLabel, 3 )
piaxis( gca, 'y', '$\angle H$',2)
grid on

%% Impulse response 
N = 20;

% Create impulse and calculate response
nZero =2;
n = ( -nZero:N ) ;     % Include some zero-points before and after the filter length
x = zeros( size(n) );
x( nZero+1 )=1;         
h = filter( b, a, x );

subplot(2,2,3)
stem( n, h, 'filled' )
xlim( [ min(n) max(n) ] )
ylim( 1.2*[ min(h) max(h) ] )
grid on
xlabel ('n')
ylabel ('h[n]')

