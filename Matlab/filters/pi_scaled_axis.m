function  pi_scaled_axis(hax,ax,label,N,w)
% function  pi_scaled_axis(hax,ax,label,N,w)
%
% Format axis scaled and labeled in units of pi
% Use for normalized frequency axis (omega hat) and phase
%
% hax   Handle to graph axes
% ax    Axis to use: 'x' or 'y'
% label Text on label
% N     Label positions as pi/N
% w     Frequency axis

% Lars Hoff, USN, Feb 2020

if nargin<5, w=[-pi,pi]; end
if nargin<4, N=4;        end
if nargin<3, label='';   end

%--- Define x or y axis
lim = sprintf('%slim',ax);
tick= sprintf('%stick',ax);
ticklabel= sprintf('%sticklabel',ax);
labelname= sprintf('%slabel',ax);

%--- Label names ---
switch N
    case 2,    set(hax, ticklabel, {'-\pi','-\pi/2','0','\pi/2','\pi'} );
    case 3,    set(hax, ticklabel, {'-\pi','-2\pi/3','-\pi/3','0','\pi/3','2\pi/3','\pi'} );
    case 4,    set(hax, ticklabel, {'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'} );
    case 6,    set(hax, ticklabel, {'-\pi','-5\pi/6','-2\pi/3','-\pi/2','-\pi/3','-\pi/6','0','\pi/6','\pi/3','\pi/2','2\pi/3','5\pi/6','\pi'} );
    otherwise, N=2; set(hax, ticklabel, {'-\pi','-\pi/2','0','\pi/2','\pi'} );
end

%--- Axis scale ---
wmax= ceil(max(w)/pi);
wmin= floor(min(w)/pi);
set(hax, lim, pi*[wmin wmax]);
dw=1/N;
set(hax, tick, pi*[wmin:dw:wmax] );

feval(labelname, hax, label)

end

