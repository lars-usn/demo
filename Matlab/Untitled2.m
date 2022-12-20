h = animatedline;
axis([0,4*pi,-1,1])

numpoints = 1000;
x = linspace(0,4*pi,numpoints);
y = sin(x);
for k = 1:numpoints
    addpoints(h,x(k),y(k))
    drawnow limitrate
end