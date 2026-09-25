fs = 20e3;
T  = 3.0;
t  = (0:1/fs:T);
f  = 500+ [0 2];
A  = [1 0.8];

col= {'b-', 'r-', 'k-'};

N=length(f);
for k=1:N
    s(:,k)= A(k)*cos(2*pi*f(k)*t);
end
ss = sum(s,2);

Tmax= T/3;
Amax= sum(A);

figure(1)

subplot(3,1,1)
cla
for k=1:N
    plot(t,s(:,k), col{k});
    axis([0 Tmax  Amax*[-1 1]])
    hold on
end
hold off
xlabel('Time [s]')

subplot(3,2,3)
cla
for k=1:N
    plot(t,s(:,k), col{k});
    axis([0.25+0.005*[-1 1]   Amax*[-1 1]])
    hold on
end
plot(t,ss, col{3});
hold off
xlabel('Time [s]')
subplot(3,2,4)

cla
for k=1:N
    plot(t,s(:,k), col{k});
    axis([0.50+0.005*[-1 1]   Amax*[-1 1]])
    hold on
end
plot(t,ss, col{3});
hold off
xlabel('Time [s]')

subplot(3,1,3)
plot(t,ss, col{3});
axis([0 Tmax Amax*[-1 1]])
xlabel('Time [s]')

%soundsc(ss,fs)