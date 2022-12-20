function a = timedlooptest
x=1:10;
y=x.^2;
h=plot(x(1),y(1));

a = timer('ExecutionMode','fixedRate','Period',0.1,'TimerFcn',@myfun,'TasksToExecute',10);
start(a);
wait(a);
disp('timer done')
end
function myfun(obj,evt)
for i=1:1
    disp(datestr(now));
     a=obj
%     e=evt.Data
end
disp('===============');
end