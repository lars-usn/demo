t = timer;
x=1:10;
y=x.^2;
t.StartFcn = @(~,thisEvent)k=1; plot(x(1),y(1));
t.TimerFcn = @(~,thisEvent)plot(x(k),y(k));;
t.StopFcn = @(~,thisEvent)disp([thisEvent.Type ' executed '...
    datestr(thisEvent.Data.time,'dd-mmm-yyyy HH:MM:SS.FFF')]);
t.Period = 2;
t.TasksToExecute = 3;
t.ExecutionMode = 'fixedRate';
start(t)