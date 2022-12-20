function results = AddPoint(app,event)
addpoints(app.Display.Graph, app.Response.t(k), app.Response.s(k) )
app.Gauge.Value=app.Response.s(k);
drawnow limitrate
results=0;

end

