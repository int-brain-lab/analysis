
% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);
axis_square =  @(gramm_obj) gramm_obj.axe_property('PlotBoxAspectRatio', [1 1 1]);

