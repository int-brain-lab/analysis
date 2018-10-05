% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath('~/Documents/code/gramm/');

if ispc,
    usr = getenv('USERNAME');
    homedir = getenv('USERPROFILE');
    datapath = fullfile(homedir, 'Google Drive');
elseif ismac,
    usr = getenv('USER');
    homedir = getenv('HOME');
    datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
end

data = readAlf_allData(datapath, ...
    {'6722', '6723', '6724', ...
    '6725', '6726', ...
    '4573', '4576', '4577', '4579', '4580', '4581', '4619'});
%, ...
   % '4573', '4576', '4577', '4579', '4580', '4581', '4619'});

% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);

% PLOT
close;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'subset', data.dayidx_rev > -5);
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)');
g.facet_wrap(data.animal, 'ncols', 4);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.axe_property('ylim', [0 1], 'xlim', [-105 100]);
g.set_color_options('map', zeros(length(unique(data.animal)), 3));
g.no_legend;
custom_psychometric(g);
g.draw()

% save
g.export('file_name', fullfile(datapath, 'CSHL', 'figures', 'choiceWorld_comparison.pdf'))
print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', 'Lisbon_data.pdf'));


close all;

data = readAlf_allData(datapath, ...
    {'4581', 'Axon'});
%, ...
   % '4573', '4576', '4577', '4579', '4580', '4581', '4619'});

% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);

% PLOT
close;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'subset', data.dayidx_rev > -5);
g.set_names('x', 'Contrast (%)', 'y', 'P(rightwards)');
g.facet_wrap(data.lab, 'ncols', 4);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.axe_property('ylim', [-0.01 1], 'xlim', [-105 100], 'ytick', [0 0.5 1], 'yticklabel', [0 50 100]);
g.set_color_options('map', zeros(length(unique(data.animal)), 3));
g.no_legend;
custom_psychometric(g);
g.draw()

% save
print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', 'Lisbon_data_4581.pdf'));

