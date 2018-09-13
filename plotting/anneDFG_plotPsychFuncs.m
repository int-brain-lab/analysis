%function data = plotPsychFuncs_overSessions_allLabs()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath('~/Documents/code/gramm/');
addpath('~/Google Drive/IBL_DATA_SHARE/CSHL/code');

clear all; close all; clc;
%eval(gramm_helperfuncs);
set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');

data = readAlf_allData([], {'Myelin', 'Mouse2', 'Axon', 'Arthur', 'M5', 'M6', 'M7', ...
    'IBL_34', 'IBL_1b'});

% data = readAlf_allData([], {'Myelin', 'Mouse2', 'Axon', 'Arthur', 'M5', 'M6', 'M7'});
data(data.inclTrials ~= 1, :) = [];

% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);
axis_square =  @(gramm_obj) gramm_obj.axe_property('PlotBoxAspectRatio', [1 1 1]);

%% PSYCHOMETRIC FUNCTIONS FOR ALL MICE
close all; clear g
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.name,...
    'subset', (data.dayidx > 10));
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.set_color_options('map', cbrewer('qual', 'Set2', numel(unique(data.animal)))); % black

% add a line for the psychometric function fit
custom_psychometric(g);
g.no_legend();
g.draw()

% overlay the summary psychometric in black for the later sessions
g.update('color', ones(size(data.animal)));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'bootci', 'geom', 'point');
g.set_color_options('map', [0 0 0]); % black
custom_psychometric(g);
g.axe_property('ylim', [0 1], 'xlim', [-110 100], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 18);
g.geom_vline('extent', 20, 'style', '--k');
axis_square(g);
g.draw()

% make the average more salient
g.results.stat_fit.line_handle.LineWidth = 2;
g.results.stat_summary.point_handle.MarkerSize = 8;
g.export('file_name', '~/Dropbox/Proposals/2018 DFG/figures/psychfuncs.eps', 'file_type', 'eps');

%% PSYCHOMETRIC FUNCTIONS FOR ALL MICE - shifted by history
data.prevresp       = circshift(data.response, 1);
data.prevcorrect    = circshift(data.correct, 1);
close all; clear g;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.prevresp, ...
    'subset', (data.dayidx > 10) & data.prevcorrect == 0);
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
rdBu = cbrewer('qual', 'Set1', 3);
g.set_color_options('map', rdBu([2 1], :)); % red and blue as schematic
g.no_legend();
g.geom_vline('extent', 20, 'style', '--k');
custom_psychometric(g);

% general layout
g.axe_property('ylim', [0 1], 'xlim', [-110 100], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 18);
axis_square(g);
g.draw();
g.export('file_name', '~/Dropbox/Proposals/2018 DFG/figures/historybias.pdf', 'file_type', 'pdf');
g.export('file_name', '~/Dropbox/Proposals/2018 DFG/figures/historybias.eps', 'file_type', 'eps');

%% HISTORY STRATEGY SPACE

% ============================================================== %
% STRATEGY PLOT
% ============================================================== %

data.prevresp_success    = data.prevresp;
data.prevresp_failure    = data.prevresp;
data.prevresp_success(data.prevcorrect == 0) = 0;
data.prevresp_failure(data.prevcorrect == 1) = 0;

busseFit = @(stim, success, failure, resp) {glmfit([stim, success, failure], (resp > 0), 'binomial')};
betas    = splitapply(busseFit, data.signedContrast / 100, data.prevresp_success, data.prevresp_failure, data.response, findgroups(data.animal));
historyBetas = cat(2, betas{:})';

close all; clear g
g = gramm('x', historyBetas(:, 3), 'y', historyBetas(:, 4), 'color', unique(data.name));
g.geom_point();
g.set_names('x', '\beta_{success}', 'y', '\beta_{failure}');
g.set_color_options('map', cbrewer('qual', 'Set2', numel(unique(data.animal)))); % black
g.no_legend();
axis_square(g);
g.axe_property('ylim', [-1 1], 'xlim', [-1 1], 'ytick', [-1:0.5:1], 'xtick',[-1:0.5:1]);
g.geom_hline('extent', 20, 'style', '--k');
g.geom_vline('extent', 20, 'style', '--k');
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 20, 'interpreter','tex');
g.draw();

for i = 1:length(unique(data.animal)),
    g.results.geom_point_handle(i).MarkerSize = 15;
    g.results.geom_point_handle(i).MarkerEdgeColor = 'w';
end

g.export('file_name', '~/Dropbox/Proposals/2018 DFG/figures/strategyplot.eps', 'file_type', 'eps');


