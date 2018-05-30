%function data = plotPsychFuncs_overSessions_allLabs()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
addpath('~/Desktop/code/gramm/');
%eval(gramm_helperfuncs);
data = readAlf_allData();

% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);

%% remove mice that have less then 2 weeks of data from all plots
[gr, mousename]             = findgroups(data.animal);
trainingdays_permouse       = splitapply(@(x) numel(unique(x)), data.date, gr);
data(ismember(data.animal, mousename(trainingdays_permouse < 14)), :) = [];

data(data.dayidx > 30, :) = [];

%% for now, handcode which mice are trained on manual vs automatic protocols
mice_manualtraining = {'ALK068', 'ALK070', 'ALK071', 'ALK074', 'ALK075'};
mice_automatedtraining = unique(data.animal);
mice_automatedtraining(ismember(mice_automatedtraining, mice_manualtraining)) = [];

%% how many mice did each institution train on the automated protocol?
labs = unique(data.lab);
fprintf('%d mice at %s, %d at %s, %d at %s \n', ...
    sum(ismember(mice_automatedtraining, data.animal(strcmp(data.lab, labs{1})))), labs{1}, ...
    sum(ismember(mice_automatedtraining, data.animal(strcmp(data.lab, labs{2})))), labs{2}, ...
    sum(ismember(mice_automatedtraining, data.animal(strcmp(data.lab, labs{3})))), labs{3});

%% also remove mice manually that did not learn the task at all
% 2 in Lisbon, 1 at CSHL, X in London
[gr, mousename]             = findgroups(data.animal);
asymptotic_performance      = splitapply(@(correct, contrast, days) nanmean(correct(abs(contrast) >= 80 & days > 10)), ...
    data.correct, data.signedContrast, data.dayidx, gr);
bar(asymptotic_performance); set(gca, 'xtick', 1:length(mousename), 'xticklabel', mousename', 'xticklabelrotation', -30); grid on;
data(ismember(data.animal, mousename(asymptotic_performance < 0.7)), :) = [];

%% BEHAVIOR ON MANUALLY TRAINED MICE
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.animal,...
    'subset', (ismember(data.animal, mice_manualtraining) & data.dayidx > 10));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)');
g.facet_wrap(data.lab, 'ncols', 3);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.no_legend();
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.stat_summary('type', 'sem', 'geom', 'line');
% g.set_color_options('map', map); 
g.draw()

% overlay the summary psychometric in black for the later sessions
g.update('color', ones(size(data.animal)));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.stat_summary('type', 'sem', 'geom', 'line');

% g.stat_summary('type', 'sem', 'geom', 'point');
% g.stat_summary('type', 'sem', 'geom', 'line');
%custom_psychometric(g);
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.axe_property('ylim', [0 1], 'xlim', [-105 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 12);

g.draw();

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_manualtraining.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_manualtraining.png');
print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_manualtraining.eps');

%% COMPARE STABLE BEHAVIOR ACROSS LABS - automated training
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.name,...
    'subset', (data.dayidx > 10 & ismember(data.animal, mice_automatedtraining)));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)');
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.stat_summary('type', 'sem', 'geom', 'line');
g.facet_wrap(data.lab, 'ncols', 3);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 14);
g.no_legend();

% colormap: one color type for each lab
cmap = linspecer(30);
cmap = cmap([1:7 9:11 21:30], :);
blues = cbrewer('seq', 'Blues',8);
individual_cmap = [cbrewer('seq', 'Oranges', 7); blues([5 8], :); cbrewer('seq', 'Greens', 9)];

g.set_color_options('map', individual_cmap	); % black
g.axe_property('ylim', [0 1], 'xlim', [-110 100], 'ytick', [0 0.5 1]);

g.draw()

%g.set_title('Most recent training week for each mouse');
%g.set_color_options('map', repmat(cbrewer('qual', 'Pastel1', 8), 10,1 )); 
%g.set_color_options('map', repmat(linspecer(21), 10,1 )); 

% overlay the summary psychometric in black for the later sessions
% g.update('color', ones(size(data.animal)));
% g.stat_summary('type', 'bootci', 'geom', 'area');
% g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
% axis_square(g);

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_all_perlab.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_all_perlab.png');
print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_all_perlab.eps');

[gr, names] = findgroups(data.name);
maxdays = splitapply(@max, data.dayidx, gr);
maxweeks = maxdays / 5;
fprintf('Length of training: %d-%d days at CCU, %d-%d at UCL, %d-%d at CSHL \n', ...
  min(maxdays(strncmp(names, 'Lisbon', 2))),  max(maxdays(strncmp(names, 'Lisbon', 2))), ...
    min(maxdays(strncmp(names, 'London', 2))),  max(maxdays(strncmp(names, 'London', 2))), ...
  min(maxdays(strncmp(names, ' NY', 2))),  max(maxdays(strncmp(names, ' NY', 2))));

% UCL mice that have been trained manually 
% 'ALK068', 'ALK070', 'ALK071', 'ALK074' and 'ALK075' 
% ends


%% PSYCHOMETRIC FUNCTION PER LAB
% ONLY USE MICE THAT ARE CONSIDERED TRAINED!
% correct = data.correct;
% correct(abs(data.signedContrast) < 80) = NaN;
% correct(data.dayidx < 11) = NaN;
% [gr, animalName] = findgroups(data.animal);
% correctPerMouse = splitapply(@nanmean, correct, gr);
% goodAnimals = animalName(correctPerMouse > 0.6);

goodAnimals = {'4581', '4619', 'M6', 'Mouse2', 'MW45', 'ALK068'};
close all;
data.name(ismember(data.name, {'Lisbon 4581'})) = {' Lisbon 4581'};
data.name(ismember(data.name, {'NY Mouse2'})) = {' NY Mouse2'};
data.name(ismember(data.name, {'London MW45'})) = {' London MW45'};

g = gramm('x', data.signedContrast, 'y', double(data.response > 0), ...
    'subset', (data.dayidx_rev > -7 & ismember(data.animal, goodAnimals)));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'Row', []);
custom_psychometric(g);

% overlay logistic fit in black
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.facet_wrap(data.name, 'ncols', 3);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.no_legend();
g.set_title('Most recent training week for each mouse');
%axis_square(g);
g.draw();

% summary stats
g.update();
g.stat_summary('type', 'bootci', 'geom', 'errorbar', 'setylim', 1);
g.stat_summary('type', 'bootci', 'geom', 'point', 'setylim', 1);
red = linspecer(2);
g.set_color_options('map', [ 0.5 0.5 0.5]); % black
g.draw();

print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_perlab.eps');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_perlab.png');
data.name(ismember(data.name, {' NY Mouse2'})) = {'NY Mouse2'};
data.name(ismember(data.name, {' Lisbon 4581'})) = {'Lisbon 4581'};
data.name(ismember(data.name, {' London MW45'})) = {'London MW45'};

%% OVERVIEW MOSAIC OF ALL ANIMALS
bestAnimals = unique(data.animal);
% bestAnimals(~cellfun(@isempty, strfind(bestAnimals, 'ALK'))) = [];
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.dayidx, ...
    'subset', ismember(data.animal, bestAnimals));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'Days', 'Row', 'animal');
g.set_continuous_color('active', false);  
g.set_color_options('map', flipud(plasma(150)));
g.stat_summary('type', 'std', 'geom', 'line'); % no errorbars within a session
g.facet_wrap(data.name, 'ncols', 7);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.draw();

% SUMMARY IN BLACK; LAST WEEK'S SESSIONS
g.update('color', ones(size(data.dayidx)), 'subset', ismember(data.animal, bestAnimals) & data.dayidx_rev > -7);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
custom_psychometric(g);
%g.stat_summary('type', 'std', 'geom', 'line'); % hack to get a connected errorbar
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.draw();

% ADD A COLORBAR FOR SESSION NUMBER 
colormap(flipud(plasma));
subplot(7,8,56);
c = colorbar;
c.Location = 'EastOutside';
axis off;
prettyColorbar('Training days');
c.Ticks = [0 1];
c.TickLabels = [min(data.dayidx) max(data.dayidx)];

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.png');
print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.eps');

%% SAME FOR REACTION TIMES

% leave out 0% contrast
close all;
g = gramm('x', abs(data.signedContrast), 'y', data.rt, 'color', data.dayidx, ...
    'subset', (abs(data.signedContrast) > 0 & ismember(data.animal, mice_automatedtraining)));
g.set_names('x', 'Contrast (%)', 'y', 'RT (ms)', 'color', 'Days', 'Column', 'animal');
g.set_continuous_color('active', false);  
g.set_color_options('map', flipud(plasma(100)));
g.stat_summary('type', 'quartile', 'geom', 'line', 'setylim', 1); % no errorbars within a session
g.facet_wrap(data.name, 'ncols', 8);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.axe_property('ylim', [0 2]);
g.draw();

% overlay the summary psychometric in black for the later sessions
g.update('x', abs(data.signedContrast), 'y', data.rt, ...
   'color', ones(size(data.dayidx)), 'subset', (data.dayidx > 7 & abs(data.signedContrast) > 0));
g.stat_summary('type', 'quartile', 'geom', 'line', 'setylim', 1); % hack to get a connected errorbar
% g.stat_summary('type', 'quartile', 'geom', 'errorbar', 'setylim', 1); % hack to get a connected errorbar
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.draw();

% ADD A COLORBAR FOR SESSION NUMBER
colormap(flipud(plasma));
subplot(7,8,56);
c = colorbar;
c.Location = 'EastOutside';
axis off;
prettyColorbar('Sessions');
c.Ticks = [0 1];
c.TickLabels = {'early', 'late'};

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/rts_alllabs.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/rts_alllabs.png');


%% LEARNING RATES ACROSS 
close all;
g = gramm('x', data.dayidx, 'y', data.correct, 'color', data.name,...
    'subset', (abs(data.signedContrast) >= 50 & data.dayidx < 31 &  ismember(data.animal, mice_automatedtraining)));
g.set_names('x', 'Training day', 'y', 'Performance on easy trials (%)', 'color', 'Lab', 'Row', 'Lab');
g.geom_hline('yintercept', 0.5);
g.set_color_options('map', individual_cmap); % black
g.stat_summary('type', 'sem', 'geom', 'line', 'setylim', 1); % no errorbars within a session
%g.facet_wrap(data.lab, 'ncols', 3);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 14);
g.axe_property('ylim', [0.3 1], 'xlim', [0 30]);
g.no_legend();
g.draw();

% overlay the summary psychometric in black for the later sessions
g.update('color', ones(size(data.lab)), 'group', ones(size(data.animal)));
g.stat_summary('type', 'sem', 'geom', 'line', 'setylim', 1); % hack to get a connected errorbar
% g.stat_summary('type', 'bootci', 'geom', 'errorbar', 'setylim', 1); % hack to get a connected errorbar
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.draw();

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/learningrates_alllabs.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/learningrates_alllabs.png');
print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/learningrates_alllabs.eps');

% UCL mice that have been trained manually 
% 'ALK068', 'ALK070', 'ALK071', 'ALK074' and 'ALK075' 
% ends


%% REWARD SHIFTED
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.highRewardSide, ...
    'subset', (~isnan(data.highRewardSide) & ismember(data.animal, 'LEW006')));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'High reward side');
g.set_color_options('map', linspecer(2));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
custom_psychometric(g);
g.facet_wrap(data.name);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.axe_property('ylim', [0 1], 'xlim', [-105 100]);
g.no_legend();
g.draw();
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/rewardshift.png');

