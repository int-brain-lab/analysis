%function data = plotPsychFuncs_overSessions_allLabs()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
addpath('~/Desktop/code/gramm/');
%eval(gramm_helperfuncs);
data = readAlf_allData();
 
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
data.name(ismember(data.name, {'CCU 4581'})) = {' CCU 4581'};
data.name(ismember(data.name, {'CSHL Mouse2'})) = {' CSHL Mouse2'};
data.name(ismember(data.name, {'UCL MW45'})) = {' UCL MW45'};

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
g.set_color_options('map', repmat(red(1, :), 3, 1)); % black
g.draw();

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_perlab.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_perlab.png');


%% OVERVIEW MOSAIC OF ALL ANIMALS
bestAnimals = unique(data.animal);
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.dayidx, ...
    'subset', ismember(data.animal, bestAnimals));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'Days', 'Row', 'animal');
g.set_continuous_color('active', false);  
g.set_color_options('map', flipud(plasma(150)));
g.stat_summary('type', 'std', 'geom', 'line'); % no errorbars within a session
g.facet_wrap(data.name, 'ncols', 9);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.draw();

% SUMMARY IN BLACK; LAST WEEK'S SESSIONS
g.update('color', ones(size(data.dayidx)), 'subset', ismember(data.animal, bestAnimals) & data.dayidx_rev > -7);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'std', 'geom', 'line'); % hack to get a connected errorbar
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

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.png');
print(gcf, '-depsc', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_alllabs.eps');

%% SAME FOR REACTION TIMES

% leave out 0% contrast
close all;
g = gramm('x', abs(data.signedContrast), 'y', data.rt, 'color', data.dayidx, 'subset', abs(data.signedContrast) > 0);
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
    'subset', (abs(data.signedContrast) >= 50 & data.dayidx < 31));
g.set_names('x', 'Training day', 'y', 'Performance on each trials (%)', 'color', 'Animal', 'Row', 'Lab');
g.geom_hline('yintercept', 0.5);
g.set_color_options('map', repmat(linspecer(10, 'qualitative'), 5, 1));
g.stat_summary('type', 'sem', 'geom', 'line', 'setylim', 1); % no errorbars within a session
g.facet_wrap(data.lab, 'ncols', 3);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
% g.set_title('Performance on > 80% contrast trials');
g.draw();

% overlay the summary psychometric in black for the later sessions
g.update('color', data.lab);
g.stat_summary('type', 'sem', 'geom', 'area', 'setylim', 1); % hack to get a connected errorbar
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.no_legend();
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
    'subset', ~isnan(data.highRewardSide));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'High reward side');
g.set_color_options('map', linspecer(2));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
custom_psychometric(g);
g.facet_wrap(data.name);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.draw();
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/rewardshift.png');
