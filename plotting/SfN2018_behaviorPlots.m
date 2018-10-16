

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath(genpath('/Users/urai/Documents/code/analysis_IBL'));
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

close all;
set(groot, 'defaultaxesfontsize', 8, 'DefaultFigureWindowStyle', 'normal');

mice       = {'IBL_34', 'LEW010', '438'};
data_all   = readAlf_allData(datapath, mice);

% % TAKE THE LAST 3 DAYS BEFORE BIASED BLOCKS ARE INTRODUCED
% data = data(data.probabilityLeft == 0.5, :);
% data.dayidx_rev2 = data.dayidx_rev;
% for m = unique(data.animal)',
%     data.dayidx_rev2(contains(data.animal, m)) = data.dayidx_rev(contains(data.animal, m)) ...
%         - max(data.dayidx_rev(contains(data.animal, m)));
% end

%% overview - mice that are doing well!

data = data_all(data_all.probabilityLeft == 0.5, :);

% TAKE THE LAST 3 DAYS BEFORE BIASED BLOCKS ARE INTRODUCED
data.dayidx_rev2 = data.dayidx_rev;
for m = unique(data.animal)',
    data.dayidx_rev2(contains(data.animal, m)) = data.dayidx_rev(contains(data.animal, m)) ...
        - max(data.dayidx_rev(contains(data.animal, m)));
end
    
% PLOT FOR THE LAST WEEK 
close all; clear g;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'subset', (data.dayidx_rev2 > -6));
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.facet_wrap(data.name, 'ncols', 3);
g.stat_summary('type', 'bootci', 'geom', {'errorbar', 'point'});
g.set_color_options('map', [0 0 0]); % red and blue as schematic
g.no_legend();
%g.geom_vline('extent', 20, 'style', ':k');
custom_psychometric(g);

% general layout
g.axe_property('ylim', [0 1], 'xlim', [-110 101], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
axis_square(g);
g.draw();
g.export('file_name', '~/Google Drive/Rig building WG/Posters/SfN2018/psychometrics_examples.pdf', 'file_type', 'pdf');
g.export('file_name', '~/Google Drive/Rig building WG/Posters/SfN2018/psychometrics_examples.eps', 'file_type', 'eps');

%% BIASED BLOCKS FOR THE 2 THAT HAVE THESE DATA
data = data_all(data_all.probabilityLeft ~= 0.5 & ~isnan(data_all.probabilityLeft), :);
data.probabilityLeft(data.probabilityLeft < 0.5) = 0.2;
data.probabilityLeft(data.probabilityLeft > 0.5) = 0.8;

% plot
close all; clear g;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.probabilityLeft);
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.facet_wrap(data.name, 'ncols', 3);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
rdBu = cbrewer('qual', 'Set1', 3);
g.set_color_options('map', rdBu([2 1], :)); % red and blue as schematic
g.no_legend();
g.geom_vline('extent', 20, 'style', '--k');
custom_psychometric(g);

% general layout
g.axe_property('ylim', [0 1], 'xlim', [-110 100], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 16);
axis_square(g);
g.draw();
g.export('file_name', '~/Google Drive/Rig building WG/Posters/SfN2018/psychometrics_biased.eps', 'file_type', 'eps');

%% LEARNING CURVES!

% overview
mice = {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
     'IBL_1', 'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10', ...
     'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17', ...
     'LEW009', 'LEW010', 'ALK081', 'LEW008', '6812', '6814', '437', '438'};
data_all   = readAlf_allData(datapath, mice);
data_all.dayidx = data_all.dayidx - min(data_all.dayidx) + 1; % make sure the 1st day where there is data (not an empty folder) is dayidx 1

for m = unique(data_all.animal)',
    data_all.dayidx(contains(data_all.animal, m)) = data_all.dayidx_rev(contains(data_all.animal, m)) ...
        - min(data_all.dayidx_rev(contains(data_all.animal, m))) + 1;
end

%%  SHOW LEARNING CURVES - % CORRECT ON >50% CONTRASTS
data_all.response(abs(data_all.signedContrast) < 50) = NaN;
data_all.correct(abs(data_all.signedContrast) < 50) = NaN;

[gr, sj, days] = findgroups(data_all.animal, data_all.dayidx);
tab = array2table(days, 'variablenames', {'days'});
tab.animal = sj;
tab.performance = splitapply(@nanmean, data_all.correct, gr);
tab.lab = splitapply(@unique, data_all.lab, gr);

% plot
close all; clear g;
g = gramm('x', tab.days, 'y', 100 * tab.performance, 'group', tab.animal, 'color', tab.lab);
g.set_names('x', 'Training days', 'y', {'Performance'  'on easy trials (%)'});
%g.facet_wrap(data.name, 'ncols', 3);
g.geom_line()
g.geom_point()
cols = cbrewer('qual', 'Set2', 8);
g.set_color_options('map', cols([1 3 5], :));
%g.no_legend();
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 16);
g.geom_hline('yintercept', 50, 'style', '-k');
g.axe_property('ylim', [0 100], 'ytick', [0 50 100],  'yticklabel', [0 50 100], 'xlim', [0 roundn(max(data_all.dayidx), 1)]);
g.draw();

g.update('group', ones(size(tab.animal)), 'color', ones(size(tab.animal)));
g.stat_summary('type', 'ci', 'geom', 'line');
g.set_color_options('map', [ 0 0 0]); % black
g.draw()
g.results.stat_summary.line_handle.LineWidth = 4;
set(gcf, 'Position',   [560   722   804   226])
g.export('file_name', '~/Google Drive/Rig building WG/Posters/SfN2018/learningCurves.eps', 'file_type', 'eps');


