function rewardShiftedPsychometricFunction


% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
data = readAlf_allData();
addpath('~/Desktop/code/gramm/');
data(isnan(data.highRewardSide), :) = [];

%% FIRST, REPLICATE LAUREN'S FIGURES

% 1. LEW006, SEPARATELY FOR EACH DATE
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.highRewardSide, ...
    'subset', ismember(data.animal, {'LEW006'}));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'RewardSide');
g.set_color_options('map', linspecer(2));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
custom_psychometric(g);
g.facet_wrap(data.datestr, 'ncols', 8);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.set_title('LEW006');
g.draw();
print(gcf, '-dpng', '~/Downloads/rewardShifted_1.png');

% 1. LEW006, across the good days
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.highRewardSide, ...
    'subset', ismember(data.animal, {'LEW006'}) & data.date > datetime('2018-04-27') & data.date < datetime('2018-05-10'));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'RewardSide');
g.set_color_options('map', linspecer(2));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
custom_psychometric(g);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.set_title('LEW006');
g.draw();
print(gcf, '-dpng', '~/Downloads/rewardShifted_2.png');

% 1. TRY FOR A CSHL MOUSE
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.highRewardSide, ...
    'subset', ismember(data.animal, {'Arthur'}));
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'RewardSide');
g.set_color_options('map', linspecer(2));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
custom_psychometric(g);
g.facet_wrap(data.datestr, 'ncols', 8);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.no_legend();
g.set_title('Arthur');
g.draw();
print(gcf, '-dpng', '~/Downloads/rewardShifted_3.png');


%% NOW THE SUMMARY
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
print(gcf, '-dpng', '~/Downloads/rewardShifted_all.png');

end
