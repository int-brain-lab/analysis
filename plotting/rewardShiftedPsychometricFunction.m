function rewardShiftedPsychometricFunction


% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
data = readAlf_allData();
addpath('~/Desktop/code/gramm/');

% select only those animals where there is a highrewardside indicated
data(isnan(data.highRewardSide), :) = [];

%% use gramm for easy dataviz
close all;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.highRewardSide);
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)', 'color', 'High reward');
g.set_continuous_color('active', false);  
g.stat_summary('type', 'std', 'geom', 'line'); % no errorbars within a session
g.stat_summary('type', 'bootci', 'geom', 'errorbar'); % no errorbars within a session
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 9);
g.draw();

print(gcf, '-dpdf', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_rewardshifted.pdf');
print(gcf, '-dpng', '/Users/anne/Google Drive/Rig building WG/Data/psychfuncs_rewardshifted.png');


% g.facet_wrap(data.name, 'ncols', 3);
