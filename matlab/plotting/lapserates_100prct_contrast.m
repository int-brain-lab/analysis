function data = lapserates_100prct_contrast()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath(genpath('/Users/urai/Documents/code/analysis_IBL'));
clear all; 

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
set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');

%% overview - all the choiceWorld mice
mice = {'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_2b', 'IBL_4b', 'IBL_5b', 'IBL_7b',  'IBL_9b', ...
    '6722', '6723', '6724', '6725', '6726', 'ALK081', 'LEW008', ...
    'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17', ...
    'IBL_1b', 'IBL_3b', 'IBL_6b', 'IBL_8b',  'IBL_10b', ...
    'LEW009', 'LEW010', ...
    '6812', '6814', '437', '438', ...
    'IBL_1b', 'IBL_3b', 'IBL_6b', 'IBL_8b',  'IBL_10b', ...
    'LEW009', 'LEW010', ...
    '6812', '6814', '437', '438'};

data_all       = readAlf_allData(datapath, mice);

% proposal: as soon as the 12.5% is introduced, remove 100% contrast
has12prct  = @(x) (numel(unique(abs(x(~isnan(x))))) == 6);
[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
remove100  = splitapply(has12prct, data_all.signedContrast, gr);
gridx = unique(gr);
data       = data_all(ismember(gr, gridx(remove100)), :);
data_all   = data;

% fit one psychometric function per day
[gr, days, subjects] = findgroups(data.dayidx, data.animal);
fitPsych = @(x,y) {fitErf(x, y>0)};
params   = splitapply(fitPsych, data.signedContrast, data.response, gr);
params   = cat(1, params{:});

% now fit without the 100% trials, compare
data(abs(data.signedContrast) == 100, :) = [];
[gr, days, subjects] = findgroups(data.dayidx, data.animal);
params_no100   = splitapply(fitPsych, data.signedContrast, data.response, gr);
params_no100   = cat(1, params_no100{:});

%% ===================== % 

figure;
colormap(linspecer(numel(unique(subjects)), 'qualitative'));
for p = 1:4,
    subplot(2,2,p); hold on;
    % plot(params(:, p), params_no100(:, p), '.');
    scatter(params(:, p), params_no100(:, p), 5, findgroups(subjects), 'filled');

    axis tight; axis square; axisEqual; 
    l = refline(1,0); l.Color = 'k'; l.LineWidth = 0.5;
    l2 = lsline; l2.Color = [0.5 0.5 0.5];
    offsetAxes;
    set(gca, 'xtick', get(gca, 'ytick'));

    switch p
        case 1
            title('Bias');
        case 2
            title('Threshold');
        case 3
            title('Gamma');
        case 4
            title('Lambda');
    end
    
end
suplabel('All contrast levels used', 'x');
suplabel('100% contrast removed', 'y');
suplabel('100% contrast removed for all sessions with 6 absolute contrast levels', 't')

foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-09-25', 'Test_remove100prctContrast');
if ~exist(foldername, 'dir'), mkdir(foldername); end
print(gcf, '-dpdf', fullfile(foldername, 'test_remove100prct_contrast_6levels.pdf'));

%% ========================== %
% ADD SOME EXAMPLES
% ========================== %

[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
erfFunc = @(x,p) p(3) + (1 - p(3) - p(4)) * (erf( (x-p(1))/p(2) ) + 1 )/2;
xval1 = -100:0.1:100;
xval2 = -50:0.1:50;

close all;
for g = unique(gr)',
   
    subplot(ceil(sqrt(length(unique(gr)))),ceil(sqrt(length(unique(gr)))),g); hold on;
    tmpdata = data_all(gr == g, :);
    errorbar(unique(tmpdata.signedContrast(~isnan(tmpdata.signedContrast))), ...
        splitapply(@nanmean, tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
        splitapply(@(x) (bootstrappedCI(x, 'mean', 'low')), tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
        splitapply(@(x) (bootstrappedCI(x, 'mean', 'high')), tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
        'color', 'k', 'capsize', 0, 'marker', '.', 'markerfacecolor', 'w', 'markersize', 2, 'linestyle', 'none');
    
    plot(xval1, erfFunc(xval1, params(g, :)), 'color', 'r');
    plot(xval2, erfFunc(xval2, params_no100(g, :)), 'color', 'b');
    axis off; xlim([-100 100]); ylim([0 1]); axis square;
end
print(gcf, '-dpdf', fullfile(foldername, 'test_remove100prct_contrast_levels_psychfuncs.pdf'));


end