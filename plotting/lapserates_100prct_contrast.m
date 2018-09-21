function data = lapserates_100prct_contrast()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');

if ispc,
    usr = getenv('USERNAME');
    homedir = getenv('USERPROFILE');
    datapath = fullfile(homedir, 'Google Drive');
elseif ismac,
    usr = getenv('USER');
    homedir = getenv('HOME');
    datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
end

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
has12prct  = @(x) any(ismember(abs(x), 0));
[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
remove100  = splitapply(has12prct, data_all.signedContrast, gr);
data       = data_all(ismember(data_all.dayidx, days(remove100)) & contains(data_all.animal, subjects(remove100)), :);

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

    axis tight; 
    l = refline(1,0); l.Color = 'k'; l.LineWidth = 0.5;
    l2 = lsline; l2.Color = [0.5 0.5 0.5];
    axis square; axisEqual; 
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
suplabel('100% contrast removed for all sessions including 0% contrast', 't')

foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-09-25');
if ~exist(foldername, 'dir'), mkdir(foldername); end
print(gcf, '-dpdf', fullfile(foldername, 'test_remove100prct_contrast_0prct.pdf'));

end