function data = biasedBlocks()
% make overview plots across labs
% uses the gramm toolbox: https://github.com/piermorel/gramm
% Anne Urai, 2018

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath(genpath('/Users/urai/Documents/code/analysis_IBL'));

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
isbiased   = @(x) (numel(unique(x(~isnan(x)))) > 1);
[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
biased     = splitapply(isbiased, data_all.probabilityLeft, gr);
gridx = unique(gr);
data       = data_all(ismember(gr, gridx(biased)), :);
data_all = data;

% correct
data_low = data(data.probabilityLeft < 0.5, :);
data_high = data(data.probabilityLeft > 0.5, :);

% fit one psychometric function per day and per biased condition
[gr, days, subjects] = findgroups(data_low.dayidx, data_low.animal);
fitPsych = @(x,y) {fitErf(x, y>0)};

params_low   = splitapply(fitPsych, data_low.signedContrast, data_low.response, gr);
params_low   = cat(1, params_low{:});

[gr, days, subjects] = findgroups(data_high.dayidx, data_high.animal);
params_high   = splitapply(fitPsych, data_high.signedContrast, data_high.response, gr);
params_high   = cat(1, params_high{:});

%% ===================== % 

figure;
colormap(linspecer(numel(unique(subjects)), 'qualitative'));
for p = 1:4,
    subplot(2,2,p); hold on;
    scatter(params_low(:, p), params_high(:, p), 10, findgroups(subjects), 'filled');

    axis tight; 
    l = refline(1,0); l.Color = 'k'; l.LineWidth = 0.5;
    % l2 = lsline; l2.Color = [0.5 0.5 0.5];
    axis square; axisEqual; 
    offsetAxes;
    set(gca, 'xtick', get(gca, 'ytick'));
    
    ploterr(nanmean(params_low(:, p)), nanmean(params_high(:, p)), ...
        nanstd(params_low(:, p)) ./ sqrt(length(params_low)), ...
        nanstd(params_high(:, p)) ./ sqrt(length(params_high)), 'ko', 'abshhxy', 0);

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
suplabel('P(left) = 0.2', 'x');
suplabel('P(left) = 0.8', 'y');

foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-09-25');
if ~exist(foldername, 'dir'), mkdir(foldername); end
print(gcf, '-dpdf', fullfile(foldername, 'psychfuncparams_biased.pdf'));

% ================= %

[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
erfFunc = @(x,p) p(3) + (1 - p(3) - p(4)) * (erf( (x-p(1))/p(2) ) + 1 )/2;
xval1 = -100:0.1:100;

close all;
for g = unique(gr)',
    subplot(5,5,g); hold on;
    plot(xval1, erfFunc(xval1, params_low(g, :)), 'color', 'b');
    plot(xval1, erfFunc(xval1, params_high(g, :)), 'color', 'r');
   xlim([-100 100]); ylim([0 1]); axis square;
   set(gca, 'xticklabel', [], 'yticklabel', []);
end
print(gcf, '-dpdf', fullfile(foldername, 'psychfuncparams_biased_psychfuncs.pdf'));


end