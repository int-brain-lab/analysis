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
mice = {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_1', 'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10', ...
    'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17', ...
    'LEW009', 'LEW010', 'ALK081', 'LEW008'}

data_all       = readAlf_allData(datapath, mice);

% % proposal: as soon as the 12.5% is introduced, remove 100% contrast
isbiased   = @(x) (numel(unique(x(~isnan(x)))) > 1);
[gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
biased     = splitapply(isbiased, data_all.probabilityLeft, gr);
gridx = unique(gr);
data       = data_all(ismember(gr, gridx(biased)), :);
data_all = data;

%% correct
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

foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-10-09');
if ~exist(foldername, 'dir'), mkdir(foldername); end
print(gcf, '-dpdf', fullfile(foldername, 'psychfuncparams_biased.pdf'));

%% ================= %
% propose some criterion
% ================= %

close;
subplot(221);
plotBetasSwarm([params_low(:, 1), params_high(:, 1)]);
ylabel('Bias');
set(gca, 'xticklabel', {'0.2', '0.8'});

subplot(222);
histogram([params_low(:, 1), params_high(:, 1)], 100, 'edgecolor', 'none');
xlabel('\DeltaBias');
box off; 

print(gcf, '-dpdf', fullfile(foldername, 'psychfuncparams_biased_criterion.pdf'));

% 
% [gr, days, subjects] = findgroups(data_all.dayidx, data_all.animal);
% erfFunc = @(x,p) p(3) + (1 - p(3) - p(4)) * (erf( (x-p(1))/p(2) ) + 1 )/2;
% xval1 = -100:0.1:100;
% 
% close all;
% for g = unique(gr)',
%     subplot(5,5,g); hold on;
%     plot(xval1, erfFunc(xval1, params_low(g, :)), 'color', 'b');
%     plot(xval1, erfFunc(xval1, params_high(g, :)), 'color', 'r');
%    xlim([-100 100]); ylim([0 1]); axis square;
%    set(gca, 'xticklabel', [], 'yticklabel', []);
% end
% print(gcf, '-dpdf', fullfile(foldername, 'psychfuncparams_biased_psychfuncs.pdf'));

%% ================================== % 
% PLOT FOR SFN
% ================================== % 

addpath('~/Documents/code/gramm/');

data        = data_all;
data.probabilityLeft(data.probabilityLeft < 0.5) = 0.2;
data.probabilityLeft(data.probabilityLeft > 0.5) = 0.8;

% fit psychometric
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) (g + (1 - g - l) * (erf( (x-a)/b ) + 1 )/2),...
'StartPoint', [0 20 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);
axis_square =  @(gramm_obj) gramm_obj.axe_property('PlotBoxAspectRatio', [1 1 1]);

% plot
close all; clear g;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', data.probabilityLeft);
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.facet_wrap(data.animal, 'ncols', 3);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
rdBu = cbrewer('qual', 'Set1', 3);
g.set_color_options('map', rdBu([2 1], :)); % red and blue as schematic
g.no_legend();
g.geom_vline('extent', 20, 'style', '--k');
custom_psychometric(g);

% general layout
g.axe_property('ylim', [0 1], 'xlim', [-110 100], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
%axis_square(g);
g.draw();
g.export('file_name', '~/Google Drive/Rig building WG/Posters/SfN2018/biasedPsychometrics.pdf', 'file_type', 'pdf');
g.export('file_name', '~/Google Drive/Rig building WG/Results/DataFigures/biasedPsychometrics.pdf', 'file_type', 'pdf');


%% ================================== % 
% PLOT FOR ALEX POUGET: PSYCHOMETRIC FUNCTIONS 3 DAYS BEFORE BIAS WAS
% INTRODUCED (CHAT 12 OCT 2018)
% ================================== % 

mice = {'IBL_1', 'IBL_10', 'IBL_3', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'LEW009', 'LEW010'};
data       = readAlf_allData(datapath, mice);

% TAKE THE LAST 3 DAYS BEFORE BIASED BLOCKS ARE INTRODUCED
data = data(data.probabilityLeft == 0.5, :);
data.dayidx_rev2 = data.dayidx_rev;
for m = unique(data.animal)',
    data.dayidx_rev2(contains(data.animal, m)) = data.dayidx_rev(contains(data.animal, m)) ...
        - max(data.dayidx_rev(contains(data.animal, m)));
end
    
% plot
close all; clear g;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'subset', (data.dayidx_rev2 > -3));
g.set_names('x', 'Stimulus contrast (%)', 'y', 'Rightwards choices (%)');
g.facet_wrap(data.animal, 'ncols', 3);
g.stat_summary('type', 'bootci', 'geom', {'errorbar', 'point'});
g.set_color_options('map', [0 0 0]); % red and blue as schematic
g.no_legend();
%g.geom_vline('extent', 20, 'style', ':k');
custom_psychometric(g);

% general layout
g.axe_property('ylim', [0 1], 'xlim', [-110 101], 'ytick', [0 0.5 1],  'yticklabel', [0 50 100], 'xtick', [-100 -50 -20 0 20 50 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
%axis_square(g);
g.draw();
g.export('file_name', '~/Google Drive/Rig building WG/Results/DataFigures/psychometrics_3daysbeforePrior.pdf', 'file_type', 'pdf');


end