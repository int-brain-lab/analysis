

% grab all the data that's on Drive
addpath('~/Documents/code/npy-matlab//');
addpath(genpath('/Users/urai/Documents/code/analysis_IBL'));
addpath('~/Documents/code/gramm/');
figfolder = '~/Google Drive/Rig building WG/Posters/SfN2018/Panels';

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

%% OVERALL PSYCHOMETRIC

mice       = {'IBL_34', 'LEW010', '438'};
for m = 1:length(mice),
    
    data   = readAlf_allData(datapath, mice{m});
    
    % subselect some data
    close all  ; subplot(331);
    %if ~contains(data.name, 'CCU'),
        yyaxis right;
        plotChronoFunc(data(data.dayidx >= 20 & data.dayidx <= 25, :));
    %end
    yyaxis left;
    plotPsychFunc(data(data.dayidx >= 20 & data.dayidx <= 25, :));
    set(gca, 'ycolor', 'k');
    tightfig;
    print(gcf, '-dpdf', sprintf('%s/psychometrics_%s.pdf', figfolder, mice{m}));
    
    if numel(unique(data.probabilityLeft(~isnan(data.probabilityLeft)))) > 1,
        
        colors = linspecer(2);
        close all;
        subplot(331);
        plotPsychFunc(data(data.probabilityLeft < 0.5, :), colors(1, :));
        plotPsychFunc(data(data.probabilityLeft > 0.5, :), colors(2, :));
        
        tightfig;
        print(gcf, '-dpdf', sprintf('%s/psychometrics_biased_%s.pdf', figfolder, mice{m}));
        
    end
end

%% LEARNING CURVES

% overview
mice{1} = {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_1', 'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10', ...
    'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17'}
mice{2} = {'LEW009', 'LEW010', 'ALK081', 'LEW008'};
mice{3} = {'6812', '6814', '437', '438'};

% PLOT
cols = cbrewer('qual', 'Set2', 8);
colors = cols([1 3 5], :);
close all;
subplot(311); hold on;

for l = length(mice):-1:1,
    data_all   = readAlf_allData(datapath, mice{l});
    data_all.dayidx = data_all.dayidx - min(data_all.dayidx) + 1; % make sure the 1st day where there is data (not an empty folder) is dayidx 1
    
    for m = unique(data_all.animal)',
        data_all.dayidx(contains(data_all.animal, m)) = data_all.dayidx_rev(contains(data_all.animal, m)) ...
            - min(data_all.dayidx_rev(contains(data_all.animal, m))) + 1;
    end
    
    %  SHOW LEARNING CURVES - % CORRECT ON >50% CONTRASTS
    data_all.response(abs(data_all.signedContrast) < 50) = NaN;
    data_all.correct(abs(data_all.signedContrast) < 50) = NaN;
    [gr, sj, days] = findgroups(data_all.animal, data_all.dayidx);
    tab = array2table(days, 'variablenames', {'days'});
    tab.animal = sj;
    tab.performance = splitapply(@nanmean, 100 * data_all.correct, gr);
    %  tab.lab = splitapply(@unique, data_all.lab, gr);
    mat = unstack(tab, 'performance', 'days');
    mat = mat{:, 2:end};
    
    % PLOT
    p{l} = plot(mat', '-', 'color', colors(l, :));
    allmats{l} = padarray(mat', 45-size(mat, 2), NaN, 'post');
    
end

l = legend([p{3}(1) p{2}(1) p{1}(1)], {sprintf('Lisbon (CCU), %d animals', numel(mice{3})), ...
    sprintf('London (UCL), %d animals', numel(mice{2})), ...
    sprintf('NY (CSHL), %d animals', numel(mice{1}))}, ...
    'Location', 'southeast', 'AutoUpdate','off');
l.Box = 'off';

avg = nanmean(cat(2, allmats{:}), 2);
hold on;
plot(avg, 'k', 'linewidth', 2);
xlabel('Training days'); ylabel({'Performance (%)' 'on easy trials'});

axis tight; hline(50);
ylim([0 100]); offsetAxes;
tightfig;
print(gcf, '-dpdf', sprintf('%s/learningcurves.pdf', figfolder));
