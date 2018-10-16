function lapses_single_vs_separate()
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

set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');
msz = 4;

%% overview
mice = ({'IBL_1', 'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10', ...
    'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17', ...
    'LEW009', 'LEW010', 'ALK081', 'LEW008', '6812', '6814', '437', '438',});

for m = 1:length(mice),
    
    close all;
    data_all = readAlf_allData(datapath, mice{m});
    data_all.dayidx = data_all.dayidx - min(data_all.dayidx) + 1; % make sure the 1st day where there is data (not an empty folder) is dayidx 1
    if isempty(data_all), continue; end
    data_clean_all = data_all(data_all.inclTrials ~= 0, :);
    
    % =============================================== %
    % DETERMINE WHETHER (AND WHEN) THIS MOUSE IS TRAINED
    % =============================================== %
    
    % Decision: For each session, performance at high contrast > 80%.
    % Min trial per session 200.
    % On fitted data (over 3 session): |bias| < 16%, threshold > 19%, lapse < 0.2.
    
    % for each day, test the 2 top criteria
    useTrls = (abs(data_all.signedContrast) > 50 & data_all.inclTrials == 1);
    accuracy_crit = splitapply(@nanmean, 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls)));
    accuracy_crit = (accuracy_crit > 0.8);
    
    ntrials = splitapply(@numel, data_all.rt, findgroups(data_all.dayidx));
    ntrials_crit = (ntrials > 200);
    
    % additional criterion: all contrasts must be present
    allcontrasts = @(x) (numel(unique(abs(x(~isnan(x))))) == 6);
    contrasts_crit = splitapply(allcontrasts, data_all.signedContrast, findgroups(data_all.dayidx));
    
    % fit psychometric function over days
    fitPsych = @(x,y) {fitErf(x, y>0)};
    fitPsych_singleLapse = @(x,y) {fitErf_singleLapse(x, y>0)};
    
    usedays = unique(data_all.dayidx);
    % usedays(usedays < 3) = [];
    psychfuncparams_twolapse = nan(numel(usedays), 4);
    psychfuncparams_singlelapse = nan(numel(usedays), 3);
    
    for d = usedays',
        if d >= 3,
            ThreeSessionTrls = (data_clean_all.dayidx >= d-2 & data_clean_all.dayidx <= d);
            psychfuncparams_twolapse(find(d==usedays), :) = ...
                fitErf(data_clean_all.signedContrast(ThreeSessionTrls), (data_clean_all.response(ThreeSessionTrls)>0) );
            psychfuncparams_singlelapse(find(d==usedays), :) = ...
                fitErf_singleLapse(data_clean_all.signedContrast(ThreeSessionTrls), (data_clean_all.response(ThreeSessionTrls)>0) );
            
        end
    end
    psychfuncparams_twolapse(isnan(sum(psychfuncparams_twolapse, 2)), :) = 0;
    psychfuncparams_singlelapse(isnan(sum(psychfuncparams_singlelapse, 2)), :) = 0;
    
    % test if the criteria are true
    psychfunc_crit = (abs(psychfuncparams_twolapse(:, 1)) < 16 & psychfuncparams_twolapse(:, 2) > 19 ...
        & psychfuncparams_twolapse(:, 3) < 0.2 & psychfuncparams_twolapse(:, 4) < 0.2);
    has_learned = (accuracy_crit & ntrials_crit & contrasts_crit & psychfunc_crit);
    if any(has_learned),
        istrained = true;
        day_trained = find(has_learned == 1, 1, 'first');
    else
        istrained = false;
    end
    
    % test if the criteria are true
    psychfunc_crit = (abs(psychfuncparams_singlelapse(:, 1)) < 16 & psychfuncparams_singlelapse(:, 2) > 19 ...
        & psychfuncparams_singlelapse(:, 3) < 0.2);
    has_learned_singlelapse = (accuracy_crit & ntrials_crit & contrasts_crit & psychfunc_crit);
    if any(has_learned_singlelapse),
        istrained_singlelapse = true;
        day_trained_singlelapse = find(has_learned_singlelapse == 1, 1, 'first');
    else
        istrained_singlelapse = false;
    end
    
    % =============================================== %
    % PSYCHOMETRIC FUNCTION OVER DAYS
    % TWO SEPARATE LAPSE TERMS
    % =============================================== %
    
    % fit psychometric function over days
    params   = splitapply(fitPsych, data_clean_all.signedContrast, data_clean_all.response, findgroups(data_clean_all.dayidx));
    params   = cat(1, params{:});
    
    subplot(9, 4,[1 2]); hold on;
    plot(unique(data_all.dayidx), params(:, 1), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx));
    ylabel('Bias'); ylim([-50 50]);
    hline(16); hline(-16);
    box off;  xlim([0 max(data_all.dayidx)]);
    set(gca, 'xcolor', 'w');
    if istrained, vline(day_trained); end
    
    switch istrained
        case 1
            trainedStr = sprintf('trained from day %d', day_trained);
        case 0
            trainedStr = 'not trained';
    end
    titlestr = sprintf('Lab %s, mouse %s, %s', data_all.Properties.UserData.lab, ...
        regexprep(mice{m}, '_', ''), trainedStr);
    title(titlestr);
    
    subplot(9, 4,[5 6]); hold on;
    plot(unique(data_all.dayidx), params(:, 2), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx));
    ylabel('Threshold'); ylim([0 100]);
    hline(19);
    box off;  xlim([0 max(data_all.dayidx)]);
    set(gca, 'xcolor', 'w');
    if istrained, vline(day_trained); end
    
    subplot(9, 4,[9 10]); hold on;
    plot(unique(data_all.dayidx), params(:, 3), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx));
    ylabel({'Lapse' '(low)'}); ylim([0 1]);
    hline(0.2);
    box off;  xlim([-0.05 max(data_all.dayidx)]);
    set(gca, 'xcolor', 'w');
    if istrained, vline(day_trained); end
    
    subplot(9, 4,[13 14]); hold on;
    plot(unique(data_all.dayidx), params(:, 4), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx));
    ylabel({'Lapse' '(high)'}); ylim([-0.05 1]);
    hline(0.2);
    box off;  xlim([0 max(data_all.dayidx)]);
    xlabel('Days');
    if istrained, vline(day_trained); end
    
    % =============================================== %
    % PSYCHOMETRIC FUNCTION OVER DAYS
    % single lapse term
    % =============================================== %
    
    % fit psychometric function over days
    params   = splitapply(fitPsych_singleLapse, data_clean_all.signedContrast, data_clean_all.response, findgroups(data_clean_all.dayidx));
    params   = cat(1, params{:});
    
    subplot(9, 4,[3 4]); hold on;
    plot(unique(data_all.dayidx), params(:, 1), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx), 'yaxislocation', 'right');
    ylabel('Bias'); ylim([-50 50]);
    hline(16); hline(-16);
    box off;  xlim([0 max(data_all.dayidx)]);
    set(gca, 'xcolor', 'w');
    if istrained, vline(day_trained_singlelapse); end
    
    switch istrained_singlelapse
        case 1
            trainedStr = sprintf('trained from day %d', day_trained_singlelapse);
        case 0
            trainedStr = 'not trained';
    end
    titlestr = sprintf('Lab %s, mouse %s, %s', data_all.Properties.UserData.lab, ...
        regexprep(mice{m}, '_', ''), trainedStr);
    title(titlestr);
    
    subplot(9, 4,[7 8]); hold on;
    plot(unique(data_all.dayidx), params(:, 2), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx), 'yaxislocation', 'right');
    ylabel('Threshold'); ylim([0 100]);
    hline(19);
    box off;  xlim([0 max(data_all.dayidx)]);
    set(gca, 'xcolor', 'w');
    if istrained, vline(day_trained_singlelapse); end
    
    subplot(9, 4,[11 12]); hold on;
    plot(unique(data_all.dayidx), params(:, 3), '-ko', 'markeredgecolor', 'w', 'markerfacecolor', 'k', 'markersize', msz);
    set(gca, 'xtick', unique(data_all.dayidx), 'yaxislocation', 'right');
    ylabel({'Lapse'}); ylim([0 1]);
    hline(0.2);
    box off;  xlim([-0.05 max(data_all.dayidx)]);
    if istrained, vline(day_trained_singlelapse); end
    xlabel('Days');
    
    % =============================================== %
    % SAVE
    % =============================================== %
    
    foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', ...
        'DataFigures', 'BehaviourData_Weekly', '2018-10-22');
    if ~exist(foldername, 'dir'), mkdir(foldername); end
    print(gcf, '-dpdf', fullfile(foldername, sprintf('%s_%s_%s_lapseTest.pdf', datestr(now, 'yyyy-mm-dd'), ...
        data_all.Properties.UserData.lab, mice{m})));
    
end
end

