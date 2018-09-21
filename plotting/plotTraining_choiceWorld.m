function data = plotTraining_choiceWorld()
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

%% overview
batches(1).name = {'choiceWorld'};
batches(1).mice = {'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_2b', 'IBL_4b', 'IBL_5b', 'IBL_7b',  'IBL_9b', ...
    '6722', '6723', '6724', '6725', '6726', 'ALK081', 'LEW008', ...
    'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17'};

batches(end+1).name = {'choiceWorld_1screen'};
batches(end).mice = {'IBL_1b', 'IBL_3b', 'IBL_6b', 'IBL_8b',  'IBL_10b', ...
    'LEW009', 'LEW010', ...
    '6812', '6814', '437', '438'};

batches(end+1).name = {'choiceWorld_bigstim'};
batches(end).mice = {'IBL_11b', 'IBL_12b'};

% clear batches;
% batches(1).name = {'choiceWorld'};
% batches(1).mice = {'ALK081', 'LEW008',  'LEW009', 'LEW010'};

for bidx = length(batches):-1:1,
    for m = 1:length(batches(bidx).mice),
        
        close all;
        data_all = readAlf_allData(datapath, batches(bidx).mice{m});
        data_clean_all = data_all(data_all.inclTrials ~= 0, :);
        
        % first, 3x an overview of the performance over the last 3 days
        days = sort(unique(data_all.dayidx_rev));
        days = days(end-2:end);
        for didx = 1:length(days),
            
            % use only the data for this day
            data_clean  = data_clean_all(data_clean_all.dayidx_rev == days(didx), :);
            data        = data_all(data_all.dayidx_rev == days(didx), :);
            
            %% top: psychometric function
            subplot(4,4,didx);
            set(gca,'ColorOrder', [0.7 0.7 0.7; 0 0 0]); hold on;
            
            % right y-axis: chronometric function
            yyaxis left;
            errorbar(unique(data_clean.signedContrast(~isnan(data_clean.signedContrast))), ...
                splitapply(@nanmedian, data_clean.rt, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'median', 'low')), data_clean.rt, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'median', 'high')), data_clean.rt, findgroups(data_clean.signedContrast)), ...
                'capsize', 0, 'marker', 'o', 'markerfacecolor', 'w', 'markersize', 2);
            if didx == 1, ylabel('RT (s)'); end
            xlim([-105 105]);
            
            % add psychometric function
            yyaxis right;
            % add bootstrapped datapoints
            %psychFuncPred = @(x, mu, sigma, gamma, lambda) gamma+(1-gamma-lambda) * (1./(1+exp(- ( mu + sigma.*x ))));
            psychFuncPred = @(x, mu, sigma, gamma, lambda) gamma + (1 - gamma - lambda) * (erf( (x-mu)/sigma ) + 1 )/2;
            
            data_clean.probabilityLeft = roundn(data_clean.probabilityLeft, -2);
            leftProbs = unique(data_clean.probabilityLeft(~isnan(data_clean.probabilityLeft)));
            if isempty(leftProbs), data_clean.probabilityLeft(:) = 0.5; leftProbs = 0.5; end
            colors = linspecer(numel(leftProbs));
            for lp = 1:length(leftProbs),
                tmpdata = data_clean(data_clean.probabilityLeft == leftProbs(lp), :);
                try
                    errorbar(unique(tmpdata.signedContrast(~isnan(tmpdata.signedContrast))), ...
                        splitapply(@nanmean, tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
                        splitapply(@(x) (bootstrappedCI(x, 'mean', 'low')), tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
                        splitapply(@(x) (bootstrappedCI(x, 'mean', 'high')), tmpdata.response > 0, findgroups(tmpdata.signedContrast)), ...
                        'color', colors(lp, :), 'capsize', 0, 'marker', 'o', 'markerfacecolor', 'w', 'markersize', 2, 'linestyle', 'none');
                    
                end
                [mu, sigma, gamma, lambda] = fitErf(tmpdata.signedContrast, tmpdata.response > 0);
                
                y = psychFuncPred(linspace(min(tmpdata.signedContrast), max(tmpdata.signedContrast), 100), ...
                    mu, sigma, gamma, lambda);
                plot(linspace(min(tmpdata.signedContrast), max(tmpdata.signedContrast), 100), y, '-', 'color', colors(lp, :));
            end
            
            xlabel('Contrast (%)'); if didx == 3, ylabel('P(right)'); end
            box off;
            xlim([-105 105]); ylim([0 1]); % offsetAxes;
            set(gca, 'yminortick', 'on');
            
            % add date and psychometric function parameters
            [mu, sigma, gamma, lambda] = fitErf(data_clean.signedContrast, data_clean.response > 0);
            if abs(mu)<15 && sigma>15 && gamma<0.2 && lambda<0.2,
                try titlecol = cbrewer('seq', 'Greens', 6); titlecol = titlecol(end, :);
                catch; titlecol = [0 1 0]; end
                fontweigth = 'bold';
            else
                titlecol = [0 0 0];
                fontweigth = 'normal';
            end
            title({sprintf('%s, %d trials total', datestr(unique(data_clean.date)), numel(data.rt(~isnan(data.rt)))), ...
                sprintf('\\mu %.2f \\sigma %.2f \\gamma %.2f \\lambda %.2f', mu, sigma, gamma, lambda)}, 'fontweight', fontweigth, 'color', titlecol); % show date
            
            %% middle: RTs over time
            subplot(4,4,didx+4); hold on; colormap(linspecer(2));
            s3 = scatter(data.trialNum(data.inclTrials == 0), data.rt(data.inclTrials ==0), 3, '.k');
            s1 = scatter(data.trialNum(data.inclTrials == 1 & data.correct == 1), data.rt(data.inclTrials == 1 & data.correct == 1), 3, '.b');
            s2 = scatter(data.trialNum(data.inclTrials == 1 & data.correct == 0), data.rt(data.inclTrials == 1 & data.correct == 0), 3, '.r');
            
            xlabel('# trials');
            if didx == 1, ylabel('RT (s)'); end
            axis tight; xlim([-2 max(data.trialNum)]); ylim([-0.1 max(get(gca, 'ylim'))]);
            
            if didx == length(days),
                lh = legend([s1 s2 s3], {'correct', 'error', 'repeat'});
                lh.Box = 'off';
                lh.Position(1) = lh.Position(1) + 0.1;
            end
            
            %             %% bottom: performance within the session
            %             subplot(4,4,didx+8); hold on;
            %             data.stimOnTime = data.stimOnTime - data.stimOnTime(1);
            %
            %             % divide data into 1-minute segments
            %             minutes = discretize(data.stimOnTime, 0:30:max(data.stimOnTime));
            %             s1 = plot(splitapply(@mean, data.stimOnTime, findgroups(minutes)) / 60, ...
            %                 splitapply(@mean, 100*data.correct, findgroups(minutes)), 'r');
            %
            %             % again, but without repeated trials
            %             data.correct(data.inclTrials == 0) = NaN;
            %             s2 = plot(splitapply(@mean, data.stimOnTime, findgroups(minutes)) / 60, ...
            %                 splitapply(@nanmean, 100*data.correct, findgroups(minutes)), 'k');
            %
            %             axis tight;
            %             hline(50);
            %             hline(75);
            %             xlabel('Time (minutes)'); if didx == 1, ylabel('Performance (%)'); end
            %             ylim([0 100]); xlim([0 max(get(gca, 'xlim'))]); set(gca, 'ytick', [0 25 50 75 100]);
            %
            %             if didx == 3,
            %                 lh = legend([s1 s2], {'all trials', 'repeats removed'});
            %                 lh.Box = 'off';
            %                 lh.Position(1) = lh.Position(1) + 0.2;
            %             end
        end
        
        %% end: learning curve from the start
        subplot(4,4,[9 10]);
        useTrls = (abs(data_all.signedContrast) > 50 & data_all.inclTrials == 1);
        errorbar(unique(data_all.dayidx), splitapply(@nanmean, 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            splitapply(@(x) (bootstrappedCI(x, 'mean', 'low')), 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            splitapply(@(x) (bootstrappedCI(x, 'mean', 'high')), 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            'capsize', 0, 'color', 'k');
        ylabel({'Performance (%)' 'on >50% contrast' 'repeat trials excluded'});
        set(gca, 'xtick', unique(data_all.dayidx));
        % try offsetAxes; end
        box off; ylim([0 100]); xlim([0 max(data_all.dayidx)+1]);
        hline(50); hline(75);
        
        % fit psychometric function over days
        fitPsych = @(x,y) {fitErf(x, y>0)};
        params   = splitapply(fitPsych, data_clean_all.signedContrast, data_clean_all.response, findgroups(data_clean_all.dayidx));
        params   = cat(1, params{:});
        
        
        subplot(4,5,[16 17]);
        plot(unique(data_all.dayidx), params(:, 1));
        set(gca, 'xtick', unique(data_all.dayidx));
        ylabel('Bias', 'color', 'r'); ylim([-100 100]);
        r = refline(0, 15); r.LineStyle = ':';
        r = refline(0, -15); r.LineStyle = ':';
        
        yyaxis right;
        plot(unique(data_all.dayidx), params(:, 2));
        set(gca, 'xtick', unique(data_all.dayidx));
        ylabel('Threshold'); ylim([0 100]);
        r = refline(0, 15); r.LineStyle = '--';
        box off;
        xlabel('Days');
        
        subplot(4,5,[19 20]);
        plot(unique(data_all.dayidx), params(:, 3));
        set(gca, 'xtick', unique(data_all.dayidx));
        ylim([0 1]);
        r = refline(0, 0.2); r.LineStyle = ':';
        
        ylabel('Lapse (low)');
        yyaxis right;
        plot(unique(data_all.dayidx), params(:, 4));
        set(gca, 'xtick', unique(data_all.dayidx));
        r = refline(0, 0.2); r.LineStyle = ':';
        ylim([0 1]);
        ylabel('Lapse (high)');
        box off;
        xlabel('Days');
        
        %% save
        titlestr = sprintf('Lab %s, task %s, mouse %s', data.Properties.UserData.lab, ...
            batches(bidx).name{1}, batches(bidx).mice{m});
        try suplabel(titlestr, 'x'); end
        
        foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-09-25');
        if ~exist(foldername, 'dir'), mkdir(foldername); end
        print(gcf, '-dpdf', fullfile(foldername, sprintf('%s_%s_%s_%s.pdf', datestr(now, 'yyyy-mm-dd'), ...
            data.Properties.UserData.lab, batches(bidx).name{1}, batches(bidx).mice{m})));
        
    end
end
end


function y = bootstrappedCI(x, fun, bound)

fun = str2func(fun);
ci = bootci(2000,fun,x);
switch bound
    case 'low'
        y = fun(x) - ci(1);
    case 'high'
        y = ci(2) - fun(x);
end

end

function y = dpr(stim, resp, what)

[d, crit] = dprime(stim, resp);
switch what
    case 'dprime'
        y = d;
    case 'criterion'
        y = crit;
end
end
