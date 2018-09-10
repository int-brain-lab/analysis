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
batches(1).name = {'centralOrientation_Glickfeld'};
batches(1).mice = {'IBL_28', 'IBL_29', 'IBL_30', 'IBL_31', 'IBL_32'};

batches(end+1).name = {'choiceWorld'};
batches(end).mice = {'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_2b', 'IBL_4b', 'IBL_5b', 'IBL_7b',  'IBL_9b'};

batches(end+1).name = {'choiceWorld_orientation'};
batches(end).mice = {'IBL_38', 'IBL_39', 'IBL_40', 'IBL_41', 'IBL_42'};

batches(end+1).name = {'choiceWorld_1screen'};
batches(end).mice = {'IBL_1b', 'IBL_3b', 'IBL_6b', 'IBL_8b',  'IBL_10b'};

batches(end+1).name = {'choiceWorld_bigstim'};
batches(end).mice = {'IBL_11b', 'IBL_12b'};

batches(end+1).name = {'choiceWorld'};
batches(end).mice = {'6722', '6723', '6724', '6725', '6726'};

batches(end+1).name = {'choiceWorld'};
batches(end).mice = {'ALK081', 'LEW008', 'LEW009', 'LEW010'};

for bidx = length(batches):-1:length(batches)-1,
    for m = 1:length(batches(bidx).mice),
        
        close all;
        set(gcf,'defaultAxesColorOrder',[0 0 0; 0.7 0.7 0.7]);
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
            try
            errorbar(unique(data_clean.signedContrast(~isnan(data_clean.signedContrast))), ...
                splitapply(@nanmean, data_clean.response > 0, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'mean', 'low')), data_clean.response > 0, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'mean', 'high')), data_clean.response > 0, findgroups(data_clean.signedContrast)), ...
                'capsize', 0, 'marker', 'o', 'markerfacecolor', 'w', 'markersize', 3);
            catch
               assert(1==0);
            end
            xlabel('Contrast (%)'); if didx == 1, ylabel('P(right)'); end
            box off;
            xlim([-105 105]); ylim([0 1]); % offsetAxes;
            
            % right y-axis: chronometric function
            yyaxis right;
            errorbar(unique(data_clean.signedContrast(~isnan(data_clean.signedContrast))), ...
                splitapply(@nanmedian, data_clean.rt, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'median', 'low')), data_clean.rt, findgroups(data_clean.signedContrast)), ...
                splitapply(@(x) (bootstrappedCI(x, 'median', 'high')), data_clean.rt, findgroups(data_clean.signedContrast)), ...
                'capsize', 0, 'marker', 'o', 'markerfacecolor', 'w', 'markersize', 2);
            if didx == 3, ylabel('RT (s)'); end
            xlim([-105 105]);
            title(datestr(unique(data_clean.date))); % show date
            
            %% middle: RTs over time
            subplot(4,4,didx+4); hold on; colormap(linspecer(2));
            s1 = scatter(data.trialNum(data.inclTrials == 1 & data.correct == 1), data.rt(data.inclTrials == 1 & data.correct == 1), 5, 'ob');
            s2 = scatter(data.trialNum(data.inclTrials == 1 & data.correct == 0), data.rt(data.inclTrials == 1 & data.correct == 0), 5, 'or');
            s3 = scatter(data.trialNum(data.inclTrials == 0), data.rt(data.inclTrials ==0), 5, 'dk');
            
            xlabel('# trials');
            if didx == 1, ylabel('RT (s)'); end
            axis tight; xlim([-2 max(data.trialNum)]); ylim([-0.1 max(get(gca, 'ylim'))]);
            
            if didx == length(days),
                lh = legend([s1 s2 s3], {'correct', 'error', 'repeat'});
                lh.Box = 'off';
                lh.Position(1) = lh.Position(1) + 0.1;
            end
            
            %% bottom: performance within the session
            subplot(4,4,didx+8); hold on;
            data.stimOnTime = data.stimOnTime - data.stimOnTime(1);
            
            % divide data into 1-minute segments
            minutes = discretize(data.stimOnTime, 0:30:max(data.stimOnTime));
            s1 = plot(splitapply(@mean, data.stimOnTime, findgroups(minutes)) / 60, ...
                splitapply(@mean, 100*data.correct, findgroups(minutes)), 'r');
            
            % again, but without repeated trials
            data.correct(data.inclTrials == 0) = NaN;
            s2 = plot(splitapply(@mean, data.stimOnTime, findgroups(minutes)) / 60, ...
                splitapply(@nanmean, 100*data.correct, findgroups(minutes)), 'k');
            
            axis tight;
            hline(50);
            hline(75);
            xlabel('Time (minutes)'); if didx == 1, ylabel('Performance (%)'); end
            ylim([0 100]); xlim([0 max(get(gca, 'xlim'))]); set(gca, 'ytick', [0 25 50 75 100]);
            
            if didx == 3,
                lh = legend([s1 s2], {'all trials', 'repeats removed'});
                lh.Box = 'off';
                lh.Position(1) = lh.Position(1) + 0.2;
            end
        end
        
        %% end: learning curve from the start
        subplot(4,4,[13 14]);
        useTrls = (abs(data_all.signedContrast) > 50 & data_all.inclTrials == 1);
        errorbar(unique(data_all.dayidx), splitapply(@nanmean, 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            splitapply(@(x) (bootstrappedCI(x, 'mean', 'low')), 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            splitapply(@(x) (bootstrappedCI(x, 'mean', 'high')), 100*data_all.correct(useTrls), findgroups(data_all.dayidx(useTrls))), ...
            'capsize', 0, 'color', 'k');
        hline(50);
        xlabel('Days'); ylabel({'Performance (%)' 'on >50% contrast' 'repeat trials excluded'});
        set(gca, 'xtick', unique(data_all.dayidx), 'xticklabel', datestr(unique(data_all.date)), 'xticklabelrotation', -30);
        % try offsetAxes; end
        box off; ylim([0 100]); xlim([0 max(data_all.dayidx)]);
        
        subplot(4,4,[15 16]);
        plot(unique(data_all.dayidx), splitapply(@(x,y) (dpr(x,y, 'dprime')), ...
            sign(data_all.signedContrast(useTrls)), data_all.response(useTrls), findgroups(data_all.dayidx(useTrls))));
        set(gca, 'xtick', unique(data_all.dayidx), 'xticklabel', datestr(unique(data_all.date)), 'xticklabelrotation', -30);
        xlim([0 max(data_all.dayidx)+1]); ylabel('D''');
        
        yyaxis right;
        plot(unique(data_all.dayidx), splitapply(@(x,y) (dpr(x,y, 'criterion')), ...
            sign(data_all.signedContrast(useTrls)), data_all.response(useTrls), findgroups(data_all.dayidx(useTrls))));
          set(gca, 'xtick', unique(data_all.dayidx), 'xticklabel', datestr(unique(data_all.date)), 'xticklabelrotation', -30);
        xlim([0 max(data_all.dayidx)+1]); ylabel('Criterion');
        
        %% save
        titlestr = sprintf('Lab %s, task %s, mouse %s', data.Properties.UserData.lab, ...
            batches(bidx).name{1}, batches(bidx).mice{m});
        try suplabel(titlestr, 't'); end
        
        foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-09-11');
        if ~exist(foldername, 'dir'), mkdir(foldername); end
        print(gcf, '-dpdf', fullfile(foldername, sprintf('%s_%s_%s_%s.pdf', datestr(now, 'yyyy-mm-dd'), ...
            batches(bidx).name{1}, data.Properties.UserData.lab, batches(bidx).mice{m})));
        
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



