function quantifyDisengagement_perSession()
% Anne Urai, 17 April 2018

imagepath       = '~/Data/IBL_data/Disengagement_figures';
datapath        = '/Users/anne/Google Drive/IBL_DATA_SHARE';
paths           = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
close all;
figure(1);

% SET SOME PARAMETERS
k               = 10; % SLIDING WINDOW SIze
minimumNrTrials = 200; % minimum number of trials to be completed
rtChange        = 150; % in % of original median RT
perfChange      = 0.8; % in fraction of original performance
totalChange     = 50;
summarytab      = struct('lab', [], 'animal', [], 'date', [], 'session', [], ...
    'totalTrials', [], 'taskTime', [], 'totalReward', []); idx = 1;
plotEachSubject  = 1;

for p = 1:length(paths),
    
    mypath    = sprintf('%s/%s/', datapath, paths{p});
    subjects  = nohiddendir(mypath);
    subjects  = {subjects.name};
    subjects(ismember(subjects, {'default', 'saveAlf.m'})) = [];
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects)
        days  = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        
        site = strsplit(paths{p}, '/');
        name = regexprep(subjects(sjidx), '_', ' ');
        
        % make subplots
        close all;
        nsubpl = ceil(sqrt(length(days))) + 1;
        spcnt  = 1;
        
        % skip the first week
        for dayidx = 8:length(days), % skip the first week!
            
            sessions = nohiddendir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
            for sessionidx = 1:length(sessions),
                try
                    subplot(nsubpl, nsubpl, spcnt); hold on;
                catch
                    axis off;
                    continue;
                end
                
                %% READ DATA
                data = readAlf(sprintf('%s/%s', sessions(sessionidx).folder, sessions(sessionidx).name));
                if isempty(data) || height(data) < 20,
                    axis off;
                    continue;
                end
                
                if height(data) < minimumNrTrials,
                    continue;
                end
                
                % REACTION TIME
                %plot(data.trialNum, data.rt, 'k-', 'linewidth', 0.1);
                %xlim([0 700]); ylim([0 60]);
                %set(gca, 'xtick', 0:300:600, 'xminortick', 'on', 'xticklabel', [], 'ytick', 0:20:60, 'yticklabel', [], 'ycolor', 'w'); % save space in the plot
                
                % ACCURACY ON EASIEST TRIALS IN A SLIDING WINDOW
                %yyaxis right
                
                % for bins of 10 trials, compute the average correct
                %plot(movmean(data.trialNum(easytrials), k), movmean(data.correct(easytrials), k, 'omitnan'), '-', 'linewidth', 0.5);
                %ylim([0.5 1]);
                
                % layout
                %box off;
                % title(sprintf('%s, s%d', days(dayidx).name, sessionidx), 'fontsize', 1, 'fontweight', 'normal');
                %set(gca, 'xtick', 0:100:700, 'xticklabel', [], 'ytick', 0.5:0.1:1, 'yticklabel', [],  'ycolor', 'w'); % save space in the plot
                
                %% A PROPOSED CRITERION:
                % 1. compute median RT and % correct on high contrast for
                % first 300 trials
                data.correct_easy = double(data.correct);
                data.correct_easy((abs(data.signedContrast) < 80)) = NaN;
                
                first300_rt     = nanmedian(data.rt(data.trialNum < minimumNrTrials));
                first300_rt_std = nanstd(data.rt(data.trialNum < minimumNrTrials));
                
                %  first300_perf   = nanmean(data.errors_easy(data.trialNum < minimumNrTrials));
                
                % then, plot a running average of the percentage of that
                ratio = @(new, old) 100 * (new - old) ./ old ;
                
                % to track RT, take the moving mean
                data.rt_track    = movmedian(data.rt, k, 'Endpoints','shrink') ./ first300_rt;
                %  data.perf_track  = movmean(data.errors_easy, k, 'omitnan', 'Endpoints','shrink') * 100; % - first300_perf;
                
                data.inverse_efficiency = data.rt ./ movmean(data.correct_easy, k*3, 'omitnan', 'Endpoints','shrink');
                
                first300_ie   = nanmean(data.inverse_efficiency(data.trialNum < minimumNrTrials))
                
                %                 data.total_track = data.rt_track .* data.perf_track;
                %                 assert(~all(isnan(data.total_track)));
                %                 data.total_track = movmean(data.rt, k, 'Endpoints','shrink')
                
                data.total_track = (data.inverse_efficiency ./ first300_ie) ./ first300_ie;
                
                if plotEachSubject,
                    % plot this
                    cla;
                    plot(data.rt, 'linewidth', 0.2); hold on;
                    axis on; box off;
                    spcnt = spcnt + 1;
                    ylim([0 50]);
                    xlim([0 800]); ylim([0 60]);
                    set(gca, 'xtick', 0:100:800, 'yticklabel', []);
                    
                    yyaxis right
                    plot(data.rt_track, 'linewidth', 0.2);
                    xlim([0 800]);  ylim([0 30]);
                    set(gca, 'xtick', 0:100:800);
                    set(gca, 'xticklabel', [], 'ytick', 0:10:50, 'yticklabel', []);
                    vline(minimumNrTrials, 'color', [0.5 0.5 0.5]);
                end
                
                %% SET A RULE
                data.rt_track(1:minimumNrTrials)    = NaN;
                %data.perf_track(1:minimumNrTrials)  = NaN;
                
                %                 % check for either RT or performance drop
                %                 stopTheSession = min([find(data.rt_track > rtChange, 1, 'first') ...
                %                     find(data.perf_track < perfChange, 1, 'first')]);
                %
                %                 % require both
                %                 stopTheSession = find(data.rt_track > rtChange & ...
                %                     data.perf_track < perfChange, 1, 'first');
                %
                %                 data.total_track(1:minimumNrTrials) = NaN;
                %                 stopTheSession = find(data.total_track > totalChange, 1, 'first');
                %
                stopTheSession = find(data.rt_track > rtChange, 1, 'first');
                stopTheSession = find(data.rt_track > 10, 1, 'first');
                
                if plotEachSubject,
                    if ~isempty(stopTheSession)
                        plot(stopTheSession, max(get(gca, 'ylim')), 'g*');
                        % vline(stopTheSession, 'color', 'g');
                    else
                        %                         % require only one
                        %                         stopTheSession = max([find(data.rt_track > rtChange, 1, 'first') ...
                        %                             find(data.perf_track < perfChange, 1, 'first')]);
                        %                         if ~isempty(stopTheSession),
                        %                             plot(stopTheSession, 1, 'g*');
                        %                         end
                        % assert(1==0)
                    end
                end
                
                %% TRACK THE TOTAL NUMBER OF TRIALS, TIME AND REWARD BASED ON THIS STOPPING RULE
                if ~isempty(stopTheSession)
                    summarytab(idx).lab          = site{1};
                    summarytab(idx).animal       = name{1};
                    summarytab(idx).date         = datetime(days(dayidx).name);
                    summarytab(idx).session      = sessionidx;
                    summarytab(idx).totalTrials  = stopTheSession;
                    summarytab(idx).actualTrials = height(data);
                    summarytab(idx).actualReward = nansum(data.rewardVolume);
                    summarytab(idx).taskTime     = data.stimOnTime(stopTheSession) ./ 60; % in minutes
                    summarytab(idx).totalReward  = nansum(data.rewardVolume(1:stopTheSession));
                    idx                          = length(summarytab)+1;
                end
            end
        end
        
        if plotEachSubject,
            suplabel([name{1} ' - ' site{1}], 't');
            suplabel('Trial [0-600 #]', 'x');
            suplabel('Inverse efficiency, ratio with baseline', 'yy');
            suplabel('RT (s)', 'y');
            print(gcf, '-dpdf', sprintf('%s/Disengagement_RT_%s_%s.pdf', imagepath, name{1}, site{1}));
        end
    end
end

%% OVERVIEW

summarytab       = struct2table(summarytab);
summarytab(1, :) = [];
% summarytab.totalReward(summarytab.totalReward == 0) = NaN;
% summarytab.totalTrials(isempty(summarytab.totalTrials)) = NaN;
% summarytab.taskTime(isempty(summarytab.taskTime)) = NaN;

gr = findgroups(summarytab.animal);
nsubpl = ceil(sqrt(length(unique(gr))));

close all;
for a = unique(gr)',
    subplot(nsubpl, nsubpl, a);
    scatter(summarytab.actualReward(gr == a), summarytab.totalReward(gr == a), ...
        10, datenum(summarytab.date(gr == a)));
    title(sprintf('%s, %s', summarytab.animal{find(gr == a, 1)}, ...
        summarytab.lab{find(gr == a, 1)}));
    axis square; axis tight; offsetAxes;
    set(gca, 'xlim', get(gca, 'ylim'), 'xtick', get(gca, 'ytick')); % equal axi
    
end
suplabel('Received total water reward (\muL)', 'x');
suplabel('Proposed total water reward (\muL)', 'y');
suplabel(sprintf('After %d trials, if RT over %d trials increases %d from initial median and performance drops below %.2f of initial average', minimumNrTrials, k, rtChange, perfChange));

end

