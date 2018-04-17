function quantifyDisengagement_perSession()
% Anne Urai, 17 April 2018

paths           = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
close all;
figure(1);
for p = 1:length(paths),
    
    mypath    = sprintf('/Users/anne/Google Drive/IBL_DATA_SHARE/%s/', paths{p});
    subjects  = nohiddendir(mypath);
    subjects  = {subjects.name};
    subjects(ismember(subjects, {'default', 'saveAlf.m'})) = [];
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects)
        days  = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        
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
                
                % REACTION TIME
                %plot(data.trialNum, data.rt, 'k-', 'linewidth', 0.1);
                %xlim([0 700]); ylim([0 60]);
                %set(gca, 'xtick', 0:300:600, 'xminortick', 'on', 'xticklabel', [], 'ytick', 0:20:60, 'yticklabel', [], 'ycolor', 'w'); % save space in the plot
                
                % ACCURACY ON EASIEST TRIALS IN A SLIDING WINDOW
                %yyaxis right
                
                % for bins of 10 trials, compute the average correct
                k = 10;
                %plot(movmean(data.trialNum(easytrials), k), movmean(data.correct(easytrials), k, 'omitnan'), '-', 'linewidth', 0.5);
                %ylim([0.5 1]);
                
                % layout
                %box off;
                % title(sprintf('%s, s%d', days(dayidx).name, sessionidx), 'fontsize', 1, 'fontweight', 'normal');
                %set(gca, 'xtick', 0:100:700, 'xticklabel', [], 'ytick', 0.5:0.1:1, 'yticklabel', [],  'ycolor', 'w'); % save space in the plot
                
                %% A PROPOSED CRITERION:
                % 1. compute median RT and % correct on high contrast for
                % first 300 trials
                minimumNrTrials = 200;
                data.correct_easy = double(data.correct);
                data.correct_easy((abs(data.signedContrast) < 40)) = NaN;
                
                first300_rt     = nanmedian(data.rt(data.trialNum < minimumNrTrials));
                first300_perf   = nanmean(data.correct_easy(data.trialNum < minimumNrTrials));
                
                % then, plot a running average of the percentage of that
                percentageChange = @(new, old) (new-old) ./ old;
                
                data.rt_track    = percentageChange(movmean(data.rt, k, 'Endpoints','shrink'), first300_rt);
                data.perf_track  = movmean(data.correct_easy, k, 'omitnan', 'Endpoints','shrink') - first300perf + 1;
                
                % plot this
                cla;
                plot(data.rt_track); hold on; 
                axis on; box off;
                spcnt = spcnt + 1;
                xlim([0 600]); ylim([0 50]);
                set(gca, 'xtick', 0:100:600, 'yticklabel', []);
                
                yyaxis rights
                plot(data.perf_track);
                xlim([0 600]);  ylim([0 1]);
                set(gca, 'xtick', 0:100:600);
                
                %axis off;
                set(gca, 'xticklabel', [], 'yticklabel', []);
                
                %% SET A RULE
                vline(minimumNrTrials, 'color', [0.5 0.5 0.5]);
                
                rtChange = 20; % in % of original median RT
                perfChange  = 0.8; % in fraction of original performance
                data.rt_track(1:minimumNrTrials)    = NaN;
                data.perf_track(1:minimumNrTrials)  = NaN;
                stopTheSession = min([find(data.rt_track > rtChange, 1, 'first') ...
                    find(data.perf_track < perfChange, 1, 'first')]);
                if ~isempty(stopTheSession)
                    vline(stopTheSession, 'color', 'g');
                end
            end
        end
        
        site = strsplit(paths{p}, '/');
        name = regexprep(subjects(sjidx), '_', ' ');
        suplabel([name{1} ' - ' site{1}], 't');
        suplabel('Trial [0-600 #]', 'x');
        suplabel('RT [0-60s]', 'y');
        suplabel('Accuracy on highest contrast [50-100%]', 'yy');
        print(gcf, '-dpdf', sprintf('%s/Disengagement_RT_%s_%s.pdf', '~/Data/IBL_data/Disengagement_figures', name{1}, site{1}));
    end
    
end
