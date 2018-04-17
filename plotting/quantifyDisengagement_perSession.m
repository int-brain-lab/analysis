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
        cmap  = viridis(numel(days));
        
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
                plot(data.trialNum, data.rt, 'k-', 'linewidth', 0.1);
                xlim([0 700]); ylim([0 60]);
                set(gca, 'xtick', 0:100:700, 'xticklabel', [], 'ytick', 0:20:60, 'yticklabel', [], 'ycolor', 'w'); % save space in the plot
                
                % ACCURACY ON EASIEST TRIALS IN A SLIDING WINDOW
                yyaxis right
                
                % for bins of 10 trials, compute the average correct
                k = 10;
                easytrials = find((abs(data.signedContrast) > 40));
                plot(movmean(data.trialNum(easytrials), k), movmean(data.correct(easytrials), k, 'omitnan'), '-', 'linewidth', 0.5);
                ylim([0.5 1]);
                
                % layout
                box off;
                % title(sprintf('%s, s%d', days(dayidx).name, sessionidx), 'fontsize', 1, 'fontweight', 'normal');
                spcnt = spcnt + 1;
                set(gca, 'xtick', 0:100:700, 'xticklabel', [], 'ytick', 0.5:0.1:1, 'yticklabel', [],  'ycolor', 'w'); % save space in the plot
            end
        end
        
        site = strsplit(paths{p}, '/');
        name = regexprep(subjects(sjidx), '_', ' ');
        suplabel([name{1} ' - ' site{1}], 't');
        suplabel('Trial [0-700 #]', 'x');
        suplabel('RT [0-60s]', 'y');
        suplabel('Accuracy on highest contrast [50-100%]', 'yy');
        print(gcf, '-dpdf', sprintf('%s/Disengagement_RT_%s_%s.pdf', '~/Data/IBL_data/Disengagement_figures', name{1}, site{1}));
    end
    
end
