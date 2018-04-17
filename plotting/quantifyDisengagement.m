function quantifyDisengagement(whichplot)
% Anne Urai, 17 April 2018

paths           = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
nsubpl          = 3;
spcnt           = 1;
sjcnt           = 0;
close all;
figure(1);

for whichplot = 1:10,
    for p = 1; %:length(paths),
        mypath    = sprintf('/Users/anne/Google Drive/IBL_DATA_SHARE/%s/', paths{p});
        subjects  = nohiddendir(mypath);
        subjects  = {subjects.name};
        
        % remove some
        subjects(ismember(subjects, {'default', 'saveAlf.m'})) = [];
        
        %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
        for sjidx = 1:length(subjects)
            sjcnt = sjcnt + 1; cnt = 0;
            subplot(nsubpl, nsubpl, spcnt); hold on; spcnt = spcnt + 1;
            
            days = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
            cmap = viridis(numel(days));
            
            % skip the first week
            for dayidx = 1:length(days),
                sessions = nohiddendir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
                for sessionidx = 1:length(sessions),
                    
                    %% READ DATA
                    data = readAlf(sprintf('%s/%s', sessions(sessionidx).folder, sessions(sessionidx).name));
                    if isempty(data), continue; end
                    
                    %% GET OUT THE MEASURE OF INTEREST
                    switch whichplot
                        case 1
                            %% RT
                            plot(data.trialNum, data.rt, '-', 'linewidth', 0.1, 'color', cmap(dayidx, :));
                        case 2
                            %% 2. reward rate, normalized by running average
                            timeBins        = discretize(timeOnTask, [0:1:60]);
                            [gr, idx]       = findgroups(timeBins);
                            rew_hour        = splitapply(@nansum, reward, gr);
                            xvar            = splitapply(@nanmean, timeOnTask, gr);
                            
                            % compute a running average
                            runningAvg      = cumsum(rew_hour) ./ transpose(1:length(rew_hour));
                            yvar            = rew_hour ./ runningAvg;
                            
                        case 3
                            %%  3. trials per minute
                            timeBins        = discretize(timeOnTask, [0:1:60]);
                            [gr, idx]       = findgroups(timeBins);
                            yvar            = splitapply(@numel, reward, gr);
                            xvar            = splitapply(@nanmean, timeOnTask, gr);
                            
                        case 4
                            %%  4. trials per minute
                            timeBins        = discretize(timeOnTask, [0:1:60]);
                            [gr, idx]       = findgroups(timeBins);
                            yvar            = splitapply(@numel, reward, gr);
                            xvar            = splitapply(@nanmean, timeOnTask, gr);
                            
                            % compute a running average
                            runningAvg      = cumsum(yvar) ./ transpose(1:length(yvar));
                            yvar            = yvar ./ runningAvg;
                            
                        case 5
                            %% 5. RT histograms
                            RTbins          = discretize(RT, 0:0.01:120);
                            [gr, idx]       = findgroups(RTbins);
                            yvar            = splitapply(@numel, RT, gr);
                            xvar            = splitapply(@nanmean, RT, gr);
                            
                        case 6
                            timeBins        = discretize(timeOnTask, [0:1:60]);
                            [gr, idx]       = findgroups(timeBins);
                            yvar            = splitapply(@nanmean, RT, gr);
                            xvar            = splitapply(@nanmean, timeOnTask, gr);
                            
                    end
                    
                end
            end
            
            % layout
            axis tight; box off;
            site = strsplit(paths{p}, '/');
            name = regexprep(subjects(sjidx), '_', ' ');
            title([name{1} ' - ' site{1}]);
        end
    end
    
    %% ADD A COLORBAR FOR SESSION NUMBER
    subplot(5,5,25);
    c = colorbar;
    c.Location = 'NorthOutside';
    axis off;
    prettyColorbar('Session #');
    c.Ticks = [0 1];
    c.TickLabels = {'early', 'late'};
    
    %  LABELs
    switch whichplot
        case 1
            lb = {'RT (second)'};
        case 2
            lb = {'Reward rate (\muL / minute)' 'current / running average'};
        case 3
            lb = {'Number of trials / minute'};
        case 4
            lb = {'Number of trials / minute' 'current / running average'};
        case 5
            lb = {'Time between choices (s)'};
        case 6
            lb = {'Time between choices (s)'};
    end
    
    suplabel('Trial #', 'x');
    suplabel([lb{:}], 'y'); 
    
    tightfig;
    switch whichplot
        case 1
            print(gcf, '-dpdf', sprintf('Disengagement_RT.pdf'));
        case 2
            print(gcf, '-dpdf', sprintf('Disengagement_RewardRateRatio.pdf'));
        case 3
            print(gcf, '-dpdf', sprintf('Disengagement_TrialRate.pdf'));
        case 4
            print(gcf, '-dpdf', sprintf('Disengagement_TrialRateRatio.pdf'));
        case 5
            suplabel('# of trials', 'y');
            suplabel('Time between choices (s)', 'x');
            print(gcf, '-dpdf', sprintf('Disengagement_ReactionTimeHistogram.pdf'));
        case 6
            print(gcf, '-dpdf', sprintf('Disengagement_ReactionTime.pdf'));
            
    end
    print(gcf, '-dpng', sprintf('Disengagement_%d.png', whichplot));
end
end