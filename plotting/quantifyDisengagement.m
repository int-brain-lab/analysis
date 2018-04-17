function quantifyDisengagement(whichplot)

addpath(genpath('/Users/anne/Desktop/code/npy-matlab'));
set(groot, ...
    'DefaultFigureColormap', viridis, ...
    'DefaultFigureColor', 'w', ...
    'DefaultAxesColorOrder', viridis, ...
    'DefaultAxesLineWidth', 0.5, ...
    'DefaultAxesXColor', 'k', ...
    'DefaultAxesYColor', 'k', ...
    'DefaultAxesFontUnits', 'points', ...
    'DefaultAxesFontSize', 6, ...
    'DefaultAxesFontName', 'Helvetica', ...
    'DefaultLineLineWidth', 1, ...
    'DefaultTextFontUnits', 'Points', ...
    'DefaultTextFontSize', 6, ...
    'DefaultTextFontName', 'Helvetica', ...
    'DefaultAxesLabelFontSizeMultiplier', 1, ...
    'DefaultAxesBox', 'off', ...
    'DefaultAxesTickDir', 'out', ...
    'DefaultAxesTickLength', [0.02 0.05], ...
    'DefaultAxesXMinorTick', 'off', ...
    'DefaultAxesYMinorTick', 'off');

%%
%whichplot   = 1;
%%

paths       = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
nsubpl      = 5;
spcnt       = 1;
close all; figure(1);
allanimals_xvar = nan(20, 1000);
allanimals_yvar = nan(20, 1000);
sjcnt       = 0;

for p = 1:length(paths),
    mypath    = sprintf('/Users/anne/Google Drive/IBL_DATA_SHARE/%s/', paths{p});
    subjects  = nohiddendir(mypath);
    subjects  = {subjects.name};
    
    % remove some
    subjects(ismember(subjects, { 'Anne_mouse'   'Mouse1'   , ...
        'Mouse'    'Mouse3'    'Mouse4'    'Mouse999'  , ...
        'Tank'    'default'    'exampleSubject', 'saveAlf.m'})) = [];
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects)
        sjcnt = sjcnt + 1; cnt = 0;
        subplot(nsubpl, nsubpl, spcnt); hold on; spcnt = spcnt + 1;
        
        days = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        cmap = (viridis(numel(days)));
        thisanimal_xvar = nan(length(days)*5, 1000);
        thisanimal_yvar = nan(length(days)*5, 1000);
        
        % skip the first week
        for dayidx = 1:length(days),
            sessions = nohiddendir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
            for sessionidx = 1:length(sessions),
                
                %% READ DATA
                [timeOnTask, RT, reward] = readData(sprintf('%s/%s/cwResponse.times.npy', sessions(sessionidx).folder, sessions(sessionidx).name), ...
                    sprintf('%s/%s/cwReward.type.npy', sessions(sessionidx).folder, sessions(sessionidx).name));
                if p == 3 & (isempty(timeOnTask) | isempty(RT))
                    continue;
                elseif p == 3
                    cnt = cnt + 1;
                    reward = 2.5*ones(size(RT));
                elseif isempty(timeOnTask) | isempty(RT) | isempty(reward),
                    continue;
                else
                    cnt = cnt + 1;
                end
                
                if strcmp(subjects{sjidx}, 'Arthur') & dayidx == length(days),
                    assert(1==0)
                end
                
                %% GET OUT THE MEASURE OF INTEREST
                switch whichplot
                    case 1
                        
                        %% 1. reward rate
                        timeBins        = discretize(timeOnTask, [0:1:60]);
                        [gr, idx]       = findgroups(timeBins);
                        yvar            = splitapply(@nansum, reward, gr);
                        xvar            = splitapply(@nanmean, timeOnTask, gr);
                        
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
                
                %% plot the RT/mean(RT) ratio as a function of timeOnTask and cumulative reward
                plot(xvar, yvar, 'linewidth', 0.1, 'color', cmap(dayidx, :));
                
                %% SAVE FOR AVERAGES
                thisanimal_xvar(cnt, 1:length(xvar)) = xvar;
                thisanimal_yvar(cnt, 1:length(yvar)) = yvar;
            end
        end
        
        % layout
        axis tight; box off;
        
        % do a separate axis label with the cumulative RTratio???
        thisanimal_xvar_avg = (squeeze(nanmean(thisanimal_xvar)));
        thisanimal_yvar_avg = (squeeze(nanmean(thisanimal_yvar)));
        plot(thisanimal_xvar_avg, thisanimal_yvar_avg, 'k', 'linewidth', 1);
        
        allanimals_xvar(sjcnt, :) = thisanimal_xvar_avg;
        allanimals_yvar(sjcnt, :) = thisanimal_yvar_avg;
        
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

%% add GRAND AVERAGE ACROSS ANIMALS
subplot(nsubpl, nsubpl, spcnt);
boundedline(squeeze(nanmean(allanimals_xvar)), squeeze(nanmean(allanimals_yvar)), ...
    squeeze(nanstd(allanimals_yvar)), 'cmap', [0 0 0]);
xlabel('Time in session (min)');

%  LABELs
switch whichplot
    case 1
        lb = {'Reward rate (\muL / minute)'};
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

suplabel('Time in session (min)', 'x');
suplabel([lb{:}], 'y'); ylabel(lb);
% suplabel('Cumulative reward (\muL)', 'yy');

tightfig; 
switch whichplot
    case 1
        print(gcf, '-dpdf', sprintf('Disengagement_RewardRate.pdf'));
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function   [timeOnTask, RT, reward] = readData(file1, file2)

timeOnTask = [];
RT = [];
reward =[];
try
    dat         = readNPY(file1);
    timeOnTask  = dat(2:end) - dat(1);
    timeOnTask  = timeOnTask / 60;
    RT          = diff(dat); % remove the cumulative timeline
end
try
    dat2         = readNPY(file2);
    reward       = dat2(2:end);
end

if ~isempty(timeOnTask) & ~isempty(RT) & ~isempty(reward),
    if ~ ( (length(timeOnTask) == length(RT)) && (length(RT) == length(reward))),
        timeOnTask = [];
        RT = [];
        reward =[];
    end
end

end



%                 % do this in a sliding window fashion
%                 windowSize  = 3; % number of trials
%                 binIdx      = repmat(1:ceil(length(RTratio) ./ windowSize), windowSize, 1);
%                 binIdx      = binIdx(:);
%                 binIdx      = binIdx(1:length(RTratio));
%                 RTratio     = splitapply(@nanmean, RTratio, binIdx);
%                 timeOnTask  = splitapply(@nanmax, timeOnTask, binIdx);
%                 cum_rew     = splitapply(@nanmax, cum_rew, binIdx);
