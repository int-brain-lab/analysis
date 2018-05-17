function plotPsychFunc(foldername, color)

if ~exist('color', 'var'), color = [0 0 0]; end

% addpath(genpath('/Users/anne/Desktop/code/npy-matlab')); % user specific environment code should be kept out of the repository

% if no input was specified, prompt the user
if ~exist('foldername', 'var') || (exist('foldername', 'var') & isempty(foldername)),
    foldername = uigetdir('', 'Choose a session folder with Alf files');
end

if 
data             = readAlf(foldername);

if all(isnan(data.highRewardSide)),
    
    [gr, idx]        = findgroups(data.signedContrast);
    avg_stim         = splitapply(@nanmean, data.signedContrast, gr);
    
    % also compute binomial confidence intervals
    tmp             = splitapply(@binoCI, (data.response > 0), gr);
    
    % plot
    % close all; subplot(221);
    
    % this doesn't work on 8.6.0.267246 (R2015b)
    % errorbar(avg_stim, tmp(:, 1), tmp(:, 2)-tmp(:, 1), tmp(:, 3)-tmp(:, 1), 'k-o', ...
    %     'MarkerSize', 5,'markerfacecolor', 'w', 'markeredgecolor', 'k', 'capsize', 0, 'linewidth', 1);
    
    errorbar(avg_stim, tmp(:, 1), tmp(:, 2)-tmp(:, 1), tmp(:, 3)-tmp(:, 1), '-o', 'color', color, ...
        'MarkerSize', 5,'markerfacecolor', 'w', 'markeredgecolor', color, 'linewidth', 1);
    
else
    
    rewardSides = unique(data.highRewardSide);
    colors = linspecer(2);
    for h = 1:2,
        thisdata         = data(data.highRewardSide == rewardSides(h), :);
        [gr, idx]        = findgroups(thisdata.signedContrast);
        avg_stim         = splitapply(@nanmean, thisdata.signedContrast, gr);
        
        % also compute binomial confidence intervals
        tmp             = splitapply(@binoCI, (thisdata.response > 0), gr);
        
        % plot
        % close all; subplot(221);
        
        % this doesn't work on 8.6.0.267246 (R2015b)
        % errorbar(avg_stim, tmp(:, 1), tmp(:, 2)-tmp(:, 1), tmp(:, 3)-tmp(:, 1), 'k-o', ...
        %     'MarkerSize', 5,'markerfacecolor', 'w', 'markeredgecolor', 'k', 'capsize', 0, 'linewidth', 1);
        hold on;
        errorbar(avg_stim, tmp(:, 1), tmp(:, 2)-tmp(:, 1), tmp(:, 3)-tmp(:, 1), '-o', 'color', colors(h, :), ...
            'MarkerSize', 5,'markerfacecolor', 'w', 'markeredgecolor', colors(h, :), 'linewidth', 1);
    end
end

xlabel('Signed contrast (%)');
ylabel('P(choose right)');

titleparts = strsplit(foldername, '/');
title(sprintf('%s, %s, session %s', titleparts{end-2}, titleparts{end-1}, titleparts{end}));
box off;

try
    offsetAxes;
    tightfig;
end
print(gcf, '-dpdf', sprintf('%s/psychfunc.pdf', foldername));

end
