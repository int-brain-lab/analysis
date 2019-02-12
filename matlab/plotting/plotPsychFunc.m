function plotPsychFunc(data, color)

if ~exist('color', 'var'), color = [0 0 0]; end
hold on;

% fit psychometric function
params = fitErf(data.signedContrast, (data.response > 0));
psychFuncPred = @(x, mu, sigma, gamma, lambda) gamma + (1 - gamma - lambda) * (erf( (x-mu)/sigma ) + 1 )/2;

y = psychFuncPred(linspace(min(data.signedContrast), max(data.signedContrast), 100), ...
    params(1), params(2), params(3), params(4));
plot(linspace(min(data.signedContrast), max(data.signedContrast), 100), y, '-', 'color', color);

% also compute binomial confidence intervals
[gr, idx]        = findgroups(data.signedContrast);
avg_stim         = splitapply(@nanmean, data.signedContrast, gr);

tmp             = splitapply(@binoCI, (data.response > 0), gr);
errorbar(avg_stim, tmp(:, 1), tmp(:, 2)-tmp(:, 1), tmp(:, 3)-tmp(:, 1), 'o', 'color', color, ...
    'MarkerSize', 5,'markerfacecolor', color, 'markeredgecolor', 'w', 'capsize', 0);

xlabel('Stimulus contrast (%)');
ylabel('Choose right (%)');
title({sprintf('%s', data.name{1}), ...
    sprintf('days %d-%d', min(data.dayidx), max(data.dayidx))}, 'interpreter', 'none');
box off; ylim([0 1]);
offsetAxes; axis square;
set(gca, 'ytick', [0 0.5 1], 'xtick', [-100:50:100]);

end
