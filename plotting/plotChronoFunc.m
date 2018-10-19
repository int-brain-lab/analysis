function plotChronoFunc(data, color)

if ~exist('color', 'var'), color = [0.5 0.5 0.5]; end
set(gca, 'ycolor', color);
hold on;

% also compute binomial confidence intervals
[gr, idx]        = findgroups(data.signedContrast);
rt               = splitapply(@nanmedian, data.rt, gr);
stim               = splitapply(@nanmedian, data.signedContrast, gr);

errorbar(stim, rt, splitapply(@(x) bootstrappedCI(x, 'nanmedian', 'low'), data.rt, gr), ...
    splitapply(@(x) bootstrappedCI(x, 'nanmedian', 'high'), data.rt, gr),...
    '-', 'color', color, ...
    'MarkerSize', 1, 'marker', 'o', 'markerfacecolor', color, 'markeredgecolor', color, 'capsize', 0);

xlabel('Stimulus contrast (%)');
ylabel('RT (s)');
title({sprintf('%s', data.name{1}), ...
    sprintf('days %d-%d', min(data.dayidx), max(data.dayidx))}, 'interpreter', 'none');
box off; 
axisNotSoTight; axis square;
ylim([0 1.2]);
set(gca, 'xtick', [-100:50:100], 'ytick', 0:0.6:1.2);

end
