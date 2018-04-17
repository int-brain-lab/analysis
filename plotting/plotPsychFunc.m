function plotPsychFunc()

addpath(genpath('/Users/anne/Desktop/code/npy-matlab'));
selpath = uigetdir();


stimLeft         = readNPY(sprintf('%s/cwStimOn.contrastLeft.npy', selpath));
stimRight        = readNPY(sprintf('%s/cwStimOn.contrastRight.npy', selpath));
stim             = sum([-stimLeft stimRight], 2);

resp             = readNPY(sprintf('%s/cwResponse.choice.npy', selpath));
resp(resp == 1)  = 0;
resp(resp == 2)  = 1;

% remove the last stimulus, end of session
if length(stim) > length(resp),
    stim = stim(1:length(resp));
end

[gr, idx]        = findgroups(stim);
p_right          = splitapply(@nanmean, resp, gr);
avg_stim         = splitapply(@nanmean, stim, gr);

% also compute binomial confidence intervals
tmp             = splitapply(@binoCI, resp, gr);
close all; subplot(221);
ploterr(avg_stim, tmp(:, 1), [], {tmp(:, 2) tmp(:, 3)}, '-ko', 'hhxy', 0)
hold on; plot(avg_stim, p_right, 'o', 'markerfacecolor', 'w', 'markeredgecolor', 'k');
xlabel('Signed contrast (%)');
ylabel('P(choose right)');

titleparts = strsplit(selpath, '/');
title(sprintf('%s, %s, session %s', titleparts{end-2}, titleparts{end-1}, titleparts{end}));
offsetAxes; box off;
tightfig;
print(gcf, '-dpdf', sprintf('%s/psychfunc.pdf', selpath));

end

function outp = binoCI(x)

[binomP, binoCI] = binofit(sum(x),numel(x));
outp = [[binomP, binoCI]];
end
