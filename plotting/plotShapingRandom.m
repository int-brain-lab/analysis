function plotShapingRandom(mice)

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
addpath(genpath('\\NEW-9SE8HAULSQE\Users\IBL_Master\Documents\IBLData_Shared\code\tmp_analysis_matlab'));
addpath(genpath('C:\Users\Tony\Documents\Github\tmp_analysis_matlab'));
addpath(genpath('/Users/urai/Documents/code/tmp_analysis_matlab'));
addpath(genpath('/Users/urai/Documents/code/npy-matlab'));
assert(exist('readNPY', 'file') > 0, ...
    'readNPY must be on your Matlab path, get it at github.com/kwikteam/npy-matlab');

% GOOGLE DRIVE IS ALWAYS UNDER HOME
if ispc,
    usr = getenv('USERNAME');
    homedir = getenv('USERPROFILE');
    datapath = fullfile(homedir, 'Google Drive');
elseif ismac,
    usr = getenv('USER');
    homedir = getenv('HOME');
    datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
end

if ~exist('mice', 'var'),
    switch usr
        case 'Tony'
            mice = {'CK_1', 'CK_2', 'CK_4','IBL_11', 'IBL_12'};
        case 'IBL_Master'
            mice = { 'IBL_1','IBL_2','IBL_3', 'IBL_4', ...
                'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10'};
        otherwise
            mice = {'CK_1', 'CK_4', 'IBL_1','IBL_2','IBL_3', 'IBL_4', ...
                'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10'};
    end
end

set(groot, 'defaultaxesfontsize', 6, ...
    'DefaultAxesTickLength', [0.01 0.0125], 'DefaultFigureWindowStyle', 'normal');
close all; figure; colormap(flipud(jet));
clf;

% ============================================================== %
% BLOCKED VS. RANDOM PERFORMANCE OVER TIME
% ============================================================== %

for m = 1:length(mice),
    
    data = readAlf_allData(datapath, mice{m});
    data(isnan(data.stimOnTime), :) = [];
    subplot(length(mice), 2, (m*2)-1); hold on;
    plot([1 height(data)], [50 50], ':k');
    
    % discard RTs that were in the slowest 10 percent
    correct = data.correct * 100;
    correct(data.rt > prctile(data.rt, 90)) = NaN;
    
    correct_blocked = correct;
    correct_blocked(data.blockType == 2) = NaN;
    correct_random = correct;
    correct_random(data.blockType ~= 2) = NaN;
    
    % compute sliding window
    slidingWindowSize = 200;
    correct_blocked = movmean(correct_blocked, slidingWindowSize, ...
        'omitnan', 'endpoints', 'fill');
    correct_random = movmean(correct_random, slidingWindowSize, ...
        'omitnan', 'endpoints', 'fill');

    % plot random separately
    plot(correct_blocked, 'b-'); 
    plot(correct_random, 'r-');
    set(gca, 'xticklabel', [], 'xtick', 1:500:height(data));
    axis tight; ylim([35 100]);
    title(regexprep(mice{m}, '_', ' '));
    
end
xlabel('# Trials'); ylabel('Accuracy (%)');
try; tightfig; end
print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', ...
    sprintf('shaping_stimPC_random_allMice.pdf')));
assert(1==0);

% ============================================================== %
% STRATEGY PLOT
% ============================================================== %

close; b = nan(length(mice), 4);
for m = 1:length(mice),
    
    data = readAlf_allData(datapath, mice{m});
    data(isnan(data.stimOnTime), :) = [];
    
    % only random blocks
    data = data(data.blockType == 2, :);
    
    if ~isempty(data),
        % FIT BUSSE-STYLE LOGISTIC REGRESSION MODEL
        prevresp            = circshift(sign(data.response), 1);
        prevcorrect         = circshift(data.correct, 1);
        prevresp_success    = prevresp;
        prevresp_failure    = prevresp;
        prevresp_success(prevcorrect == 0) = 0;
        prevresp_failure(prevcorrect == 1) = 0;
        
        % do the fit
        designM = [data.signedContrast / 100, prevresp_success, prevresp_failure];
        b(m, :) = glmfit(designM, (data.response > 0), 'binomial');
    end
    
end

% NOW PLOT
subplot(221); plotBetasSwarm(b);
set(gca, 'xtick', 1:4, 'xticklabel', ...
    {'Side bias', '\beta_{stimulus}', '\beta_{success}', '\beta_{failure}'});
ylabel('Logistic regression weights');

subplot(222);
plot(b(:, 3), b(:, 4), 'o');
set(gca, 'xlim', [-3 3], 'ylim', [-3 3]);
hline(0); vline(0);
xlabel('\beta_{success}', 'interpreter', 'tex'); ylabel('\beta_{failure}');
title('Random blocks');

try; tightfig; end
print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', ...
    sprintf('shaping_strategies.pdf')));

end
