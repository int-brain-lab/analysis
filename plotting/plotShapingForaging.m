function plotShapingForaging(mice)

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
            mice = {'CK_1', 'CK_2', 'CK_4', 'IBL_1','IBL_2','IBL_3', 'IBL_4', ...
                'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10', 'IBL_11', 'IBL_12'};
    end
end

set(groot, 'defaultaxesfontsize', 5);
close all; figure;
for m = 1:length(mice),
    
    close all;
    alldata = readAlf_allData(datapath, mice{m});
    cnt = 1;
    
    for s = unique(alldata.dayidx)',
        
        data = alldata(alldata.dayidx == s, :);
        subplot(4,6,cnt); cnt = cnt + 1;
        scatter(1:height(data), data.signedContrast / 80, 10, [0 0 0], '.');
        
        % give random blocks a different color
        stim = data.signedContrast / 80;
        stim(data.blockType < 0) = NaN;
        scatter(1:height(data), stim, 10, [0.8 0.8 0.8], '.');
        
        hold on;
        try colormap(flipud(linspecer(2))); end
        scatter(1:height(data), data.response, data.rt, data.correct, '.');
        
        % ADD AVERAGE RESPONSE OVER A SLIDING WINDOW OF 5 TRIALS
        M = movmean(data.response,5);
        plot(1:height(data), M, 'color', [0.5 0.5 0.5], 'linewidth', 0.5);
        
        axis tight; ylim([-2 2]);
        try; offsetAxes; end
        
        % FOR MOVING MICE TO THE NEXT DAY, COUNT ONLY THE RTS WHERE THEY
        % WERE ENGAGED
        correct = data.correct * 100;
        correct(data.rt > prctile(data.rt, 90)) = NaN; % discard RTs that were in the slowest 10 percent
        
        if any(~isnan(data.blockType)),
            title({sprintf('%s', datestr(unique(data.date))), ...
                sprintf('+%d%% -%d%%, R%d%%', ...
                round(nanmean(correct(data.signedContrast > 0 & data.blockType < 2))), ...
                round(nanmean(correct(data.signedContrast < 0 & data.blockType < 2))), ...
                round(nanmean(correct(data.blockType == 2)))});
        else
            title({sprintf('%s', datestr(unique(data.date))), ...
                sprintf('+%d%% -%d%%', round(nanmean(correct(data.signedContrast > 0))), ...
                round(nanmean(correct(data.signedContrast < 0))))});
        end
        
        if s == 1,
            ylabel('Stimulus / Response');
            xlabel('Trial #');
        end
    end
    
    % suplabel(regexprep(mice{m}, '_', ' '), 't');
    print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', sprintf('shaping_stimPC_%s.pdf', mice{m})));
    
end
end