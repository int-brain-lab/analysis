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
            mice = {'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17','IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37','IBL_1b','IBL_2b','IBL_3b', 'IBL_4b','IBL_5b','IBL_6b','IBL_7b','IBL_8b','IBL_9b','IBL_10b','IBL_11b','IBL_12b'};
        case 'IBL_Master'
            mice = { 'IBL_18', 'IBL_19', 'IBL_20',...
        'IBL_21', 'IBL_22', 'IBL_23', 'IBL_24', 'IBL_25', 'IBL_26','IBL_27','IBL_28','IBL_29','IBL_30','IBL_31','IBL_32'  'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', 'IBL_38', 'IBL_39','IBL_40','IBL_41','IBL_42', 'IBL_1b','IBL_2b','IBL_3b', 'IBL_4b'...
         'IBL_5b','IBL_6b','IBL_7b','IBL_8b','IBL_9b','IBL_10b','IBL_11b','IBL_12b'};% 'IBL_1','IBL_2','IBL_3', 'IBL_4', ...
               % 'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10',
        otherwise
            mice = {'CK_1', 'CK_2', 'CK_4', 'IBL_1','IBL_2','IBL_3', 'IBL_4', ...
                'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10', 'IBL_11', 'IBL_12', ...
                'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17', ...
                'IBL_18', 'IBL_19', 'IBL_20',...
            'IBL_21', 'IBL_22', 'IBL_23', 'IBL_24', 'IBL_25', 'IBL_26', 'IBL_27','IBL_28','IBL_29','IBL_30','IBL_31','IBL_32'
            'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', 'IBL_38', 'IBL_39','IBL_40','IBL_41','IBL_42'...
             'IBL_1b','IBL_2b','IBL_3b', 'IBL_4b'...
             'IBL_5b','IBL_6b','IBL_7b','IBL_8b','IBL_9b','IBL_10b','IBL_11b','IBL_12b'};
          
     
    end
end

set(groot, 'defaultaxesfontsize', 5, 'DefaultFigureWindowStyle', 'normal');
close all; figure; colormap(flipud(jet));
for m = 1:length(mice),

    clf;
    alldata = readAlf_allData(datapath, mice{m});
    if isempty(alldata), continue; end
    cnt = 1;

    for s = unique(alldata.dayidx)',

        data = alldata(alldata.dayidx == s, :);
        days = length(unique(alldata.dayidx));
        dims = ceil(sqrt(days));
        subplot(dims,dims,cnt); cnt = cnt + 1;
        hold on;

        stim = data.signedContrast / 80;
        scatter(1:height(data), stim, 10, [0 0 0], '.');

        % give random blocks a different color
        if any(~isnan(data.blockType))
            stim(data.blockType < 2) = NaN;
            scatter(1:height(data), stim, 10, [0.8 0.8 0.8], '.');
        end

        scatter(1:height(data), data.response, data.rt, data.correct, '.');
        
        % FOR MOVING MICE TO THE NEXT DAY, COUNT ONLY THE RTS WHERE THEY WERE ENGAGED
        correct = data.correct * 100;
        correct(data.rt > prctile(data.rt, 90)) = NaN; % discard RTs that were in the slowest 10 percent
        correct(data.response == 0) = NaN;
        
        % ADD AVERAGE RESPONSE OVER A SLIDING WINDOW OF 5 TRIALS
        M = movmean(correct/100,5);
        plot(1:height(data), M, 'color', [0.5 0.5 0.5], 'linewidth', 0.5);

        axis tight; ylim([-1.5 1.5]);
        try; offsetAxes; end

        if any(~isnan(data.blockType))
            if all(data.blockType(~isnan(data.blockType)) == 2),
                title({sprintf('%s', datestr(unique(data.date))), ...
                    sprintf('R %d%%', ...
                    round(nanmean(correct(data.blockType == 2))))});
            else
                title({sprintf('%s', datestr(unique(data.date))), ...
                    sprintf('+%d%% -%d%%, R %d%%', ...
                    round(nanmean(correct(data.signedContrast > 0 & data.blockType < 2))), ...
                    round(nanmean(correct(data.signedContrast < 0 & data.blockType < 2))), ...
                    round(nanmean(correct(data.blockType == 2))))});
            end

        else
            title({sprintf('%s', datestr(unique(data.date))), ...
                sprintf('+%d%% -%d%%, all %d%%, %d trials', round(nanmean(correct(data.signedContrast > 0))), ...
                round(nanmean(correct(data.signedContrast < 0))), ...
                round(nanmean(correct)), length(correct(~isnan(correct))))});
        end

        if s == 1,
            ylabel('Stimulus / Response');
            xlabel('Trial #');
        end
    end

    % suplabel(regexprep(mice{m}, '_', ' '), 't');
    print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', ...
        sprintf('shaping_stimPC_%s.pdf', mice{m})));

end
end
