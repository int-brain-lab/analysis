function plotShapingForaging

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
assert(exist('readNPY', 'file') > 0, ...
    'readNPY must be on your Matlab path, get it at github.com/kwikteam/npy-matlab');

%datapath = 'C:\Users\IBL_Master\Google Drive';
mice = {'CK_1', 'CK_2', 'CK_4', 'IBL_1','IBL_2','IBL_3', 'IBL_4' ...
    'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10',}
set(groot, 'defaultaxesfontsize', 6);
close all; figure;

for m = 1:length(mice),
    
    close all;
    alldata = readAlf_allData([], mice{m});
    cnt = 1;
    
    for s = unique(alldata.dayidx)',
        
        data = alldata(alldata.dayidx == s, :);
        subplot(4,4,cnt); cnt = cnt + 1;
        scatter(1:height(data), data.signedContrast / 80, 10, [0 0 0]);
        hold on;
        try colormap(flipud(linspecer(2))); end
        scatter(1:height(data), data.response, data.rt, data.correct);

        % ADD AVERAGE RESPONSE OVER A SLIDING WINDOW OF 5 TRIALS
        M = movmean(data.response,5);
        plot(1:height(data), M, 'color', [0.5 0.5 0.5]);
        
        axis tight; ylim([-2 2]);
        % try; offsetAxes; end 
        title({datestr(unique(data.date)) ...
            sprintf('+%d%%, -%d%%', round(100*nanmean(data.correct(data.signedContrast > 0))), ...
            round(100*nanmean(data.correct(data.signedContrast < 0))))}, ...
            'fontsize', 3);
        % add lines to indicate the different sessions
    end
    
    %% indicate the specific task that this mouse trained on
    switch mice{m}
        case {'IBL_1', 'IBL_2', 'IBL_5', 'IBL_8', 'IBL_9'}
            task = 'Lateralized stimuli';
        case {'IBL_3', 'IBL_4', 'IBL_6', 'IBL_7', 'IBL_10'}
            task = 'Central stimuli';
        otherwise
            task = [];
    end
    
    suplabel('Stimulus / Response', 'y');
    suplabel('Trial #', 'x');
    suplabel([regexprep(mice{m}, '_', ' '), sprintf(', %s', task)], 't');
   
    print(gcf, '-dpdf', sprintf('/Users/anne/Google Drive/IBL_DATA_SHARE/CSHL/figures/shaping_%s.pdf', mice{m}));

end

end