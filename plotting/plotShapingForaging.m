function plotShapingForaging

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
addpath('~/Desktop/code/gramm/');

mice = {'CK_1', 'CK_2', 'CK_4', 'IBL_1','IBL_2','IBL_3', ...
    'IBL_5','IBL_6','IBL_7','IBL_8','IBL_9','IBL_10',};
close all; figure;

for m = 1:length(mice),
    
    close all;
    alldata = readAlf_allData(mice{m});
    cnt = 1;
    
    for s = unique(alldata.dayidx)',
        
        data = alldata(alldata.dayidx == s, :);
        subplot(4,4,cnt); cnt = cnt + 1;
        scatter(1:height(data), data.signedContrast / 80, 10, [0 0 0]);
        hold on;
        colormap(flipud(linspecer(2)));
        scatter(1:height(data), data.response, data.rt, data.correct);

        % ADD AVERAGE RESPONSE OVER A SLIDING WINDOW OF 5 TRIALS
        M = movmean(data.response,5);
        plot(1:height(data), M, 'color', [0.5 0.5 0.5]);
        
        axis tight; ylim([-2 2]);
        offsetAxes;
        title(datestr(unique(data.date)), 'fontsize', 3);
        % add lines to indicate the different sessions
    end
    
    suplabel('Stimulus / Response', 'y');
    suplabel('Trial #', 'x');
    suplabel(regexprep(mice{m}, '_', ' '), 't');
    print(gcf, '-dpdf', sprintf('/Users/anne/Google Drive/Rig building WG/Data/thresholdRampResults_%s.pdf', mice{m}));


end


end