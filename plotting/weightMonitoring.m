function weightMonitoring(weightFile)

if ~exist('weightFile', 'var'),
    % make an informed guess
    if exist('~/Google Drive/Rig building WG/Data/weightMonitoringCSHL.xls', 'file'),
        weightFile = '~/Google Drive/Rig building WG/Data/weightMonitoringCSHL.xls';
    else
        weightFile = uigetfile('*.xls', 'Where is the weight monitoring Excel file?');
    end
end

% grab the latest data from Google Drive
[~,sheetsnames] = xlsfinfo(weightFile);

for a = 1:length(sheetsnames),
    tab = readtable(weightFile, 'sheet', a);
    tab.Properties.VariableNames = cellfun(@lower, tab.Properties.VariableNames, 'un', 0);
    
    % PLOT
    close all;
    subplot(2,2,[1 2]);
    hold on;
    yyaxis left;
    try
    stem(tab.times, tab.water, 'filled', 'markersize', 5, 'linewidth', 2, 'linestyle', '-',  'marker', '.');
    stem(tab.times, tab.hydrogel,  'filled','color', [0.5 0.5 0.5], 'markersize', 2, 'linewidth', 3, 'linestyle', '-',  'marker', '.');
    catch
        assert(1==0);
    end
    ylabel('Water intake (microlitre)');
    
    yyaxis right
    plot([min(tab.times)-1 max(tab.times)+1], [tab.weight(1)*0.85 tab.weight(1)*0.85],  'color', [0.5 0.5 0.5]);
    plot([min(tab.times)-1 max(tab.times)+1], [tab.weight(1)*0.80 tab.weight(1)*0.80],  'color', [0.5 0.5 0.5]);
    plot(tab.times, tab.weight, 'ko-', 'markerfacecolor', 'k', 'markeredgecolor', 'k');
        ylabel('Weight (gram)');

    xtick = (datetime(min(tab.times)):datetime(max(tab.times)));
    [~, xticklb] = weekday(xtick);
    set(gca, 'xtick', xtick, 'xticklabel', xticklb, 'xticklabelrotation', -30);
    xlim([min(tab.times)-1 max(tab.times)+1]);
    set(gca, 'ycolor', 'k');
    
    title(sprintf('%s, %s', sheetsnames{a}, datestr(today)));
    figurefile = sprintf('%s/weightMonitoring_%s.png', fileparts(weightFile), sheetsnames{a});
    print(gcf, '-dpng', figurefile);
    fprintf('Saving %s \n', figurefile);
    
end

end