function weightMonitoring

% grab the latest data from Google Drive
weightFile = '~/Google Drive/Rig building WG/Data/weightMonitoringCSHL.xls';
[~,sheetsnames] = xlsfinfo(weightFile);

for a = 1:length(sheetsnames),
    tab = readtable(weightFile, 'sheet', a);
    
    % PLOT
    close all; hold on;
    yyaxis left;
    stem(tab.times, tab.water, 'filled', 'markersize', 2, 'linewidth', 3, 'linestyle', '-',  'marker', '.');
    stem(tab.times, tab.hydrogel,  'filled','color', [0.5 0.5 0.5], 'markersize', 2, 'linewidth', 3, 'linestyle', '-',  'marker', '.');
    ylabel('Water intake (microlitre)');
    
    yyaxis right
    plot([min(tab.times) max(tab.times)+1], [tab.weight(1)*0.85 tab.weight(1)*0.85],  'color', [0.5 0.5 0.5]);
    plot([min(tab.times) max(tab.times)+1], [tab.weight(1)*0.80 tab.weight(1)*0.80],  'color', [0.5 0.5 0.5]);
    plot(tab.times, tab.weight, 'ko-', 'markerfacecolor', 'k', 'markeredgecolor', 'k');
        ylabel('Weight (gram)');

    xtick = (datetime(min(tab.times)):datetime(datestr(today, 'yyyy-mm-dd')));
    [~, xticklb] = weekday(xtick);
    set(gca, 'xtick', xtick, 'xticklabel', xticklb, 'xticklabelrotation', -30);
    xlim([min(tab.times)-1 max(tab.times)+1]);
    set(gca, 'ycolor', 'k');
    
    title(sheetsnames{a});
    print(gcf, '-dpng', sprintf('~/Google Drive/Rig building WG/Data/weightMonitoring_%s.png', sheetsnames{a}));

end

end