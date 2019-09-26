function citricAcid_ntrials

one = One();

foldername = fullfile(getenv('HOME'), 'Google Drive', 'Rig building WG', ...
    'DataFigures', 'BehaviourData_Weekly', 'AlyxPlots');

clear tab;
tab.dates  = {'2018-09-24', '2018-10-01', '2018-10-09', '2018-10-15', '2018-10-22', '2018-10-29'}';
tab.weekend_water  = {'1ml/day', 'adlib CA 5% hydrogel', 'adlib CA 2% water', ...
    'adlib 2% CA water', 'adlib 2% CA water', 'adlib CA 5% hydrogel'}';
tab.rig_water = {'tap water', 'tap water', '15% sucrose', '15% sucrose', '15% sucrose', '10% sucrose'}';
tab.water = strcat( {'weekend: '}, tab.weekend_water,  {', rig: '}, tab.rig_water);
tab = struct2table(tab);  
% tab = tab(end-1:end, :);

trialcounts = nan(6,17);
for d = 1:length(tab.dates),
    [eid, ses] = one.search('lab', 'zadorlab', 'date_range', datenum({tab.dates{d}, tab.dates{d}})) ;
    for e = 1:length(eid),
        try
            D = one.load(eid{e}, 'data', '_ibl_trials.choice', 'dclass_output', true);
            disp(D.data);
            trialcounts(d,e) = length(D.data{1});
        end
    end
end

trialcounts(trialcounts == 0) = NaN;

close all;
violinPlot(trialcounts', 'addSpread', 1, 'showMM', 4);
ylim([0 2000]);
%plotBetasSwarm(trialcounts');
set(gca, 'xtick', 1:height(tab), 'xticklabel', tab.water, 'xticklabelrotation', -90);
ylabel('Trial counts on Monday');

%tightfig;
print(gcf, '-dpdf', fullfile(foldername, 'citricAcid_trialCounts_CSHL.pdf'));
print(gcf, '-dpdf', '/Users/urai/Google Drive/2018 Postdoc CSHL/CitricAcid/citricAcid_trialCounts_CSHL_alltrials.pdf');
print(gcf, '-dpdf', '~/Google Drive/Rig building WG/Posters/SfN2018/Panels/citricAcid_trialCounts_CSHL_alltrials.pdf');

% writetable(data2, '~/Google Drive/2018 Postdoc CSHL/CitricAcid/trialcounts.csv');
% 
% %data_tmp = data(:, {'trialNum', 'animal', 'date'});
% 
% %% BARGRAPH WITH SCATTER
% 
% data_tmp(isnan(data_tmp.trialNum), :) = [];
% data_mat = unstack(data_tmp, {'trialNum'}, 'date', 'AggregationFunction', @numel);
% 
% data_mat = data_mat{:, 2:end};
% xvars = repmat([1 2 3], [size(data_mat, 1) 1]);
% set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');
% 
% close all; 
% subplot(3,3,[1]); hold on;
% s = scatter(xvars(:), data_mat(:), 10,  [0.5 0.5 0.5], 'o', 'jitter', 'on', 'jitteramount', 0.1);
% scatter([2 2], sucrosetrials, 15, [0.5 0.5 0.5], 'd', 'jitter', 'on', 'jitteramount', 0.1);
% 
% % different errorbars
% colors = linspecer(4);
% for i = 1:3,
%     e{i} = errorbar(i, nanmean(data_mat(:, i)), nanstd(data_mat(:, i)), ...
%         'o', 'color',  'k', 'markerfacecolor', 'w', 'markeredgecolor', 'k', 'linewidth', 1, 'capsize', 0);
% end
% xlim([0.5 3.5]);
% 
% water = unique(data.water);
% water = water([1 3 2]);
% set(gca, 'xtick', 1:3, 'xticklabel', water, ...
%     'xticklabelrotation', -20, 'TickLabelInterpreter', 'tex');
% title('Water restriction regimes');
% 
% ylabel({'Number of trials' 'on Monday (CSHL)'});
% %offsetAxes;
% %subplot(333); axis off;
% % 
% % % add the pairs
% % subplot(444);
% % scatter(data_mat(:, 2), data_mat(:, 4), 15, [0.5 0.5 0.5], 'o');
% % xlabel('Trials after 1ml/day'); ylabel('Trials after 2% CA water');
% % axis square;
% % axisEqual; r = refline(1,0); r.Color = 'k'; r.LineWidth = 0.5;
% % %offsetAxes;


% 
% %% pivot the table
% data2 = unstack(data, {'trialNum'}, 'weekday', 'AggregationFunction', @max);
% 
% % plot
% close all;
% g = gramm('x', data2.Fri, 'y', data2.Mon, 'color', data2.water);
% g.geom_point()
% g.geom_abline('slope', 1, 'intercept', 0, 'style', 'k-')
% g.set_names('x', '# Trials on Friday', 'y', '# Trials on Monday', 'color', 'Regime');
% g.axe_property('xlim', [150 1000], 'ylim', [150 1000]);
% 
% %g.stat_summary('type', 'ci', 'geom', 'black_errorbar');
% g.stat_glm()
% g.draw()
% 
% print(gcf, '-dpdf', fullfile(foldername, 'citricAcid_CSHL.pdf'));


end

