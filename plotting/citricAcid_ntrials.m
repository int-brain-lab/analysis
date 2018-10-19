function citricAcid_ntrials

data = readAlf_allData(fullfile(getenv('HOME'), 'Google Drive', 'IBL_DATA_SHARE'), ...
    {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_1', 'IBL_3', 'IBL_6', 'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17', ...
    'IBL_10', 'IBL_8'});
data_full = data;

foldername = fullfile(getenv('HOME'), 'Google Drive', 'Rig building WG', ...
    'DataFigures', 'BehaviourData_Weekly', '2018-10-15');

%% grab only those dates that are immediately before and after citric acid
% or normal water intervention
data = data_full;
data = data(ismember(datenum(data.date), datenum({'2018-09-24', '2018-10-09', '2018-10-15'})), :);
[~, data.weekday] = weekday(datenum(data.date));
data.weekday = cellstr(data.weekday);

data.water = data.animal;
data.trialNum(abs(data.response) ~= 1) = NaN;
%data.trialNum(data.correct == 0) = NaN; % whether or not to remove error trials from count?

data.water(datenum(data.date) == datenum('2018-09-24')) = {'weekend 1ml/day, tap water in rig'};
%data.water(datenum(data.date) == datenum('2018-10-01')) = {'CA 5% in hydrogel'};
data.water(datenum(data.date) == datenum('2018-10-09')) = {'weekend adlib 2% CA water, tap water in rig'};
data.water(datenum(data.date) == datenum('2018-10-15')) = {'weekend adlib 2% CA water, sucrose 15% water in rig'};

data = data(:, {'water', 'trialNum', 'animal', 'weekday', 'date'});
data_tmp = data(:, {'trialNum', 'animal', 'date'});
data_tmp(isnan(data_tmp.trialNum), :) = [];
data_mat = unstack(data_tmp, {'trialNum'}, 'date', 'AggregationFunction', @numel);


% separate out the mice that got sucrose on the 9th of October in the rig
sucrosetrials = data_mat{ismember(data_mat.animal, {'IBL_33', 'IBL_13'}), 3};
data_mat{ismember(data_mat.animal, {'IBL_33', 'IBL_13'}), 3} = NaN;

data_mat = data_mat{:, 2:end};
xvars = repmat([1 2 3], [size(data_mat, 1) 1]);

%% BARGRAPH WITH SCATTER
set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');

close all; 
subplot(3,3,[1]); hold on;
s = scatter(xvars(:), data_mat(:), 10,  [0.5 0.5 0.5], 'o', 'jitter', 'on', 'jitteramount', 0.1);
scatter([2 2], sucrosetrials, 15, [0.5 0.5 0.5], 'd', 'jitter', 'on', 'jitteramount', 0.1);

% different errorbars
colors = linspecer(4);
for i = 1:3,
    e{i} = errorbar(i, nanmean(data_mat(:, i)), nanstd(data_mat(:, i)), ...
        'o', 'color',  'k', 'markerfacecolor', 'w', 'markeredgecolor', 'k', 'linewidth', 1, 'capsize', 0);
end
xlim([0.5 3.5]);

water = unique(data.water);
water = water([1 3 2]);
set(gca, 'xtick', 1:3, 'xticklabel', water, ...
    'xticklabelrotation', -20, 'TickLabelInterpreter', 'tex');
title('Water restriction regimes');

ylabel({'Number of trials' 'on Monday (CSHL)'});
%offsetAxes;
%subplot(333); axis off;
% 
% % add the pairs
% subplot(444);
% scatter(data_mat(:, 2), data_mat(:, 4), 15, [0.5 0.5 0.5], 'o');
% xlabel('Trials after 1ml/day'); ylabel('Trials after 2% CA water');
% axis square;
% axisEqual; r = refline(1,0); r.Color = 'k'; r.LineWidth = 0.5;
% %offsetAxes;

tightfig;
print(gcf, '-dpdf', fullfile(foldername, 'citricAcid_trialCounts_CSHL.pdf'));
print(gcf, '-dpdf', '/Users/urai/Google Drive/2018 Postdoc CSHL/CitricAcid/citricAcid_trialCounts_CSHL_alltrials.pdf');
print(gcf, '-dpdf', '~/Google Drive/Rig building WG/Posters/SfN2018/Panels/citricAcid_trialCounts_CSHL_alltrials.pdf');

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

