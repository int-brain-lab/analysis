function citricAcid_ntrials

data = readAlf_allData(fullfile(getenv('HOME'), 'Google Drive', 'IBL_DATA_SHARE'), ...
    {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_1', 'IBL_3', 'IBL_6',...
    'IBL_11',  'IBL_12',  'IBL_13',  'IBL_14',  'IBL_15',  'IBL_16',  'IBL_17'});

% grab only those dates that are immediately before and after citric acid
% or normal water intervention

data = data(ismember(datenum(data.date), datenum({'2018-09-21', '2018-09-24', '2018-09-28', '2018-10-01'})), :);
[~, data.weekday] = weekday(datenum(data.date));
data.weekday = cellstr(data.weekday);

data.water = data.animal;
data.water(datenum(data.date) < datenum('2018-09-25')) = {'1ml/day'};
data.water(datenum(data.date) > datenum('2018-09-25')) = {'CA hydrogel'};
data = data(:, {'water', 'trialNum', 'animal', 'weekday'});

% pivot the table
data2 = unstack(data, {'trialNum'}, 'weekday', 'AggregationFunction', @max);

% plot
close all;
g = gramm('x', data2.Fri, 'y', data2.Mon, 'color', data2.water);
g.geom_point()
g.geom_abline('slope', 1, 'intercept', 0, 'style', 'k-')
g.set_names('x', '# Trials on Friday', 'y', '# Trials on Monday', 'color', 'Regime');
g.axe_property('xlim', [150 1000], 'ylim', [150 1000]);

%g.stat_summary('type', 'ci', 'geom', 'black_errorbar');
g.stat_glm()
g.draw()

foldername = fullfile(getenv('HOME'), 'Google Drive', 'Rig building WG', ...
    'DataFigures', 'BehaviourData_Weekly', '2018-10-09');
print(gcf, '-dpdf', fullfile(foldername, 'citricAcid_CSHL.pdf'));


end

