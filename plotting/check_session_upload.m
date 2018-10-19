% TEST WHETHER ALL SESSIONS ARE CORRECTLY UPLOADED TO DRIVE

addpath('~/Documents/code/npy-matlab//');
addpath(genpath('/Users/urai/Documents/code/analysis_IBL'));

if ispc,
    usr = getenv('USERNAME');
    homedir = getenv('USERPROFILE');
    datapath = fullfile(homedir, 'Google Drive');
elseif ismac,
    usr = getenv('USER');
    homedir = getenv('HOME');
    datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
end

set(groot, 'defaultaxesfontsize', 7, 'DefaultFigureWindowStyle', 'normal');
msz = 4;

%% overview
mice = sort({'IBL_16', 'IBL_1', 'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37', ...
    'IBL_3', 'IBL_6', 'IBL_8',  ...
    'IBL_13',  'IBL_14',  'IBL_15',  'IBL_10',  'IBL_17'});

for m = 1:length(mice),
    
    close all;
    data = readAlf_allData(datapath, mice{m});
    
    % use all dates
    [gr, date, session] = findgroups(data.date, data.session);
    numtrials = splitapply(@numel, data.rt, gr);
    
    close all; hold on;
    colormap(viridis);
    stem(date, numtrials, 'k');
    s = scatter(date, numtrials, 30, session, 'filled');
    set(gca, 'xgrid', 'on');
    
    % FIND THOSE DATES THAT ARE WEIRD
    % first, which dates are missing?
    alldates = date(1):date(end);
    weekdays = weekday(alldates);
    alldates(ismember(weekdays, [1 7])) = []; % ignore weekends
    
     % ignore those days when no animals were trained
    % So 3 Sep and 8 Oct were holidays, 12 Sep and 2 Oct no training because I wasn't here.
    alldates(ismember(datenum(alldates), datenum({'2018-09-03', '2018-10-08', '2018-09-12', '2018-10-02'}))) = [];
    missedsessions = setdiff(alldates, date);
    stem(missedsessions, zeros(size(missedsessions)), 'r', 'filled');
    
    % duplicate sessions
    [n, bin] = histc(datenum(date), unique(datenum(date)));
    doublesessions = date(n > 1);
    stem(missedsessions, zeros(size(missedsessions)), 'r', 'filled');

    % layout
    set(gca, 'xtick', unique([date; missedsessions']), 'xticklabelrotation', -90);
    xtickformat('dd MMM, eee');
    title(mice{m}, 'interpreter', 'none');
    ylabel('# trials');
    foldername = fullfile(homedir, 'Google Drive', 'Rig building WG', ...
        'DataFigures', 'BehaviourData_Weekly', '2018-10-22');
    if ~exist(foldername, 'dir'), mkdir(foldername); end
    print(gcf, '-dpdf', fullfile(foldername, sprintf('%s_%s_%s_sessionCheck.pdf', datestr(now, 'yyyy-mm-dd'), ...
        data.Properties.UserData.lab, mice{m})));
    
    
end