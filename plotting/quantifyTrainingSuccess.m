function quantifyTrainingSuccess(datapath)
% quantify for each mouse how long it took them to get trained on
% basicChoiceWorld, for SfN18 abstract
% Anne Urai, 2018

if ~exist('datapath', 'var'),
    % without inputs, make some informed guesses about the most likely user
    usr = getenv('USER');
    switch usr
        case 'anne'
            datapath        = '/Users/anne/Google Drive/IBL_DATA_SHARE';
    end
else % ask the user for input
    datapath = uigetdir('', 'Where is the IBL_DATA_SHARE folder?');
end

% iterate over the different labs
labs            = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
clc;

for l = 1:length(labs),
    
    mypath    = sprintf('%s/%s/', datapath, labs{l});
    subjects  = dir(mypath);
    subjects  = {subjects.name};
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects),
        if ~isdir(fullfile(mypath, subjects{sjidx})), continue; end
        days  = dir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        site  = strsplit(labs{l}, '/');
        name  = regexprep(subjects(sjidx), '_', ' ');
        if ~iscell(name), name = {name}; end
        
        for dayidx = 1:length(days), % skip the first week!
            
            clear sessiondata;
            sessions = dir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
            for sessionidx = 1:length(sessions),
                
                %% READ DATA FOR THIS ANIMAL, DAY AND SESSION
                sessiondata{sessionidx} = readAlf(sprintf('%s/%s', sessions(sessionidx).folder, sessions(sessionidx).name));
            end
            
            %% MERGE DATA FROM 2 SESSIONS IN A SINGLE DAY
            data = cat(1, sessiondata{:});
            if isempty(data ) || height(data ) < 10, continue;  end
            
            %% COMPUTE SOME MEASURES OF PERFORMANCE
            accuracy = nanmean(data.correct(abs(data.signedContrast) > 80));
            
            %% SAVE IN LARGE TABLE
            performance = struct('lab', {site(1)}, 'animal', {name}, ...
                'date', datetime(days(dayidx).name), 'dayidx', dayidx, 'session', sessionidx, ...
                'accuracy', accuracy, 'ntrials', height(data), 'rt', nanmedian(data.rt));
            
            if ~exist('performance_summary', 'var'),
                performance_summary = struct2table(performance);
            else
                performance_summary = cat(1, performance_summary, struct2table(performance));
            end
            
        end
    end
end

% ====================================== %
%% OUTPUT SUMMARY STATS
% ====================================== %

% On average, x% of all animals trained were successful (defined as x y z) (report % for each location).
performanceCriterion            = 0.85;
performance_summary.istrained   = (performance_summary.accuracy > performanceCriterion & performance_summary.rt < 3);

% group per lab & animal
[gr, lab, animal]               = findgroups(performance_summary.lab, performance_summary.animal);
performance_perlab              = array2table([lab, animal], 'variablenames', {'lab', 'animal'});
performance_perlab.istrained    = splitapply(@nanmax, performance_summary.istrained, gr);

% for mice that don't reach this on average, set all istrained to 0
performance_summary.istrained(ismember(performance_summary.animal, performance_perlab.animal(~performance_perlab.istrained))) = 0;

% summary statement about the data
fprintf('A total of %d mice were trained across the three labs (%d at UCL, %d at CCU, %d at CSHL) using the same behavioral rig and automated training protocol. \n', ...
    height(performance_perlab), sum(ismember(performance_perlab.lab, 'UCL')), ...
    sum(ismember(performance_perlab.lab, 'CCU')),  sum(ismember(performance_perlab.lab, 'CSHL')));

fprintf('On average, %d%% of all animals were successful: ', round(100*mean(performance_perlab.istrained)));
fprintf('%d%% at UCL, %d%% at CCU, %d%% at CSHL. \n', ...
    round(100*mean(performance_perlab.istrained(ismember(performance_perlab.lab, 'UCL')))), ...
    round(100*mean(performance_perlab.istrained(ismember(performance_perlab.lab, 'CCU')))), ...
    round(100*mean(performance_perlab.istrained(ismember(performance_perlab.lab, 'CSHL')))));

% On average, animals took x days of training to reach 80% performance (mean+std for each location).
performance_summary.dayidx_trained = performance_summary.dayidx;
performance_summary.dayidx_trained(~performance_summary.istrained) = NaN;

performance_perlab.days2trained    = splitapply(@nanmin, performance_summary.dayidx_trained, gr);
fprintf('On average, animals took %d days of training to reach stable performance: ', round(nanmean(performance_perlab.days2trained)));
fprintf('%d +- %d at UCL, %d +- %d at CCU, %d +- %d at CSHL. \n', ...
    round(nanmean(performance_perlab.days2trained(ismember(performance_perlab.lab, 'UCL')))), ...
    round(nanstd(performance_perlab.days2trained(ismember(performance_perlab.lab, 'UCL')))),...
    round(nanmean(performance_perlab.days2trained(ismember(performance_perlab.lab, 'CCU')))), ...
    round(nanstd(performance_perlab.days2trained(ismember(performance_perlab.lab, 'CCU')))), ...
    round(nanmean(performance_perlab.days2trained(ismember(performance_perlab.lab, 'CSHL')))), ...
    round(nanstd(performance_perlab.days2trained(ismember(performance_perlab.lab, 'CSHL')))));

% Daily trial count for trained animals was x (mean+std for each location).
performance_summary.ntrials_trained = performance_summary.ntrials;
performance_summary.ntrials_trained(~performance_summary.istrained) = NaN;

performance_perlab.ntrials_trained    = splitapply(@nanmean, performance_summary.ntrials_trained, gr);
fprintf('Trained animals did on average %d trials per day: ', round(nanmean(performance_perlab.ntrials_trained)));
fprintf('%d +- %d at UCL, %d +- %d at CCU, %d +- %d at CSHL. \n', ...
    round(nanmean(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'UCL')))), ...
    round(nanstd(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'UCL')))),...
    round(nanmean(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'CCU')))), ...
    round(nanstd(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'CCU')))), ...
    round(nanmean(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'CSHL')))), ...
    round(nanstd(performance_perlab.ntrials_trained(ismember(performance_perlab.lab, 'CSHL')))));

disp('fin');


