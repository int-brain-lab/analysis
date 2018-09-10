function alldata = readAlf_allData(datapath, useSubjects)
% READ ALL ALF DATA INTO ONE MASSIVE DATAFRAME

if ~exist('datapath', 'var') || isempty(datapath),
    % without inputs, make some informed guesses about the most likely user
    if ispc,
        usr = getenv('USERNAME');
        homedir = getenv('USERPROFILE');
        datapath = fullfile(homedir, 'Google Drive');
    elseif ismac,
        usr = getenv('USER');
        homedir = getenv('HOME');
        datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
    end
end

% iterate over the different labs
labs            = {'CSHL/Subjects', 'CCU/Subjects', 'UCL/Subjects'};
alldata         = {};

for l = 1:length(labs),
    
    mypath    = fullfile(datapath, labs{l});
    subjects  = nohiddendir(mypath);
    subjects  = {subjects.name};
    subjects(ismember(subjects, {'default'})) = [];
    subjects(ismember(subjects, {'exampleSubject'})) = [];
    subjects(ismember(subjects, {'180409'})) = [];
    
    % only select a subset if this input arg was given
    if exist('useSubjects', 'var'),
        subjects = subjects(ismember(subjects, useSubjects));
    end
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects),
        if ~isdir(fullfile(mypath, subjects{sjidx})), continue; end
        days  = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        
        % for each day, print in a different color
        for dayidx = 1:length(days), % skip the first week!
            
            clear sessiondata;
            sessions = nohiddendir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
            for sessionidx = 1:length(sessions),
                
                % READ DATA FOR THIS ANIMAL, DAY AND SESSION
                try
                    data = readAlf(fullfile(sessions(sessionidx).folder, sessions(sessionidx).name));
                catch
                    warning('Failed to read %s/%s \n', sessions(sessionidx).folder, sessions(sessionidx).name)
                    continue;
                end
                
                % add some info for the full table
                if ~isempty(data),
                    
                    % ADD SOME BLANKS FOR EASIER VISUALISATION OF SESSION BOUNDARIES
                    data = [data; array2table(nan(20, width(data)), 'variablenames', data.Properties.VariableNames)];
                    data{:, 'animal'}   = {data.Properties.UserData.animal};
                    
                    try
                        % comment Zach on 18 May: cities more salient than institutions
                        switch data.Properties.UserData.lab
                            case 'CSHL'
                                place = ' NY (CSHL)';
                            case 'CCU'
                                place = 'Lisbon (CCU)';
                            case 'UCL'
                                place = 'London (UCL)';
                        end
                        data{:, 'lab'}      = {place};
                        data{:, 'name'}     = {cat(2, place, ' ', data.Properties.UserData.animal)};
                        
                    end
                    
                    data.date           = repmat(datetime(data.Properties.UserData.date), height(data), 1);
                    data.dayidx         = repmat(dayidx, height(data), 1);
                    data.dayidx_rev     = repmat(dayidx-length(days), height(data), 1);
                    data.session        = repmat(data.Properties.UserData.session, height(data), 1);
                    
                    % add a string for dates, easier to plot as title
                    data.datestr        = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), data.date, 'un', 0);
                    
                    % keep for appending
                    alldata{end+1}      = data;
                end
            end
        end
    end
end

% put all the data in one large table
alldata = cat(1, alldata{:});

if ~isempty(alldata),
    % remove those trials that are marked in signals as 'not to be included'
    % alldata(alldata.inclTrials == 0, :) = [];
end

end


%% helper function to remove hidden directories from dir
function x = nohiddendir(p)
x = dir(p);
x = x(~ismember({x.name},{'.','..', '.DS_Store'}));
end

