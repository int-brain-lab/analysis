function alldata = readAlf_allData()
% READ ALL ALF DATA INTO ONE MASSIVE DATAFRAME

if ~exist('datapath', 'var'),
    % without inputs, make some informed guesses about the most likely user
    usr = getenv('USER');
    switch usr
        case 'anne'
            datapath = '/Users/anne/Google Drive/IBL_DATA_SHARE';
    end
else % ask the user for input
    datapath = uigetdir('', 'Where is the IBL_DATA_SHARE folder?');
end

% iterate over the different labs
labs            = {'CSHL/Subjects', 'CCU/npy', 'UCL/Subjects'};
alldata         = {};

for l = 1:length(labs),
    
    mypath    = sprintf('%s/%s/', datapath, labs{l});
    subjects  = nohiddendir(mypath);
    subjects  = {subjects.name};
    subjects(ismember(subjects, {'default'})) = [];
    subjects(ismember(subjects, {'exampleSubject'})) = [];
    
    %% LOOP OVER SUBJECTS, DAYS AND SESSIONS
    for sjidx = 1:length(subjects),
        if ~isdir(fullfile(mypath, subjects{sjidx})), continue; end
        days  = nohiddendir(fullfile(mypath, subjects{sjidx})); % make sure that date folders start with year
        
        %% FOR UCL MICE, CHECK THE PROTOCOLS.TXT FILE TO ONLY USE BASICCHOICEWORLD
        %         protocolsfile = nohiddendir(fullfile(mypath, subjects{sjidx}, 'protocols.txt'));
        %         if ~isempty(protocolsfile),
        %             protocols = readtable(sprintf('%s/%s', protocolsfile.folder, protocolsfile.name));
        %
        %             % find the days where the animals did basicChoiceWorld
        %             basicChoiceWorld_days = protocols{~cellfun(@isempty, strfind(lower(protocols{:, end}), 'basicchoiceworld')), 1};
        %             days2use = find(ismember({days(:).name}, basicChoiceWorld_days));
        %         else
        %             days2use = 1:length(days);
        %         end
        %
        % for each day, print in a different color
        for dayidx = 1:length(days), % skip the first week!
            
            clear sessiondata;
            sessions = dir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
            for sessionidx = 1:length(sessions),
                
                %% READ DATA FOR THIS ANIMAL, DAY AND SESSION
                data = readAlf(sprintf('%s/%s', sessions(sessionidx).folder, sessions(sessionidx).name));
                
                % add some info for the full table
                if ~isempty(data),
                    
                    data{:, 'animal'}   = {data.Properties.UserData.animal};
                    data{:, 'lab'}      = {data.Properties.UserData.lab};
                    data{:, 'name'}     = {cat(2, data.Properties.UserData.lab, ' ', data.Properties.UserData.animal)};
                    data.date           = repmat(datetime(data.Properties.UserData.date), height(data), 1);
                    data.dayidx         = repmat(dayidx, height(data), 1);
                    data.session        = repmat(data.Properties.UserData.session, height(data), 1);
                    
                    % keep for appending
                    alldata{end+1}      = data;
                end
            end
        end
    end
end

% put all the data in one large table
alldata = cat(1, alldata{:});

end


%% helper function to remove hidden directories from dir
function x = nohiddendir(p)
x = dir(p);
x = x(~ismember({x.name},{'.','..', '.DS_Store'}));
end

