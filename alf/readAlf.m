function outp = readAlf(foldername)
% reads in an Alf folder and outputs a structure or table (when Matlab
% version allows)
% Anne Urai, CSHL
% 17 April 2018

% requires https://github.com/kwikteam/npy-matlab
assert(exist('readNPY', 'file') > 0, 'readNPY must be on your Matlab path, get it at github.com/kwikteam/npy-matlab');

% if no input was specified, prompt the user
if ~exist('foldername', 'var') || (exist('foldername', 'var') && isempty(foldername)),
    foldername = uigetdir('', 'Choose a session folder with Alf files');
end

% skip the 'default' subjects
if strfind(foldername, 'default') | strfind(foldername, 'exampleSubject'),
    outp = [];
    warning('skipping %s \n', foldername);
    return;
end

% make sure that the directory we're using looks correct-ish
files = dir(foldername);
if sum(strfind([files(:).name], '.npy')) < 1, % if there are no .npy files in this folder
    outp = []; return;
end
fprintf('Reading Alf folder %s \n', foldername);

try
    % READ IN ALL THE RELEVANT FILES
    outp.stimOnTime         = readNPY(fullfile(foldername, 'cwStimOn.times.npy'));
    outp.contrastLeft       = readNPY(fullfile(foldername, 'cwStimOn.contrastLeft.npy'));
    outp.contrastRight      = readNPY(fullfile(foldername, 'cwStimOn.contrastRight.npy'));
    outp.signedContrast     = -outp.contrastLeft + outp.contrastRight;
    
    outp.goCueTime          = readNPY(fullfile(foldername, 'cwGoCue.times.npy')); % what's the go cue? auditory?
    
    outp.responseOnTime     = readNPY(fullfile(foldername, 'cwResponse.times.npy'));
    outp.response           = readNPY(fullfile(foldername, 'cwResponse.choice.npy'));
    outp.response(outp.response == 1)  = -1;
    outp.response(outp.response == 2)  = 1;
    
    outp.feedbackOnTime     = readNPY(fullfile(foldername, 'cwFeedback.times.npy'));
    outp.correct            = readNPY(fullfile(foldername, 'cwFeedback.type.npy'));
    outp.correct            = (outp.correct > 0);
    
    % only include trials that were not a repeat
    outp.inclTrials         = readNPY(fullfile(foldername, 'cwTrials.inclTrials.npy'));

catch
    % if for some reason not all the files are readable, return empty
    outp = [];
    warning('Failed to read %s \n', foldername);
    return;
end

% rewardVolume is written in different ways...
if exist(fullfile(foldername, 'cwFeedback.rewardVolume.npy'), 'file'),
    outp.rewardVolume       = readNPY(fullfile(foldername, 'cwFeedback.rewardVolume.npy'));
elseif exist(sprintf('%s/cwReward.type.npy', foldername), 'file'),
    % also read in the older way - at some point correct these filenames
    outp.rewardVolume       = readNPY(fullfile(foldername, 'cwReward.type.npy'));
else
    outp.rewardVolume = nan(size(outp.response));
end
if ~isequal(size(outp.rewardVolume), size(outp.response)),
    outp.rewardVolume = nan(size(outp.response));
end

% in basicChoiceWorld2, also code for highRewardSide
if exist(sprintf('%s/cwFeedback.highRewardSide.npy', foldername), 'file'),
    outp.highRewardSide       = readNPY(sprintf('%s/cwFeedback.highRewardSide.npy', foldername));
    if ~iscolumn(outp.highRewardSide),
        outp.highRewardSide = outp.highRewardSide';
    end
end
if isfield(outp, 'highRewardSide'),
    % if there is a highRewardSize file present, confirm that it makes sense
    checkHighReward = outp.rewardVolume( outp.correct(:) == 1 & (outp.response(:) == outp.highRewardSide(:)) & abs(outp.signedContrast(:)) > 0);
    assert(mean(checkHighReward == max(outp.rewardVolume)) > 0.95, 'highRewardSide does not make sense');
else
    outp.highRewardSide = nan(size(outp.response));
end

% in shaping, code for block type
if exist(fullfile(foldername, 'stimOn.blockType.npy'), 'file'),
    outp.blockType       = readNPY(fullfile(foldername, 'stimOn.blockType.npy'));
    if ~iscolumn(outp.blockType),
        outp.blockType = outp.blockType';
    end
else
    outp.blockType = nan(size(outp.response));
end

% if this doesn't look like a proper session, return empty
if all(isnan(outp.response)) || length(outp.response) < 10,
    outp = [];
    warning('Less than 10 responses present, not reading %s \n', foldername);
    return;
end

% MAKE SURE ALL ARE COLUMN VECTORS
flds = fieldnames(outp);
for f = 1:length(flds),
    if ~iscolumn(outp.(flds{f})),
        outp.(flds{f}) = outp.(flds{f})';
    end
end

% MAKE SURE THAT ALL FIELDS HAVE THE SAME SIZE
% if the task was stopped with a stimulus on the screen, pad the response &
% reward vectors
flds = fieldnames(outp);
for f = 1:length(flds),
    if length(outp.(flds{f})) < length(outp.responseOnTime),
        outp.(flds{f}) = [outp.(flds{f}); NaN];
    elseif length(outp.(flds{f})) > length(outp.responseOnTime),
        outp.(flds{f}) = outp.(flds{f})(1:length(outp.responseOnTime));
    end
end

% SANITY CHECKS ON THE FILE TIMINGS
assert(~any((outp.responseOnTime - outp.goCueTime) < 0), 'response cannot be earlier than go cue');
assert(~any((outp.responseOnTime - outp.stimOnTime) < 0), 'response cannot be earlier than stimulus');
assert(~any((outp.goCueTime - outp.stimOnTime) < 0), 'go cue cannot be earlier than stimulus');
% assert(~any((outp.feedbackOnTime - outp.responseOnTime) < 0), 'feedback cannot be earlier than response');

% check if feedback is correctly coded
if ~(nansum(abs(sign(outp.signedContrast(abs(outp.signedContrast) > 0)) == sign(outp.response(abs(outp.signedContrast) > 0))) - ...
        outp.correct(abs(outp.signedContrast) > 0)) == 0),
    warning('stimulus and response do not match to feedback');
end

% ADD SOME MORE USEFUL INFO
outp.rt                 = outp.responseOnTime - outp.goCueTime; % RT from stimulus offset = go cue
outp.trialNum           = transpose(1:length(outp.stimOnTime));
if max(abs(outp.signedContrast)) <= 1,
    outp.signedContrast = outp.signedContrast * 100; % in %
end

% if there was no reward switch within the session, ignore
if ~all(isnan(outp.highRewardSide)),
    if numel(unique(outp.highRewardSide)) == 1,
        outp.highRewardSide = nan(size(outp.highRewardSide));
    end
end

% output a table in Matlab >= 2013b
if ~verLessThan('matlab','8.2')
    
    % make sure all the fields are the same size
    outp = struct2table(outp);
    
    % add metadata to the table
    outp.Properties.VariableUnits{'stimOnTime'}     = 'seconds from experiment start';
    outp.Properties.VariableUnits{'goCueTime'}      = 'seconds from experiment start';
    outp.Properties.VariableUnits{'responseOnTime'} = 'seconds from experiment start';
    outp.Properties.VariableUnits{'feedbackOnTime'} = 'seconds from experiment start';
    outp.Properties.VariableUnits{'rt'}             = 'seconds from goCue';
    
    try outp.Properties.VariableUnits{'rewardVolume'}   = 'microlitre'; end
    outp.Properties.VariableUnits{'signedContrast'} = 'percentage';
    outp.Properties.VariableUnits{'contrastLeft'}   = 'percentage';
    outp.Properties.VariableUnits{'contrastRight'}  = 'percentage';
    
end

%% ADD HIGH-LEVEL METADATA
metadata            = strsplit(foldername, filesep); % get some info from the folder name, this will need to be much more extensive
if isempty(metadata{end}), metadata = metadata(1:end-1); end

userdata            = struct('lab', metadata{end-4}, ...
    'animal', metadata{end-2}, ...
    'date',  datetime(metadata{end-1}), ...
    'session', str2double(metadata{end}));
% save together with the data
outp.Properties.UserData = userdata;

end
