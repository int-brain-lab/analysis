function outp = readAlf(foldername)
% reads in an Alf folder and outputs a structure or table (when Matlab
% version allows)
% Anne Urai, CSHL
% 17 April 2018

% requires https://github.com/kwikteam/npy-matlab
assert(exist('readNPY', 'file') > 0, 'readNPY must be on your Matlab path, get it at github.com/kwikteam/npy-matlab');
fprintf('Reading Alf folder %s \n', foldername);

% if no input folder was specified, prompt the user
if ~exist('foldername', 'var') || (exist('foldername', 'var') & isempty(foldername)),
    foldername = uigetdir('', 'Choose a session folder with Alf files');
end

try
    % READ IN ALL THE RELEVANT FILES
    outp.stimOnTime         = readNPY(sprintf('%s/cwStimOn.times.npy', foldername));
    outp.contrastLeft       = readNPY(sprintf('%s/cwStimOn.contrastLeft.npy', foldername));
    outp.contrastRight      = readNPY(sprintf('%s/cwStimOn.contrastRight.npy', foldername));
    outp.signedContrast     = sum([-outp.contrastLeft outp.contrastRight], 2);
    
    outp.goCueTime          = readNPY(sprintf('%s/cwGoCue.times.npy', foldername)); % what's the go cue? auditory?
    
    outp.responseOnTime     = readNPY(sprintf('%s/cwResponse.times.npy', foldername));
    outp.response           = readNPY(sprintf('%s/cwResponse.choice.npy', foldername));
    outp.response(outp.response == 1)  = -1;
    outp.response(outp.response == 2)  = 1;
    
    outp.feedbackOnTime     = readNPY(sprintf('%s/cwFeedback.times.npy', foldername));
    outp.correct            = readNPY(sprintf('%s/cwFeedback.type.npy', foldername));
    outp.correct            = (outp.correct > 0);
catch
    % if for some reason not all the files are readable, return empty
    outp = [];
    return;
end

if exist(sprintf('%s/cwFeedback.rewardVolume.npy', foldername), 'file'),
    % not all files will have this...
    outp.rewardVolume       = readNPY(sprintf('%s/cwFeedback.rewardVolume.npy', foldername));
elseif exist(sprintf('%s/cwReward.type.npy', foldername), 'file'),
    outp.rewardVolume       = readNPY(sprintf('%s/cwReward.type.npy', foldername));
end

% if this doesn't look like a proper session, return empty
if all(isnan(outp.response)) || length(outp.response) < 10,
    outp = [];
    return;
end

% MAKE SURE THAT ALL FIELDS HAVE THE SAME SIZE
% if the task was stopped with a stimulus on the screen, pad the response &
% reward vectors
flds = fieldnames(outp);
for f = 1:length(flds),
    if length(outp.(flds{f})) < length(outp.stimOnTime),
        outp.(flds{f}) = [outp.(flds{f}); NaN];
    elseif length(outp.(flds{f})) > length(outp.stimOnTime),
        outp.(flds{f}) = outp.(flds{f})(1:length(outp.stimOnTime));
    end
end

% If reward volume was only written for correct trials, account for this
if length(outp.rewardVolume) < length(outp.correct),
    rewardVolume = zeros(size(outp.correct));
    rewardVolume(outp.correct == 1) = outp.rewardVolume(~isnan(outp.rewardVolume));
    outp.rewardVolume = rewardVolume;
end

% ADD SOME MORE USEFUL INFO
outp.rt                 = outp.responseOnTime - outp.goCueTime; % RT from stimulus offset = go cue
outp.trialNum           = transpose(1:length(outp.stimOnTime));
if max(abs(outp.signedContrast)) == 1, 
    outp.signedContrast = outp.signedContrast * 100; % in %
end

% output a table in Matlab >= 2013b
if ~verLessThan('matlab','8.2')
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
