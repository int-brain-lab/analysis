
function saveAlf_CSHL(subjects)
% Saves Rigbox files from Matlab to ALF structure.
% Dependencies:
%   https://github.com/kwikteam/npy-matlab
%   https://github.com/cortex-lab/alyx-matlab
% Anne Urai, CSHL, 2018
%
% 6 April: added writing reward volume
% 13 April: moving water volume to cwFeedback.rewardVolume.npy files
% 13 April: save contrast in %
% 13 Apri: temporary, remove wheel timestamps to save space

addpath('\\NEW-9SE8HAULSQE\Users\IBL_Master\Documents\IBLData_Shared\code\alyx-matlab-master');
addpath('\\NEW-9SE8HAULSQE\Users\IBL_Master\Documents\IBLData_Shared\code\npy-matlab-master');

%% TODO: MAKE SURE THE PYTHON PATH IS THE ONE THAT'S LINKED TO GOOGLE DRIVE ON THE RIG PC
global matlabPath pythonPath
matlabPath = '\\NEW-9SE8HAULSQE\Users\IBL_Master\Documents\IBLData_Shared\data\subjects';
pythonPath = '\\NEW-9SE8HAULSQE\Users\IBL_Master\Google Drive\CSHL\Subjects';

if ~isdir(pythonPath), mkdir(pythonPath); disp('Creating a new directory for ALF files'); end

%% IF NO SUBJECTS WERE GIVEN, CHECK ALL OF THEM
if ~exist('subjects', 'var'),
    sjfolders = nohiddendir(matlabPath);
    subjects  = {sjfolders.name};
    subjects(ismember(subjects, {'exampleSubject', 'default'})) = [];
end

%% LOOP OVER SUBJECTS, DAYS AND SESSIONS
for sjidx = 1:length(subjects)
    days = nohiddendir(fullfile(matlabPath, subjects{sjidx})); % make sure that date folders start with year
    for dayidx = 1:length(days),
        sessions = nohiddendir(fullfile(days(dayidx).folder, days(dayidx).name)); % make sure that date folders start with year
        for sessionidx = 1:length(sessions),
            write2alf(fullfile(subjects{sjidx}, days(dayidx).name, sessions(sessionidx).name));
        end
    end
end
end

%% DO THE ACTUAL EXTRACTION AND CONVERSION
function write2alf(filename)
global matlabPath pythonPath

% MAKE A NEW PATH IN THE PYTHON FOLDER
expPath = fullfile(matlabPath, filename);
newpath = fullfile(pythonPath, filename);
if ~exist(newpath, 'dir'),
    mkdir(newpath); fprintf('Created directory %s \n', newpath);
else
    fprintf('Directory %s already exists, skipping \n', newpath);
      return
end

% GET THE DATA FROM THE MATLAB FOLDER
files = dir([expPath filesep '*.mat']);
for f = 1:length(files),
    load(fullfile(files(f).folder, files(f).name));
end

if ~exist('block', 'var'),
    warning('No blocks file found, skipping');
    return
end

expDef = getOr(block, 'expDef', []);
if isempty(expDef); return; end
[~, expDef] = fileparts(expDef);
if ~contains(lower(expDef), 'choiceworld') || ~isfield(block, 'events') || length(block.events.newTrialValues) < 10 || isempty(block.outputs.rewardValues)
    return
end

%% Write response
response = getOr(block.events, 'responseValues', NaN);
if contains(lower(block.expDef), {'basic' 'vanilla'})
    hits = [block.events.hitValues];
    side = [block.events.trialSideValues];
    side = side(1:length(hits));
    response = nan(1,length(hits));
    response((side==-1&hits==1)|(side==1&hits==0)) = 1;
    response((side==1&hits==1)|(side==-1&hits==0)) = 2;
end
if min(response) == -1
    response(response == 0) = 3;
    response(response == 1) = 2;
    response(response == -1) = 1;
end

if isempty(response),
    return;
end
try
    writeNPY(response(:), fullfile(expPath, 'cwResponse.choice.npy'));
    movefile(fullfile(expPath, 'cwResponse.choice.npy'), newpath, 'f');
    responseTimes = [block.events.responseTimes]-block.events.expStartTimes;
    alf.writeEventseries(expPath, 'cwResponse', [block.events.responseTimes]-block.events.expStartTimes, [], []);
    movefile(fullfile(expPath, 'cwResponse.times.npy'), newpath, 'f');
catch
    warning('No ''feedback'' events recorded, cannot register to Alyx')
end

%% Write stim on times - only those which had a response
if isfield(block.events, 'stimulusOnTimes')
    stimOnTimes = [block.events.stimulusOnTimes]-block.events.expStartTimes;
else
    stimOnTimes = [block.events.stimOnTimes]-block.events.expStartTimes;
end
if length(stimOnTimes) > length(responseTimes),
    stimOnTimes = stimOnTimes(1:length(responseTimes));
end
assert(all(responseTimes > stimOnTimes), 'response cannot precede stimulus');

try
    alf.writeEventseries(expPath, 'cwStimOn', stimOnTimes, [], []);
    movefile(fullfile(expPath, 'cwStimOn.times.npy'), newpath, 'f');
catch
    warning('No ''stimulusOn'' events recorded, cannot register to Alyx')
end
contL = getOr(block.events, 'contrastLeftValues', NaN(1, length(block.events.newTrialValues)));
contR = getOr(block.events, 'contrastRightValues', NaN(1, length(block.events.newTrialValues)));
if all(isnan(contL)&isnan(contR))
    if contains(lower(block.expDef), {'basic' 'vanilla'})
        side = [block.events.trialSideValues];
        contrasts = [block.events.trialContrastValues];
        contL(side==-1) = contrasts(side==-1);
        contL(side==1) = 0;
        contR(side==1) = contrasts(side==1);
        contR(side==-1) = 0;
    else
        contrasts = [block.paramsValues.stimulusContrast];
        contL = contrasts(1,:);
        contR = contrasts(2,:);
    end
end

try
    writeNPY(contL(1:length(responseTimes))*100, fullfile(expPath, 'cwStimOn.contrastLeft.npy'));
    movefile(fullfile(expPath, 'cwStimOn.contrastLeft.npy'), newpath, 'f');
    writeNPY(contR(1:length(responseTimes))*100, fullfile(expPath, 'cwStimOn.contrastRight.npy'));
    movefile(fullfile(expPath, 'cwStimOn.contrastRight.npy'), newpath, 'f');
catch
    warning('No ''contrastLeft'' and/or ''contrastRight'' events recorded, cannot register to Alyx')
end

%% Write go cue
interactiveOn = getOr(block.events, 'interactiveOnTimes', NaN);
if isnan(interactiveOn)
    interactiveOn = [block.events.stimulusOnTimes]+[block.paramsValues.interactiveDelay];
end

if length(interactiveOn) > length(responseTimes),
    interactiveOn = interactiveOn(1:length(responseTimes));
end
% assert(all(responseTimes > interactiveOn), 'response cannot precede go cue');

try
    alf.writeEventseries(expPath, 'cwGoCue', interactiveOn-block.events.expStartTimes, [], []);
    movefile(fullfile(expPath, 'cwGoCue.times.npy'), newpath, 'f');
catch
    warning('No ''interactiveOn'' events recorded, cannot register to Alyx')
end

%% Write feedback
if isfield(block.events, 'feedbackValues')
    feedback = double(block.events.feedbackValues);
else
    feedback = double([block.events.hitValues]);
end
feedback(feedback == 0) = -1;

if isfield(block.events, 'feedbackTimes')
    feedbackTimes = block.events.feedbackTimes;
else
    feedbackTimes = [block.events.hitTimes];
end
feedbackTimes = feedbackTimes-block.events.expStartTimes;

if length(feedbackTimes) > length(responseTimes),
    % find only those feedback events that occur briefly after a response
    useFbTimes      = dsearchn(feedbackTimes', responseTimes');
    feedbackTimes   = feedbackTimes(useFbTimes);
    feedback        = feedback(useFbTimes);
elseif length(feedbackTimes) < length(responseTimes),
    feedbackTimes = [feedbackTimes NaN];
end
assert(~any(responseTimes > feedbackTimes), 'feedback cannot precede response');
%%

try
    writeNPY(feedback(:), fullfile(expPath, 'cwFeedback.type.npy'));
    movefile(fullfile(expPath, 'cwFeedback.type.npy'), newpath, 'f');
    
    alf.writeEventseries(expPath, 'cwFeedback', feedbackTimes, [], []);
    movefile(fullfile(expPath, 'cwFeedback.times.npy'), newpath, 'f');
catch
    warning('No ''feedback'' events recorded, cannot register to Alyx')
end

%% Write reward volume: cwFeedback.rewardVolume

reward = feedback;
reward(reward == -1) = 0;
rewardTimes = block.outputs.rewardTimes-block.events.expStartTimes;
if length(rewardTimes) == length(feedbackTimes(feedback == 1)),
    reward(reward == 1) = block.outputs.rewardValues();
elseif  length(rewardTimes) > length(feedbackTimes(feedback == 1)),
    % in basicChoiceWorld2, there are rewards given as motivation (not as feedback)
    
    rewardTimingOffset = abs(feedbackTimes(find(feedback == 1, 1)) - rewardTimes(1));
    % for each feedbackTime with a positive feedback, find a rewardTime that is closest
    % get a feeling for the range of the data
    foundRightRewards = 0; timingRange = 4;
    while ~foundRightRewards,
        try
            findClosest = cell2mat(arrayfun(@(a) find(abs(rewardTimes - a) < timingRange * rewardTimingOffset), ...
                feedbackTimes(feedback==1), 'uni', 0));
            reward(reward == 1) = block.outputs.rewardValues(findClosest);
            foundRightRewards = 1;
        catch
            timingRange = timingRange - 0.01;
            try
                assert(timingRange > 0, 'could not match rewards');
            catch
                findClosest = unique(dsearchn(feedbackTimes(feedback == 1)', rewardTimes'));
                reward(reward == 1) = block.outputs.rewardValues(findClosest);
            end
        end
    end
end

writeNPY(reward(:), fullfile(expPath, 'cwFeedback.rewardVolume.npy'));
movefile(fullfile(expPath, 'cwFeedback.rewardVolume.npy'), newpath, 'f');

% remove the older file, avoid clutter
if exist(fullfile(expPath, 'cwReward.type.npy'), 'file'),
    delete(fullfile(expPath, 'cwReward.type.npy'));
end
if exist(fullfile(expPath, 'cwReward.times.npy'), 'file'),
    delete(fullfile(expPath, 'cwReward.times.npy'));
end

signedContrast = contR - contL;
signedContrast = signedContrast(1:length(response));
correct = (sign(signedContrast) == sign(response - 1.5));

assert(all(reward(correct == 1 & abs(signedContrast) > 0) > 0), 'error encoding reward delivery');
assert(all(reward(correct == 0 & abs(signedContrast) > 0) == 0), 'error encoding reward delivery')

%% basicChoiceWorld2: highRewardSide
if isfield(block.events, 'highRewardSideValues'),
    % do some checks
    highRewardSide = block.events.highRewardSideValues(1:length(response));
    checkHighReward = reward(correct == 1 & (sign(response - 1.5) == highRewardSide) & abs(signedContrast) > 0);
    assert(mean(checkHighReward == max(reward)) > 0.95, 'highRewardSide does not make sense');
    
    writeNPY(highRewardSide, fullfile(expPath, 'cwFeedback.highRewardSide.npy'));
    movefile(fullfile(expPath, 'cwFeedback.highRewardSide.npy'), newpath, 'f');
end

%% Write trial intervals
alf.writeInterval(expPath, 'cwTrials',...
    block.events.newTrialTimes(:)-block.events.expStartTimes,...
    block.events.endTrialTimes(:)-block.events.expStartTimes, [], []);
movefile(fullfile(expPath, 'cwTrials.intervals.npy'), newpath, 'f');

if contains(lower(block.expDef), {'basic' 'vanilla'})
    repeatOnMiss = abs(diff([contL; contR])) > 0.4;
    hits = double([block.events.hitValues]);
    if length(hits)<length(repeatOnMiss)
        hits = [hits nan(1,length(repeatOnMiss)-length(hits))];
    end
    repeat = circshift(repeatOnMiss==1&hits==0,1);repeat(1) = 0;
    repNum = ones(1,length([block.events.newTrialValues]));
    for j = 2:length(repeat)
        if repeat(j) == true
            repNum(j) = repNum(j-1)+1;
        end
    end
else
    repNum = [block.events.repeatNumValues];
end
try
    writeNPY(repNum == 1, fullfile(expPath, 'cwTrials.inclTrials.npy'));
    movefile(fullfile(expPath, 'cwTrials.inclTrials.npy'), newpath, 'f');
    writeNPY(repNum, fullfile(expPath, 'cwTrials.repNum.npy'));
    movefile(fullfile(expPath, 'cwTrials.repNum.npy'), newpath, 'f');
catch
    warning('Saving repeatNums failed')
end

%% skip writing full wheel traces for now - takes too much space
% %% Write wheel times, position and velocity
% wheelValues = block.inputs.wheelValues(:)-block.inputs.wheelValues(1);
% switch lower(block.rigName)
%     case {'zrig1', 'zrig2', 'zrig3', 'zrig4',...
%             'zredone', 'zredtwo', 'zredthree', 'zgreyfour'} % spesh
%         encRes = 1024;
%     case {'zym1', 'zym2', 'zym3'}
%         encRes = 360;
%     otherwise
%         encRes = 1024;
% end
% wheelValues = wheelValues*(3.1*2*pi/(4*encRes));
% try
%     wheelTimes = block.inputs.wheelTimes(:);
%     wheelTimes = wheelTimes-block.events.expStartTimes;
%     alf.writeTimeseries(expPath, 'Wheel', wheelTimes, [], []);
%     movefile(fullfile(expPath, 'Wheel.timestamps.npy'), newpath, 'f');
%     writeNPY(wheelValues, fullfile(expPath, 'Wheel.position.npy'));
%     movefile(fullfile(expPath, 'Wheel.position.npy'), newpath, 'f');
%     writeNPY(wheelValues./wheelTimes, fullfile(expPath, 'Wheel.velocity.npy'));
%     movefile(fullfile(expPath, 'Wheel.velocity.npy'), newpath, 'f');
% catch
%     warning('Failed to write wheel values')
% end

%% end of writing to numpy
% disp('Writing to ALF format completed, now trying to register to Alyx');
return; 

%% Registration
try
    sessions = ai.getData(['sessions?type=Base&subject=' subject]);
    [~, expDate, seq] = dat.parseExpRef(block.expRef);
    expDate = ai.datestr(floor(expDate));
    if ~isempty(sessions)
        sessions = catStructs(sessions);
        dates = cellfun(@(a)a(1:10), {sessions.start_time}, 'uni', 0);
        base_idx = strcmp(dates, expDate(1:10));
    else
        base_idx = 0;
    end
    
    %If the date of this latest base session is not the same date as
    %today, then create a new base session for today
    if isempty(sessions) || ~any(base_idx)
        d = struct;
        d.subject = subject;
        d.procedures = {'Behavior training/tasks'};
        d.narrative = 'auto-generated session';
        d.start_time = expDate;
        d.type = 'Base';
        %       d.users = {obj.User}; % FIXME
        
        base_submit = obj.postData('sessions', d);
        assert(isfield(base_submit,'subject'),...
            'Submitted base session did not return appropriate values');
        
        %Now retrieve the sessions again
        sessions = obj.getData(['sessions?type=Base&subject=' subject]);
        latest_base = sessions{end};
    else
        latest_base = sessions(base_idx);
    end
    
    sessions = ai.getData(['sessions?type=Experiment&subject=' subject]);
    if ~isempty(sessions)
        sessions = catStructs(sessions);
        dates = cellfun(@(a)a(1:10), {sessions.start_time}, 'uni', 0);
        exp_idx = strcmp(dates, expDate(1:10))&[sessions.number]==seq;
    else
        exp_idx = 0;
    end
    
    if isempty(sessions) || ~any(exp_idx)
        %Now create a new SUBSESSION, using the same experiment number
        d = struct;
        d.subject = subject;
        d.procedures = {'Behavior training/tasks'};
        d.narrative = 'auto-generated session';
        d.start_time = expDate;
        d.type = 'Experiment';
        d.parent_session = latest_base.url;
        d.number = seq;
        %   d.users = {obj.User}; % FIXME
        subsession = obj.postData('sessions', d);
    else
        subsession = sessions(exp_idx);
    end
    url = subsession.url;
    
    if ~reg; return; end
    
    
    % Register them to Alyx
    ai.registerALF(expPath, url);
catch ex
    fprintf('Failed to register files to Alyx: %s \n', ex.message);
end

%     if isempty(obj.AlyxInstance)
%       warning('No Alyx token set');
%     else
%       try
%         if strcmp(subject,'default')||strcmp(block.endStatus,'aborted'); return; end
%         assert(obj.AlyxInstance.IsLoggedIn, 'No Alyx token set');
%         % Register saved files
%         obj.AlyxInstance.registerFile(savepaths{end}, 'mat',...
%           obj.AlyxInstance.SessionURL, 'Block', []);
%         %                 obj.AlyxInstance.registerFile(savepaths{end}, 'mat',...
%         %                     {subject, expDate, seq}, 'Block', []);
%         % Save the session end time
%         if ~isempty(obj.AlyxInstance.SessionURL)
%           obj.AlyxInstance.putData(obj.AlyxInstance.SessionURL,...
%             struct('end_time', obj.AlyxInstance.datestr(now), 'subject', subject));
%         else
%           % Infer from date session and retrieve using expFilePath
%         end
%       catch ex
%         warning(ex.identifier, 'Failed to register files to Alyx: %s', ex.message);
%       end
%     end
end

%% helper function to remove hidden directories from dir
function x = nohiddendir(p)
x = dir(p);
x = x(~ismember({x.name},{'.','..', '.DS_Store'}));
x = x([x.isdir]); % only return directories
end
