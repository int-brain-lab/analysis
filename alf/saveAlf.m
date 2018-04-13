function saveAlf(subjects)
localPath = 'C:\Users\Miles\Desktop\IBLData_Shared\OneDrive - University College London';
ai = Alyx;
for i = 1:length(subjects)
  subject = subjects{i};
  [expRef, ~] = dat.listExps(subject); % List the experiments for subject
  % Find the block files
  filelist = mapToCell(@(r) dat.expFilePath(r, 'block', 'master'),expRef);
  % Filter out those that don't exist
  existfiles = cell2mat(mapToCell(@(l) file.exists(l), filelist));
  % Load the blocks
  allBlocks = cellfun(@load,filelist(existfiles));
  % This will allow use to process blocks based on experiment definition.
  if strcmp(subject, 'ALK069'); allBlocks(13) = []; end
%   if strcmp(subject, 'Kornberg'); allBlocks = allBlocks(16:end); end
  %   expDef = arrayfun(@(b){b.block.expDef},allBlocks);
  %   [~, expDef] = cellfun(@fileparts, expDef, 'UniformOutput', false);
  
  cellfun(@alfs, struct2cell(allBlocks));
end

  function alfs(block)
    expDef = getOr(block, 'expDef', []);
    if isempty(expDef); return; end
    [~, expDef] = fileparts(expDef);
    if ~contains(lower(expDef), 'choiceworld') || ~isfield(block, 'events') || length(block.events.newTrialValues) < 10
      return
    end
    if ~any(strcmpi(expDef, {'advancedChoiceWorld', 'basicChoiceworld', 'vanillaChoiceworld'}))
      x = input(['process ' expDef '? Y/N\n'], 's');
      if strcmpi(x, 'n')
        return
      end
    end
    expPath = dat.expPath(block.expRef, 'main', 'master');
    k = strfind(expPath,'.net');
    newpath = [localPath expPath(k+4:end)];
    
    reg = iff(exist(fullfile(expPath, 'Wheel.velocity.npy'), 'file'), 0, 1);
    
    %% Write feedback
    
    if isfield(block.events, 'feedbackValues')
      feedback = double(block.events.feedbackValues);
    else
      feedback = double([block.events.hitValues]);
    end
    if isfield(block.events, 'feedbackTimes')
      feedbackTimes = block.events.feedbackTimes;
    else
      feedbackTimes = [block.events.hitTimes];
    end
    feedback(feedback == 0) = -1;
    try
      writeNPY(feedback(:), fullfile(expPath, 'cwFeedback.type.npy'));
      if ~exist(newpath, 'dir'); mkdir(newpath); end
      copyfile(fullfile(expPath, 'cwFeedback.type.npy'), [newpath '\']);
      alf.writeEventseries(expPath, 'cwFeedback', feedbackTimes-block.events.expStartTimes, [], []);
      copyfile(fullfile(expPath, 'cwFeedback.times.npy'), newpath);
    catch
      warning('No ''feedback'' events recorded, cannot register to Alyx')
    end
    
    %% Write go cue
    interactiveOn = getOr(block.events, 'interactiveOnTimes', NaN);
    
    if isnan(interactiveOn)
      interactiveOn = [block.events.stimulusOnTimes]+[block.paramsValues.interactiveDelay];
    end
    try
      alf.writeEventseries(expPath, 'cwGoCue', interactiveOn-block.events.expStartTimes, [], []);
      copyfile(fullfile(expPath, 'cwGoCue.times.npy'), newpath);
    catch
      warning('No ''interactiveOn'' events recorded, cannot register to Alyx')
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
    try
      writeNPY(response(:), fullfile(expPath, 'cwResponse.choice.npy'));
      copyfile(fullfile(expPath, 'cwResponse.choice.npy'), newpath);
      alf.writeEventseries(expPath, 'cwResponse', [block.events.responseTimes]-block.events.expStartTimes, [], []);
      copyfile(fullfile(expPath, 'cwResponse.times.npy'), newpath);
    catch
      warning('No ''feedback'' events recorded, cannot register to Alyx')
    end
    
    %% Write stim on times
    if isfield(block.events, 'stimulusOnTimes')
      stimOnTimes = [block.events.stimulusOnTimes]-block.events.expStartTimes;
    else
      stimOnTimes = [block.events.stimOnTimes]-block.events.expStartTimes;
    end
    
    try
      alf.writeEventseries(expPath, 'cwStimOn', stimOnTimes, [], []);
      copyfile(fullfile(expPath, 'cwStimOn.times.npy'), newpath);
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
      writeNPY(contL(:), fullfile(expPath, 'cwStimOn.contrastLeft.npy'));
      copyfile(fullfile(expPath, 'cwStimOn.contrastLeft.npy'), newpath);
      writeNPY(contR(:), fullfile(expPath, 'cwStimOn.contrastRight.npy'));
      copyfile(fullfile(expPath, 'cwStimOn.contrastRight.npy'), newpath);
    catch
      warning('No ''contrastLeft'' and/or ''contrastRight'' events recorded, cannot register to Alyx')
    end
    
    %% Write trial intervals
    alf.writeInterval(expPath, 'cwTrials',...
      block.events.newTrialTimes(:)-block.events.expStartTimes,...
      block.events.endTrialTimes(:)-block.events.expStartTimes, [], []);
    copyfile(fullfile(expPath, 'cwTrials.intervals.npy'), newpath);
    
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
      copyfile(fullfile(expPath, 'cwTrials.inclTrials.npy'), newpath);
      writeNPY(repNum, fullfile(expPath, 'cwTrials.repNum.npy'));
      copyfile(fullfile(expPath, 'cwTrials.repNum.npy'), newpath);
    catch
      warning('Saving repeatNums failed')
    end
    
    %% Write wheel times, position and velocity
    wheelValues = block.inputs.wheelValues(:)-block.inputs.wheelValues(1);
    switch lower(block.rigName)
      case {'zrig1', 'zrig2', 'zrig3', 'zrig4',...
          'zredone', 'zredtwo', 'zredthree', 'zgreyfour'} % spesh
        encRes = 1024;
      case {'zym1', 'zym2', 'zym3'}
        encRes = 360;
      otherwise
        encRes = 1024;
    end
    wheelValues = wheelValues*(3.1*2*pi/(4*encRes));
    wheelTimes = block.inputs.wheelTimes(:);
    wheelTimes = wheelTimes-block.events.expStartTimes;
    
    try
      alf.writeTimeseries(expPath, 'Wheel', wheelTimes, [], []);
      copyfile(fullfile(expPath, 'Wheel.timestamps.npy'), newpath);
      writeNPY(wheelValues, fullfile(expPath, 'Wheel.position.npy'));
      copyfile(fullfile(expPath, 'Wheel.position.npy'), newpath);
      writeNPY(wheelValues./wheelTimes, fullfile(expPath, 'Wheel.velocity.npy'));
      copyfile(fullfile(expPath, 'Wheel.velocity.npy'), newpath);
    catch
      warning('Failed to write wheel values')
    end
    
    %% Registration
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
    
    try
      % Register them to Alyx
      ai.registerALF(expPath, url);
    catch ex
      warning(ex.identifier, 'Failed to register alf files: %s.', ex.message);
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
end