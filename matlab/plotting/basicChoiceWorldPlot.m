function basicChoiceWorldPlot(subjects)
% BASICCHOICEWORLDPLOT Generate plots for multiple subject's performance
% over all sessions.  

% For ease of processing, make sure that subject is a cell
subjects = ensureCell(subjects);
% Create our figure
figure('Name', 'basicChoiceWorldPlot', 'NumberTitle', 'Off');
% Create the subplots for each subject
ax = cell(1,length(subjects));
for i = 1:length(subjects)
  ax{i} = subplot(2,ceil(length(subjects)/2),i);
end
% Pass each subject and axes handle to plotting function
cellfun(@(s,a)plotSessionPerformance(s,a),subjects,ax, 'UniformOutput', false);
end

function performance = plotSessionPerformance(subject, axes_handle)
% PLOTSESSIONPERFORMANCE Plots the fraction of left-ward reponses for each
% contrast in basic-, advanced and (for legacy) vanillaChoiceWorld.  After
% 2018-01-17 the responce direction of the mouse is recorded (before that
% we knew only whether it was a correct response).  This is not yet
% implemented in this code, so we're just inferring the response based on
% whether it was correct.  Function outputs a matrix where m is the session
% and n is the contrast.  The contrast values are hard-coded below.

[expRef, ~] = dat.listExps(subject); % List the experiments for subject
% Find the block files
filelist = mapToCell(@(r) dat.expFilePath(r, 'block', 'master'),expRef);
% Filter out those that don't exist
existfiles = cell2mat(mapToCell(@(l) file.exists(l), filelist));
% Load the blocks
allBlocks = cellfun(@load,filelist(existfiles));
% This will allow use to process blocks based on experiment definition.
expDef = arrayfun(@(b){b.block.expDef},allBlocks);
[~, expDef] = cellfun(@fileparts, expDef, 'UniformOutput', false); 

% Extract the events struct from the blocks
events = arrayfun(@(b){b.block.events},allBlocks); 
events = catStructs(events); % Concatenate them
% Define our complete contrast set
uniqueContrasts = [-1 -0.5 -0.25 -0.125 -0.06 0 0.06 0.125 0.25 0.5 1];
performance = nan(length(events), length(uniqueContrasts)); % Initialize performance matrix
invalid = zeros(length(events),1); % Initialize invalid block logical array
totalTrials = nan(length(events), 1);
for s = 1:length(events)
  % Check that the session has at least five trials...
  if ~isfield(events(s), 'newTrialValues')||...
      isempty(events(s).newTrialValues)||...
      length(events(s).newTrialValues)<5
    invalid(s) = true; %... if not mark as invalid
    continue
  end
  % Extract our performance data
  switch expDef{s}
    case 'advancedChoiceWorld'
      correct = [events(s).feedbackValues];
      contrasts = [events(s).contrastValues];
      repeatTrials = [events(s).repeatNumValues] > 1;
      trialSide = sign(contrasts);
    case {'vaillaChoiceworld' 'basicChoiceworld'}
      contrasts = times([events(s).trialContrastValues]',[events(s).trialSideValues]');
      % 'repeatTrial' was recently removed: it is redundant with repeatNum
      if ~isfield(events(s), 'repeatTrialValues')||isempty(events(s).repeatTrialValues)
        repeatTrials = events(s).repeatNumValues > 1;
      else
        repeatTrials = events(s).repeatTrialValues;
      end
      trialSide = [events(s).trialSideValues];
      correct = [events(s).hitValues];
    otherwise % Ignore any other experiments
      invalid(s) = true; % Mark session as invalid
      continue
  end
  
  % Trim incomplete trials
  if length(contrasts)~=length(correct)
    trialSide = trialSide(1:length(correct));
    contrasts = contrasts(1:length(correct));
    repeatTrials = repeatTrials(1:length(correct));
  end
  % Infer whether response was leftward
  left = (trialSide==-1&correct)|(trialSide==1&~correct);
  totalTrials(s) = length(contrasts);

  % Build performance matrix
  for c = 1:length(uniqueContrasts)
    % If no contrasts of this value appeared, move on leaving them as NaN
    if ~any(contrasts==uniqueContrasts(c)); continue; end
    performance(s,c) = sum(contrasts==uniqueContrasts(c)&left'&~repeatTrials')/...
        sum(contrasts==uniqueContrasts(c)&~repeatTrials');
  end
end
yyaxis(axes_handle,'left')
% Remove invalid sessions from matrix
performance = performance(~invalid,:);
totalTrials = totalTrials(~invalid);
% Get the min and max performance values
cm = colormap(axes_handle, redblue);
color_step = 1/size(cm,1);
imagesc(axes_handle,performance);
colormap(axes_handle, [[0.5,0.5,0.5]; cm]); % Add NaN colour
caxis(axes_handle, [-color_step 1]);
% change Y limit for colorbar to avoid showing NaN color
c = colorbar(axes_handle);
ylim(c,[0 1])

% Session Y-Axis
ylabel(c,'Go left (frac)');
xlabel(axes_handle,'Contrast (%)');
ylabel(axes_handle,'Session #');
% set(axes_handle, 'XTick', 1:length(uniqueContrasts));
% set(axes_handle,'XTickLabel',strsplit(num2str(uniqueContrasts)));
set(axes_handle, 'XTick', [1, 3, 6, 9, 11]);
set(axes_handle,'XTickLabel',{'-100', '-25', '0', '25', '100'});
set(axes_handle,'YTick',1:length(events));
title(axes_handle, subject);
yl = ylim(axes_handle);

% TrialNum Y-Axis
hold(axes_handle,'on')
yyaxis(axes_handle,'right')
% plot(repmat(110, length(totalTrials), 1), totalTrials);
% plot(repmat(110, length(totalTrials), 1), 1:length(events));
ylim(axes_handle, yl)
set(axes_handle, 'YTick', 1:length(events));
set(axes_handle,'YTickLabel',num2str(flipud(totalTrials)));
ylabel(axes_handle,'Total trials');
hold(axes_handle,'off')
end