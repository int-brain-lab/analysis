function behavior_learning
% MOUSE_NAME = {'4577','4579','4580','4573','4576','4581','4619'};
MOUSE_NAME = {'4573','4576','4581','4619'};

% MOUSE_NAME = {'4581'};
numMouse = numel(MOUSE_NAME);

figure;
ax = axes('TickDir','Out','LineWidth', 0.75, 'xlim', [0 60], 'xtick', [0:20:60], 'ylim', [40 100], 'ytick', [50 75 100]); hold on
plot([0 60], [50 50], 'k:', 'linewidth', 0.75); % 50%-line
cmap = linspecer(numMouse); %whatever colormap
% cmap = [1 0 0;1 0 0;1 0 0;zeros(4,3)];
for iM = 1:numMouse
    [fracHit] = behavior_learning_each_mouse(MOUSE_NAME{iM});
    plot(ax, 1:numel(fracHit), fracHit*100, 'color',cmap(iM,:), 'linewidth', 2);
end

function [fracHit] = behavior_learning_each_mouse(mouse_name)

%OUTPUT:
%fracHit fraction response right numFile x numContrast 

NUM_TRIAL_THRESHOLD = 5; %sessions with trials fewer than this number will be ignored
DATA_DIR = 'E:\Masa_Work\Google Drive\IBL\Rigbox_repository'; %change to a directory where your data is saved
%assume DATA_DIR has Repository structure \mousename\date\session\files

%get a list of fullfilenames from mouse name
fileName = {}; %to fill in below
protocolName = {}; %to fill in below

dirs_date = dir([DATA_DIR filesep mouse_name filesep '201*']);
for iDd = 1:numel(dirs_date)
    dirs_sess = dir([DATA_DIR filesep mouse_name filesep dirs_date(iDd).name]);
    numDirSess = numel(dirs_sess);
    for iDs = 1:numDirSess
        if isnan(str2double(dirs_sess(iDs).name))
            %real data should have a number
            continue
        end
        
        file = dir([DATA_DIR filesep mouse_name filesep dirs_date(iDd).name filesep dirs_sess(iDs).name filesep '*_Block.mat']);
                
        %get protocol
        load([DATA_DIR filesep mouse_name filesep dirs_date(iDd).name filesep dirs_sess(iDs).name filesep file.name(1:end-9) 'parameters.mat']);
        [z,z,z,z,z,z,splits] = regexp(parameters.defFunction, filesep);
        defFunction = splits{end};
        if strcmp(defFunction, 'advancedChoiceWorld.m')
            warning('Not ready for advancedChoiceWorld')
            %for example, what happens if no response within response window??
            continue
        end
        
        protocolName = [protocolName; {defFunction}];
        fileName = [fileName; ...
            {[DATA_DIR filesep mouse_name filesep dirs_date(iDd).name filesep dirs_sess(iDs).name filesep file.name]}, ...
            ];        
    end
end

%main part
contrastList = [-1 -0.5 0.5 1]; %only high contrast
numFile = numel(fileName);
fracHit = NaN(numFile,1);
numTr = NaN(numFile,1);

for iF = 1:numFile
% for iF = 1:5
    load(fileName{iF})
    [~,~,~,~,~,~,splits] = regexp(fileName{iF}, filesep);
    sessoin = splits{end}(1:end-15);
    
    numTr(iF) = numel(block.events.endTrialValues);
    side = block.events.trialSideValues(1:numTr(iF)); %side of the stimulus
    if strcmp(protocolName{iF}, 'vanillaChoiceworldWithRew.m') %champalimaud
        hit = block.events.hitValues(2:numTr(iF)+1);
    elseif strcmp(protocolName{iF}, 'basicChoiceworld.m')
        hit = block.events.hitValues(1:numTr(iF));
    else
        error('check');
    end
    if isfield(block.events, 'repeatTrialValues')
        repeat = block.events.repeatTrialValues(1:numTr(iF));
    elseif isfield(block.events, 'repeatNumValues')
        repeat = double(block.events.repeatNumValues(1:numTr(iF))>1);
    else
        error('check')
    end
    contrast = block.events.trialContrastValues(1:numTr(iF));       
    miss = 1 - hit;
    
    %bug (only for Champalimaud?)
    if ismember(sessoin(1:end-2), {'2018-02-01','2018-02-02','2018-02-05','2018-02-06','2018-02-07'})
        %bug when we fix bug of stim-on during quiescent
        % repeat trial is no longer saved properly
        warning('Only for champalimaud? Anyway shouldn''t hurt')
        repeat = zeros(1, numTr(iF));
        repeat(find(miss(1:end-1)==1 & contrast(1:end-1)>=0.5)+1) = 1;
        %
    end
    
    sidedContrast = contrast.*side;
    
    fracHit(iF) = mean(hit(ismember(sidedContrast, contrastList) & ~repeat));
end

%take out sessions less than 5 trials
boolSmallNumberOfTrial = (numTr<NUM_TRIAL_THRESHOLD);
fracHit(boolSmallNumberOfTrial) = [];