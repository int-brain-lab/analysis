function plotShaping_perBatch()

% grab all the data that's on Drive
addpath('~/Desktop/code/npy-matlab//');
addpath(genpath('\\NEW-9SE8HAULSQE\Users\IBL_Master\Documents\IBLData_Shared\code\tmp_analysis_matlab'));
addpath(genpath('C:\Users\Tony\Documents\Github\tmp_analysis_matlab'));
addpath(genpath('/Users/urai/Documents/code/tmp_analysis_matlab'));
addpath(genpath('/Users/urai/Documents/code/npy-matlab'));
assert(exist('readNPY', 'file') > 0, ...
    'readNPY must be on your Matlab path, get it at github.com/kwikteam/npy-matlab');
addpath('~/Documents/code/gramm/');

% GOOGLE DRIVE IS ALWAYS UNDER HOME
if ispc,
    usr = getenv('USERNAME');
    homedir = getenv('USERPROFILE');
    datapath = fullfile(homedir, 'Google Drive');
elseif ismac,
    usr = getenv('USER');
    homedir = getenv('HOME');
    datapath = fullfile(homedir, 'Google Drive', 'IBL_DATA_SHARE');
end

% batches(1).name = {'basicChoiceWorld'};
% batches(1).mice = {'Axon', 'Myelin', 'Mouse1', 'Mouse2', 'M5', 'M6', 'M7', 'Arthur'};
% 
% batches(end+1).name = {'shapingAki'};
% batches(end).mice = {'CK_1', 'CK_2', 'CK_4', 'IBL_1', 'IBL_2', 'IBL_3', 'IBL_4', ...
%     'IBL_5', 'IBL_6', 'IBL_7', 'IBL_8', 'IBL_9', 'IBL_10', 'IBL_11', 'IBL_12'};
% 
% batches(end+1).name = {'lateralOrientation_openLoop'};
% batches(end).mice = {'IBL_23', 'IBL_24', 'IBL_25', 'IBL_26', 'IBL_27'};
% 
% batches(end+1).name = {'originalPhasing_centralOrientation'};
% batches(end).mice = {'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17'};

% batches(end+1).name = {'centralOrientation_Aoki'};
% batches(end).mice = {'IBL_18', 'IBL_19', 'IBL_20', 'IBL_21', 'IBL_22'};
% 
% batches(end+1).name = {'centralOrientation_Glickfeld'};
% batches(end).mice = {'IBL_28', 'IBL_29', 'IBL_30', 'IBL_31', 'IBL_32'};

batches(1).name = {'choiceWorld'};
batches(1).mice = {'IBL_33', 'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37'};

batches(end+1).name = {'choiceWorld_v2'};
batches(end).mice = {'IBL_2b', 'IBL_4b', 'IBL_5b', 'IBL_7b',  'IBL_9b'};

% batches(end+1).name = {'choiceWorld_orientation'};
% batches(end).mice = {'IBL_38', 'IBL_39', 'IBL_40', 'IBL_41', 'IBL_42'};

batches(end+1).name = {'choiceWorld_1screen'};
batches(end).mice = {'IBL_1b', 'IBL_3b', 'IBL_6b', 'IBL_8b',  'IBL_10b'};

batches(end+1).name = {'choiceWorld_bigstim'};
batches(end).mice = {'IBL_11b', 'IBL_12b'};

set(groot, 'defaultaxesfontsize', 10, ...
    'DefaultAxesTickLength', [0.01 0.0125], 'DefaultFigureWindowStyle', 'normal');
close all; figure;

warning('error', 'stats:glmfit:IterationLimit');
warning('error', 'stats:glmfit:PerfectSeparation');

% ============================================================== %
% SHOW PERFORMANCE OVER TIME FOR ALL MICE OVERLAID PER DAYS
% ============================================================== %
if 0,
for bidx = length(batches):-1:1,
    
    close all;
    % colormap(cbrewer('qual', 'Pastel1', length(batches(bidx).mice)));
    
    % load data
    data = readAlf_allData(datapath, batches(bidx).mice);
    
    % missed trials, don't count in accuracy
    data.correct(data.response == 0) = NaN;
    
    % only use blocks where stimuli were interleaved
    data.response(data.blockType < 2) = NaN;
    
    % remove trials that were missed
    data.response(data.response == 0) = NaN;
    
    % average performance and number of trials per day
    [gr, mice, days] = findgroups(data.animal, data.dayidx);
    trlcnt = @(x) sum(~isnan(x));
    summary = struct2table(struct('mice', {splitapply(@unique, data.animal, gr)}, ...
        'days', splitapply(@unique, data.dayidx, gr), ...
        'date', splitapply(@unique, data.date, gr), ...
        'accuracy', splitapply(@nanmean, 100*data.correct, gr), ...
        'ntrials', splitapply(trlcnt, data.response, gr), ...
        'bias', splitapply(@nanmean, data.response, gr), ...
        'dprime', splitapply(@dprime, sign(data.signedContrast), data.response, gr), ...
        'stim_slope', splitapply(@stimWeights, data.signedContrast, data.response, gr)));
    
    plotFlds = {'accuracy', 'ntrials', 'stim_slope', 'dprime'};
    for p = 1:length(plotFlds),
        subplot(2,2,p);
        
        % matrix of mice by days
        mat = unstack(summary(:, {plotFlds{p}, 'mice', 'days'}), plotFlds{p}, 'mice');
        mat = sortrows(mat, 'days');
        
        % plot summary stats - dprime
        hold on;
        ph = plot(mat{:, 1}, mat{:, 2:end}, '.-', 'linewidth', 0.5);
        plot(mat{:, 1}, nanmean(mat{:, 2:end}, 2), 'k-', 'linewidth', 1.5);
        set(gca, 'xtick', 1:1:max(summary.days), 'xlim', [0.5 max(summary.days)]);
        
        if max(summary.days) > 14, set(gca, 'xtick', 1:7:max(summary.days)); end
        switch plotFlds{p}
            case 'accuracy'
                ylabel('Accuracy (%)');
                r = refline(0, 50); r.Color = 'k';
            case 'ntrials'
                ylabel('# Trials');
            case 'stim_slope'
                ylabel('\beta_{stimulus}');
                r = refline(0, 0); r.Color = 'k';
            case 'dprime'
                ylabel('d''');
                r = refline(0, 0); r.Color = 'k';
            otherwise
                ylabel(regexprep(plotFlds{p}, '_', ' '));
        end
        axis tight; % offsetAxes;
        if p == 1,
            lg = legend(ph, batches(bidx).mice, ...
                'interpreter', 'none', 'box', 'off', 'location', 'northwest');
        end
    end
    
    suplabel('Days', 'x');
    suplabel(sprintf('Batch %d, %s', bidx, regexprep(batches(bidx).name{1}, '_', ' ')), 't');
    % print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', sprintf('shapingSummary_batch%d.pdf', bidx)));
    print(gcf, '-dpng', fullfile(datapath, 'CSHL', 'figures', sprintf('shapingSummary_batch%d.png', bidx)));
    
end
end

% ================================================ %
% ONE SUMMARY PLOT WITH ALL DATA
%% ================================================ %

clear alldata data;
for bidx = length(batches):-1:1,
    tmpdat = readAlf_allData(datapath, batches(bidx).mice);
    tmpdat{:, 'task'} = {sprintf('%s, d%d-%d', batches(bidx).name{1}, max(tmpdat.dayidx)-3, max(tmpdat.dayidx))};
    alldata{bidx} = tmpdat;
end

data = cat(1, alldata{:});
data.correct(data.response == 0)    = NaN;
data.response(data.blockType < 2)   = NaN;
data.response(data.response == 0)   = NaN;

% define a number of handy gramm commands
custom_psychometric = @(gramm_obj) gramm_obj.stat_fit('fun', @(a,b,g,l,x) g+(1-g-l) * (1./(1+exp(- ( a + b.*x )))),...
    'StartPoint', [0 0.1 0.1 0.1], 'geom', 'line', 'disp_fit', false, 'fullrange', false);

% PLOT
close;
g = gramm('x', data.signedContrast, 'y', (data.response > 0), 'color', ones(size(data.animal)), 'subset', data.dayidx_rev >-4);
g.set_names('x', 'Signed contrast (%)', 'y', 'P(rightwards)');
g.facet_wrap(data.task, 'ncols', 3); % each batch separately
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
g.set_color_options('map', zeros(max(data.dayidx), 3)); % black
g.no_legend;
custom_psychometric(g);
g.draw()

% overlay individual animals
g.update('color', data.animal);
g.set_color_options('map', repmat(linspecer(8, 'qualitative'), 5, 1));
g.stat_summary('type', 'bootci', 'geom', 'errorbar');
g.stat_summary('type', 'sem', 'geom', 'point');
% g.stat_summary('type', 'sem', 'geom', 'line');
custom_psychometric(g);
g.axe_property('ylim', [0 1], 'xlim', [-105 100]);
g.set_text_options('facet_scaling', 1, 'title_scaling', 1, 'base_size', 10);
g.draw()

% save
g.export('file_name', fullfile(datapath, 'CSHL', 'figures', 'allCohorts_comparison.pdf'))
print(gcf, '-dpdf', fullfile(datapath, 'CSHL', 'figures', 'allCohorts_comparison.pdf'));


end

function s = stimWeights(stim, resp)

prevstim    = circshift(sign(stim), 1);
prevresp    = circshift(resp, 1);
prevcorrect = (prevstim == prevresp);

prevresp_success    = prevresp;
prevresp_failure    = prevresp;
prevresp_success(prevcorrect == 0) = 0;
prevresp_failure(prevcorrect == 1 | isnan(prevresp)) = 0;

designM = [stim / 100, prevresp_success, prevresp_failure];

try % only do this when there are no weird values
    b = glmfit(designM, (resp > 0), 'binomial');
    s = b(2);
catch
    s = NaN;
end

end

function [dprime, crit] = dprime(stim, resp)

% if there was only one stimulus class shown, this whole thing doesn't make
% sense
if length(unique(stim(~isnan(stim)))) == 1,
    dprime = NaN;
    return;
end
    
% use only 2 identities, however this is coded
stim(stim~=1) = -1;
resp(resp~=1) = -1;

% compute proportions
Phit = length(find(stim ==  1 & resp == 1)) / length(find(stim == 1));
Pfa  = length(find(stim == -1 & resp == 1)) / length(find(stim == -1));

% correct for 100% or 0% values, will lead to Inf norminv output
if Phit > 0.999;     Phit = 0.999;
elseif Phit < 0.001; Phit = .001; end
if Pfa < 0.001;      Pfa = 0.001;
elseif Pfa > 0.999,  Pfa = 0.999; end

% compute dprime and criterion
dprime = norminv(Phit) - norminv(Pfa);
crit   = -.5 * (norminv(Phit) + norminv(Pfa));

end
