function overallBias_perRig_CSHL

rig(1).name = {'StimPC1'};
rig(1).mice = {'IBL_11', 'IBL_12', 'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17'};

rig(2).name = {'StimPC2'};
rig(2).mice = {'IBL_1', 'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10'};

rig(3).name = {'ChrisStimPC'};
rig(3).mice = {'IBL_2', 'IBL_4', 'IBL_5', 'IBL_7', 'IBL_33', ...
    'IBL_34', 'IBL_35', 'IBL_36', 'IBL_37'};

for r = 1:length(rig),
    
    data =  readAlf_allData(fullfile(getenv('HOME'), 'Google Drive', 'IBL_DATA_SHARE'), rig(r).mice);
    
    [gr, sj, days] = findgroups(data.animal, data.dayidx);
    bias           = splitapply(@nanmean, data.response, gr);
    dat            = array2table([days bias], 'variablenames', {'day', 'bias'});
    dat.animal     = sj;
    
    subplot(3,3,r);
    gscatter(dat.day,dat.bias,dat.animal, [], [], [], 0);
    xlabel('Days'); ylabel('Bias');
    title(rig(r).name);
    box off; offsetAxes;
    
    subplot(3,3,r+3);
    histogram(dat.bias, 'edgecolor', 'none');
    vline(mean(bias));
    xlabel('Bias');
    box off; offsetAxes;
    
end

foldername = fullfile('~', 'Google Drive', 'Rig building WG', 'DataFigures', 'BehaviourData_Weekly', '2018-10-09');
print(gcf, '-dpdf', fullfile(foldername, 'CSHL_bias.pdf'));



