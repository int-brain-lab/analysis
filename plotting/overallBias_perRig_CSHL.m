function overallBias_perRig_CSHL

rig(1).name = {'StimPC1'};
rig(1).mice = {'IBL_11', 'IBL_12', 'IBL_13', 'IBL_14', 'IBL_15', 'IBL_16', 'IBL_17'};

rig(2).name = {'StimPC2'};
rig(2).mice = {'IBL_1', 'IBL_3', 'IBL_6', 'IBL_8', 'IBL_10'};

for r = 1:length(rig),
    
   data =  readAlf_allData(fullfile(getenv('HOME'), 'Google Drive', 'IBL_DATA_SHARE'), rig(r).mice);
    
    [gr, sj, days] = findgroups(data.animal, data.dayidx);
    bias           = splitapply(@nanmean, data.response, gr);
    dat            = array2table([days bias], 'variablenames', {'day', 'bias'});
    dat.animal     = sj;
    
    subplot(3,3,r);
    gscatter(dat.day,dat.bias,dat.animal, [], [], [], 0);    
    xlabel('Days'); ylabel('Bias');
    
    subplot(3,3,r+3);
    histogram(dat.bias);
    vline(mean(bias));
    xlabel('Bias'); 
    
end


