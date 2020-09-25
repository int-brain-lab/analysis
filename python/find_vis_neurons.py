"""
The International Brain Laboratory
Christopher S Krasniak, CSHL, 2020-09-17
"""

import pandas as pd
import numpy as np
import datajoint as dj
from ibl_pipeline import subject, acquisition, behavior, histology
from brainbox.singlecell import singlecell
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
import seaborn as sns
from ibllib.atlas import AllenAtlas

ba = AllenAtlas()
ephys = dj.create_virtual_module('ephys', 'ibl_ephys')


def CSK_PSTH(trials, pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025):
    """
    A PSTH function that relies on the datajoint data format returned by quieries containing ephys.AlignedTrialSpikes 
    trials: (array-like) trial_spike_times returned by the above mentrioned query (event is always at 0 seconds) 
    others are PSTH parameters
    
    """
    n_bins = round((pre_time+post_time)/bin_size)
    event_idx = round(pre_time/bin_size)
    all_counts = np.zeros([n_bins,len(trials)])
    
    if smoothing > 0: # build gaussian window for smoothing
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
#         window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
#         binned_spikes_conv = np.copy(binned_spikes)
        

    for i in range(len(trials)):        
        counts, bins = np.histogram(trials.trial_spike_times.iloc[i],bins=n_bins,range=[-pre_time,post_time])
        all_counts[:,i] = counts
    PSTH = all_counts.mean(axis=1)
    smooth_PSTH = convolve(PSTH,window,mode='same')
    return smooth_PSTH, event_idx


def binned_spike_rate(trials, pre_time=0, post_time=0.5):
    """
    relies on the datajoint data format returned by quieries containing ephys.AlignedTrialSpikes
    trials: (array-like) trial_spike_times returned by the above mentrioned query (event is always at 0 seconds) 
    pre_time: beginning of the time window to use
    post_time: end of the time window to use
    returns: array of spike counts for each trial
    """
    binned_spk_counts = np.array([])
    for i in range(len(trials)):
        st = trials.trial_spike_times.iloc[i]
        in_win = len(st[np.logical_and(st > pre_time, st < post_time)])
        np.append(binned_spk_counts,in_win)
    return binned_spk_counts


def vis_PSTH_from_dj(data, spike_times, cluster_ids, event, contrasts, alpha=.01, pre_time=0,
        post_time=0.2, bin_size=0.02, smoothing=0.01, FR_cutoff = .1):
    """
    function to find if a neuron is visual based on its PSTH. compares the PSTH of a cluster for left, right, or 
    no visual stimulus, if there is a significant difference between both left and right and zero contrast, then 
    considered visual (using alpha value above.)
    spike_times: (array-like) trial_spike_times returned by the above mentrioned query (event is always at 0 seconds) 
    cluster_ids: (array-like) len of spike_times, ID of the cluster
    event : event alligning to
    contrasts : signed contrast of the stimulus
    alpha: significance level for K-S test
    others: PSTH params
    
    returns: 
        a dataframe of significant clusters, with cluster_id, the PSTH values for left, right, and zero stimuli
             and the allen label for the brain_area that the cluster is located in
        a list of all the brain areas of all the clusters, one label per area, used to find what proportion of 
        neurons in an area are

    """
        
    #### collapse PSTH into a single value for a few different time windows then just do a t-test betwee those
    ### should see visctx in first 0-40 or 0-80ms, then motor and decision like places in the later times like 100+ms
    
    sigClusters = pd.DataFrame(columns=['cluster_id','left_PSTH', 'right_PSTH','zero_PSTH', 'brain_area'])
    u_clusters = np.unique(cluster_ids)
    u_contrasts = np.unique(contrasts)
    p1s = []
    p2s = []
    count = 0
    sig_clu = []
    sig_left = []
    sig_right = []
    sig_zero = []
    sig_region = []
    all_region = []

    for cluster in u_clusters: # loop over clusters to create PSTHs for each
        clust_idx = np.where(cluster_ids == cluster)[0]
        trials = data.iloc[clust_idx]
        days = np.unique(trials.session_start_time)
        for day in days:
            day_trials = trials[trials.session_start_time==day]
            area = day_trials.acronym.iloc[0]

            # break down by stimulus side so I can see which cells have higher activity for stim vs 0 contrast
            left_trials = np.where(day_trials.signedContrast < 0)[0]
            right_trials = np.where(day_trials.signedContrast > 0)[0]
            zero_trials = np.where(day_trials.signedContrast == 0)[0]
            left_PSTH = binned_spike_rate(day_trials.iloc[left_trials], pre_time=pre_time, post_time=post_time)
            right_PSTH = binned_spike_rate(day_trials.iloc[right_trials], pre_time=pre_time, post_time=post_time)
            zero_PSTH = binned_spike_rate(day_trials.iloc[zero_trials], pre_time=pre_time, post_time=post_time)
            if len(left_PSTH)>5 or len(right_PSTH) >5 or len(zero_PSTH)>5:
            
                _, p1 = ttest_ind(left_PSTH, zero_PSTH)  # Kolmogorov-Smirnov test to see if either is more active than zero contrast
                _, p2 = ttest_ind(right_PSTH, zero_PSTH)
                _, p3 = ttest_ind(right_PSTH, left_PSTH)
                if p3 < alpha:  # and (np.any(left_PSTH>FR_cutoff) or np.any(right_PSTH>FR_cutoff)):
                    sig_clu.append(cluster)
                    sig_left.append(left_PSTH)
                    sig_right.append(right_PSTH)
                    sig_zero.append(zero_PSTH)
                    sig_region.append(area)
                
            #if (np.any(left_PSTH>FR_cutoff) or np.any(right_PSTH>FR_cutoff)):
            all_region.append(area)  # save the name of all regions, this way a count of the regions will give a number of all clusters recorded in that region, from this I can reconstruct the frac significant per brain area
    sigClusters.cluster_id = sig_clu
    sigClusters.left_PSTH = sig_left
    sigClusters.right_PSTH = sig_right
    sigClusters.zero_PSTH = sig_zero
    # brain_areas = group_cortical(sig_region)
    sigClusters.brain_area = sig_region
    
    return sigClusters, all_region

def group_cortical(areas):
    """
    input areas: a list of brain area names from the allen ontology
    output: that same list with the layers removed from all cortical areas
    """
    exclude = ['1','2/3','4','5','6a','6b']
    for i,area in enumerate(areas):
        for num in exclude:
            if area == 'CA1':
                continue
            elif num in area:
                new_name = area.replace(num,'')
                areas[i]=new_name
    return areas

    # find all subjects with ephys_aligned_histology
hist_subs = histology.ProbeTrajectory * subject.Subject & 'insertion_data_source = "Histology track"'
subs = hist_subs.fetch('subject_nickname')
subs = np.unique(subs)

#get PSTHs for all subjects with histology

post_times = np.arange(.05,.5,.05)
sigs2 = []
all_all_areas2 = []

for sub in subs:
    sigs = []   
    all_all_areas = []
    count = 0
    print('working on subject: {}'.format(sub))
    temp = (ephys.AlignedTrialSpikes & 'event = "stim on"') * (subject.Subject & 'subject_nickname = "{}"'.format(sub)) * behavior.TrialSet.Trial * (ephys.GoodCluster & 'is_good = "1"') * histology.ClusterBrainRegion & 'insertion_data_source = "Histology track"' 
    stimons = temp & 'event= "stim on"'
    if len(stimons) > 0:
        data = pd.DataFrame(stimons.fetch('subject_uuid', 'session_start_time', 'probe_idx', 'cluster_id',
            'trial_id', 'event', 'trial_spike_times', 'trial_spikes_ts',
            'subject_nickname','trial_start_time', 'trial_end_time','trial_stim_on_time',
            'trial_stim_contrast_left', 'trial_stim_contrast_right','acronym', as_dict=True))
        data['signedContrast'] = data.trial_stim_contrast_left - data.trial_stim_contrast_right
        for post_time in post_times:
            sig, all_areas = vis_PSTH_from_dj(data, data.trial_spike_times, data.cluster_id, data.event, data.signedContrast,post_time=post_time, alpha = .001)
            sigs.append(sig)
            all_all_areas.append(all_areas)
    sigs2.append(sigs)
    all_all_areas2.append(all_all_areas)
        

# concatonate and find proportion of visual neurons in each area
sig_df = pd.concat(sigs)
all_ba = [y for x in all_all_areas for y in x]
u,uc = np.unique(sig_df.brain_area,return_counts=True)

proportion_vis = pd.DataFrame(columns=['area','prop_vis','num_neur'])
prop = []
areas = []
neurs = []
for i,area in enumerate(u):
    num = np.sum(np.array(all_ba) == area)
    prop.append(uc[i]/num)
    areas.append(area)
    neurs.append(num)
proportion_vis['area']=areas
proportion_vis['prop_vis'] = prop
proportion_vis['num_neur'] = neurs
area_new = []
plot_areas = ['Visp','VISl','VISpm','VISrl','VISa','RSP','ACA','MOs','PL','ILA','ORB','Mop','SSp','DG','CA3','CA1','POST','SUB','OLF','BLA','CP','GPe','SNr','ACB','LS','LGd','LP','LD','POL','MD','VPL','PO','VPM','RT','MG','SCs','SCm','MRN','APN','APN','PAG','ZI','CBX']

layers = ['1','2/3','4','5','6','6a','6b']
for i, area in enumerate(proportion_vis.area):
    area_id = ba.regions.id[ba.regions.acronym==area] 
    area_anc = ba.regions.ancestors(area_id)
    if any(area_anc.acronym == 'fiber tracts'):
        area_app = area_anc.acronym[1]
    elif any([plot_area in area_anc.acronym for plot_area in plot_areas]):
        area_app = np.array(plot_areas)[[plot_area in area_anc.acronym for plot_area in plot_areas]][0]
    elif any(area_anc.acronym == 'Isocortex'):
        area_app = area_anc.acronym[7]
    elif max(area_anc.level) > 3:
    # print(area_anc)
        area_app = area_anc.acronym[3] 
    else:
        area_app = area 
    area_new.append(area_app)
proportion_vis.area = area_new
u_areas = np.unique(proportion_vis.area)
u_prop_vis = pd.DataFrame(columns=proportion_vis.columns)
cnt = 0
new_u_areas = []
new_vises = []
new_nums = []
for area in u_areas:
    new_num=0
    new_vis = 0
    for i in range(len(proportion_vis)):
        if proportion_vis.area.iloc[i] == area:
            new_vis += proportion_vis.prop_vis.iloc[i]*proportion_vis.num_neur.iloc[i]
            new_num += proportion_vis.num_neur.iloc[i]
    new_u_areas.append(area)
    new_vises.append(new_vis/new_num)
    new_nums.append(new_num)
u_prop_vis.area = new_u_areas
u_prop_vis.prop_vis = new_vises
u_prop_vis.num_neur = new_num

        
# plot barplot for each neuron
sns.barplot(data=u_prop_vis,y='prop_vis',x='area')
plt.show(block=False)