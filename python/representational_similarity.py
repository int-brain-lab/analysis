# -*- coding: utf-8 -*-
#
# Representational similarity analysis
#
# Author: Naoki Hiratani (N.Hiratani@gmail.com)
#
# Trial types were merged into following 20 trial types
# (L/R block, 5 stimulus contrast (very high-Left, high-Left, low-Left, zero, low-Right, high-Right, very-high Right), L/R movement)
#  - ({high/veryhigh-Left, R-move} and {High/veryhigh-Right, L-move})
#
#
from math import *

import sys
import alf.io
from oneibl.one import ONE
from ibllib.misc import pprint
import numpy as np
import scipy.stats as scist
from os import path

one = ONE()
time_ranges = [['2019-12-01', '2020-03-31']]

class TrialData():
    def __init__(trials, eid):
        trials.contrastTypes = [0.0, 0.0625, 0.125, 0.25, 1.0]
        trials.ctlen = len(trials.contrastTypes)
        
        trials.feedbackType = one.load(eid, dataset_types=['trials.feedbackType'])
        trials.total_trial_count = len(trials.feedbackType[0])
        trials.average_performance = 0.5 + 0.5*np.mean(trials.feedbackType)

    def load_trial_data(trials, eid):
        trials.goCue_times, trials.stimOn_times, trials.feedback_times, trials.contrastLeft, trials.contrastRight, trials.probLeft, trials.choice, trials.feedbackType = \
                            one.load(eid, dataset_types=['trials.goCue_times', 'trials.stimOn_times', 'trials.feedback_times', 'trials.contrastLeft', 'trials.contrastRight', 'trials.probabilityLeft', 'trials.choice', 'trials.feedbackType'])
        
        trials.total_trial_count = len(trials.goCue_times)
        #supplementing the effective feedback time
        for tridx in range( len(trials.feedback_times) ):
            if isnan(trials.feedback_times[tridx]):
                trials.feedback_times[tridx] = trials.stimOn_times[tridx] + 60.0

    def good_behaviour(trials, eid, min_trial_count, min_average_performance, prob_right_range):
        is_good_behaviour = True
        if (trials.total_trial_count <= min_trial_count and trials.average_performance < min_average_performance):    
            is_good_behaviour = False
            return is_good_behaviour
        else:
            trials.load_trial_data(eid)
            trials.calc_trial_types()
            blockwise_prob_right = trials.calc_prob_right()
            if (prob_right_range[0][0] < blockwise_prob_right[0] and blockwise_prob_right[0] < prob_right_range[0][1]) \
                   and (prob_right_range[1][0] < blockwise_prob_right[1] and blockwise_prob_right[1] < prob_right_range[1][1])\
                   and len(trials.feedbackType) == len(trials.feedback_times):
                is_good_behaviour = True
            else:
                is_good_behaviour = False
            return is_good_behaviour

    def calc_trial_types(trials): #20-classification by block
        ctlen = trials.ctlen
        trials.trial_types = []
        trials.trial_count = np.zeros((20))
        
        for tridx in range(trials.total_trial_count):
            trtidx = -1
            if trials.probLeft[tridx] == 0.5:
                trtidx = -1
            else:
                Rchoice = 1 if trials.choice[tridx] < 0.0 else 0
                Rblock = 1 if trials.probLeft[tridx] < 0.5 else 0
                Rtrial = 1 if isnan(trials.contrastLeft[tridx]) else 0

                sgn_contrast = 0
                if Rtrial == 0:
                    if trials.contrastLeft[tridx] == 1.0:
                        sgn_contrast = 0
                    elif trials.contrastLeft[tridx] == 0.25:
                        sgn_contrast = 1
                    elif (trials.contrastLeft[tridx] == 0.125 or trials.contrastLeft[tridx] == 0.0625):
                        sgn_contrast = 2
                    else:
                        sgn_contrast = 3
                else:
                    if (trials.contrastRight[tridx] == 1.0:
                        sgn_contrast = 6
                    elif trials.contrastRight[tridx] == 0.25):
                        sgn_contrast = 5
                    elif (trials.contrastRight[tridx] == 0.125 or trials.contrastRight[tridx] == 0.0625):
                        sgn_contrast = 4
                    else:
                        sgn_contrast = 3
                
                if Rchoice == 0: #Left-move
                    if sgn_contrast == 5 or sgn_contrast == 6: #High-right stimulus, Left move
                        trtidx = -1
                    else:
                        trtidx = 10*Rblock + sgn_contrast
                else:
                    if sgn_contrast == 0 or sgn_contrast == 1: #High-left stimulus, Right move
                        trtidx = -1
                    else:
                        trtidx = 10*Rblock + 5 + (sgn_contrast-2)

            trials.trial_types.append(trtidx);
            if trtidx > -0.5:
                trials.trial_count[trtidx] += 1
                        
    #Further selection of trials by reaction time etc
    #  response time = feedback time - stimOn time
    def trial_selection(trials, response_time_range):
        min_response_time = response_time_range[0]; max_response_time = response_time_range[1]; 

        response_times = trials.feedback_times - trials.stimOn_times
        total_trial_count = trials.total_trial_count
        trials.selected_trials = np.zeros( (total_trial_count) )
        
        within_block_tridx = 0; prev_Rblock = -1
        for tridx in range(total_trial_count):
            trial_type_tmp = int(trials.trial_types[tridx])
            if trial_type_tmp >= -0.5:
                Rblock = int(floor( trial_type_tmp/(len(trials.trial_count)/2) ))
                if Rblock != prev_Rblock:
                    within_block_tridx = 0
                prev_Rblock = Rblock; within_block_tridx += 1

                #exclude first 5 trials since the switch
                if within_block_tridx < 5: 
                    trials.trial_count[ trial_type_tmp ] -= 1
                    trials.trial_types[tridx] = -1

                #exclude a trial with very short/long response time
                elif response_times[tridx] < min_response_time or max_response_time < response_times[tridx]:
                    trials.trial_count[ trial_type_tmp ] -= 1
                    trials.trial_types[tridx] = -1
                else:
                    trials.selected_trials[tridx] = 1.0
        print( 'number of selected trials: ' + str(trials.trial_count) )

    #calculate the probability of right choice for each block-stimulus pair
    def calc_prob_right(trials): 
        bs_prob_right = [[0, 0], [0, 0]]
        bs_cnt = [[0, 0], [0, 0]]
        
        for tridx in range(trials.total_trial_count):
            if trials.trial_types[tridx] >= -0.5:
                Rblock = 1 if trials.probLeft[tridx] < 0.5 else 0
                Rtrial = 1 if isnan(trials.contrastLeft[tridx]) else 0
                Rchoice = 1 if trials.choice[tridx] < 0.0 else 0
                
                bs_cnt[Rblock][Rtrial] += 1
                if Rchoice == 1:
                    bs_prob_right[Rblock][Rtrial] += 1
                    
        for bidx in range(2):
            for sidx in range(2):
                if bs_cnt[bidx][sidx] > 0:
                    bs_prob_right[bidx][sidx] = bs_prob_right[bidx][sidx]/float(bs_cnt[bidx][sidx])
        return [0.5*(bs_prob_right[0][0] + bs_prob_right[0][1]), 0.5*(bs_prob_right[1][0] + bs_prob_right[1][1])]

    #reselection of trials based on the total population firing rate during each trial
    def trial_reselection(trials, fr_vecs):
        trials.reselected_trials = np.zeros( (trials.total_trial_count) )
        trials.new_trial_count = np.zeros( (len(trials.trial_count)) )
        for tridx in range(trials.total_trial_count):
            if trials.selected_trials[tridx] >= 0.5 and np.sum( fr_vecs[tridx] ) > 0.01:
                trials.reselected_trials[tridx] = 1.0
                trials.new_trial_count[ trials.trial_types[tridx] ] += 1

class WheelData():
    def __init__(wheel, eid):
        wheel.position, wheel.timestamps = one.load(eid, dataset_types=['wheel.position', 'wheel.timestamps'])
        wheel.velocity = wheel.calc_wheel_velocity()
    
    def calc_wheel_velocity(wheel):
        wheel_velocity = []; wheel_velocity.append(0.0);
        for widx in range( len(wheel.position)-1 ):
            wheel_velocity.append( (wheel.position[widx+1] - wheel.position[widx])/(wheel.timestamps[widx+1] - wheel.timestamps[widx]) )
        return wheel_velocity
    
    def calc_trialwise_wheel(wheel, stimOn_times, feedback_times):
        #divide the wheel information into trialwise format by using the data from
        #  stimOn_time - pre_duration < t < feedback_time
        #
        wheel.stimOn_pre_duration = 0.3 #[s]
        wheel.total_trial_count = len(stimOn_times)
        
        wheel.trial_position = []
        wheel.trial_timestamps = []
        wheel.trial_velocity = []
        for tridx in range( wheel.total_trial_count ):
            wheel.trial_position.append([])
            wheel.trial_timestamps.append([])
            wheel.trial_velocity.append([])
        
        tridx = 0
        for tsidx in range( len(wheel.timestamps) ):
            timestamp = wheel.timestamps[tsidx]
            while tridx < len(stimOn_times) - 1 and timestamp > stimOn_times[tridx+1] - wheel.stimOn_pre_duration:
                tridx += 1
            
            if stimOn_times[tridx]  - wheel.stimOn_pre_duration <= timestamp and \
                timestamp < feedback_times[tridx]:
                wheel.trial_position[tridx].append( wheel.position[tsidx] )
                wheel.trial_timestamps[tridx].append( wheel.timestamps[tsidx] )
                wheel.trial_velocity[tridx].append( wheel.velocity[tsidx] )

    def calc_movement_onset_times(wheel, stimOn_times):
        #a collection of timestamps with a significant speed (>0.5) after more than 50ms of stationary period
        speed_threshold = 0.5
        duration_threshold = 0.05 #[s]
            
        wheel.movement_onset_times = []
        wheel.first_movement_onset_times = np.zeros( (wheel.total_trial_count) ) #FMOT
        wheel.last_movement_onset_times = np.zeros( (wheel.total_trial_count) ) #LMOT
        wheel.movement_onset_counts = np.zeros( (wheel.total_trial_count) )
            
        for tridx in range(len(wheel.trial_timestamps)):
            wheel.movement_onset_times.append([])
            cm_dur = 0.0; #continous stationary duration
            for tpidx in range( len(wheel.trial_timestamps[tridx]) ):
                t = wheel.trial_timestamps[tridx][tpidx];
                if tpidx == 0:
                    tprev = stimOn_times[tridx] - wheel.stimOn_pre_duration
                cm_dur += (t - tprev)
                if abs(wheel.trial_velocity[tridx][tpidx]) > speed_threshold:
                    if cm_dur > duration_threshold:# and t > stimOn_times[tridx]:
                        wheel.movement_onset_times[tridx].append( t )
                    cm_dur = 0.0;
                tprev = t
            wheel.movement_onset_counts[tridx] = len(wheel.movement_onset_times[tridx])
            if len(wheel.movement_onset_times[tridx]) == 0: #trials with no explicit movement onset
                wheel.first_movement_onset_times[tridx] = np.NaN
                wheel.last_movement_onset_times[tridx] = np.NaN
            else:
                wheel.first_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][0]
                wheel.last_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][-1]

class SpikeData():
    def __init__(spikes, eid, probe_id):
        probe_name = 'probe0' + str(probe_id)
        session_path = one.path_from_eid(eid)
        probe_path = session_path.joinpath('alf', probe_name)
        
        try:
            spikes.cluster_metrics = alf.io.load_object(probe_path, object='clusters.metrics')['metrics']
        except FileNotFoundError:
            cluster_metrics_tmp = one.load(eid, dataset_types=['clusters.metrics'])
            try:
                spikes.cluster_metrics = alf.io.load_object(probe_path, object='clusters.metrics')['metrics']
            except FileNotFoundError: 
                spikes.cluster_metrics = np.zeros((1,10))
                print('cluster metric data for eid' + str(eid) + ', probe0' + str(probe_id) + ' is not there.' )
        spikes.cluster_metrics = np.array(spikes.cluster_metrics)

        num_clusters = len(spikes.cluster_metrics)
        spikes.cluster_firing_rates = np.zeros( (num_clusters) )
        spikes.cluster_presence_ratio = np.zeros( (num_clusters) )
        for cidx in range(num_clusters):
            spikes.cluster_firing_rates[cidx] = spikes.cluster_metrics[cidx][4]
            spikes.cluster_presence_ratio[cidx] = spikes.cluster_metrics[cidx][5]

        #cluster depth
        try:
            spikes.cluster_depths = alf.io.load_object(probe_path, object='clusters.depths')['depths']
        except FileNotFoundError:
            cluster_depths_tmp = one.load(eid, dataset_types=['clusters.depths'])
            try:
                spikes.cluster_depths = alf.io.load_object(probe_path, object='clusters.depths')['depths']
            except FileNotFoundError: 
                spikes.cluster_depths = [0.0]
                print('cluster depth data for eid' + str(eid) + ', probe0' + str(probe_id) + ' is not there.' )
            
    def load_spikes(spikes, eid, probe_id):
        probe_name = 'probe0' + str(probe_id)
        session_path = one.path_from_eid(eid)
        probe_path = session_path.joinpath('alf', probe_name)
        
        try:
            st = alf.io.load_object(probe_path, object='spikes.times')['times']
        except FileNotFoundError:
            sttmp = one.load(eid, dataset_types=['spikes.times'])
            st = alf.io.load_object(probe_path, object='spikes.times')['times']

        try:
            sc = alf.io.load_object(probe_path, object='spikes.clusters')['clusters']
        except FileNotFoundError:
            sctmp = one.load(eid, dataset_types=['spikes.clusters'])
            sc = alf.io.load_object(probe_path, object='spikes.clusters')['clusters']

        stmat = []
        for stidx in range(len(st)):
            cidx = int(sc[stidx])
            while len(stmat) <= cidx:
                stmat.append([])
            stmat[cidx].append(st[stidx])

        spikes.total_spikes = []
        spikes.times = [] #trial-based spike time
        for cidx in range( len(stmat) ):
            spikes.total_spikes.append( float(len(stmat[cidx])) )
            spikes.times.append([])
            if spikes.total_spikes[-1] > 0:
                for sttmp in stmat[cidx]:
                    spikes.times[-1].append( float(sttmp) )

        print( 'number of putative neurons: ' + str(len(spikes.times)) + ', total number of spikes ' + str(len(st)) )
        stmat.clear(); #st.clear() 

    def neuron_selection(spikes, min_presence_ratio, firing_rate_range, depth_range):
        min_firing_rate = firing_rate_range[0]; max_firing_rate = firing_rate_range[1];
        min_depth = depth_range[0]; max_depth = depth_range[1]; 
        num_clusters = len(spikes.cluster_metrics)
        spikes.selected = np.ones((num_clusters))

        for cidx in range( num_clusters ):
            if spikes.cluster_presence_ratio[cidx] < min_presence_ratio:
                spikes.selected[cidx] = 0.0
            if spikes.cluster_firing_rates[cidx] < min_firing_rate or max_firing_rate < spikes.cluster_firing_rates[cidx]:
                spikes.selected[cidx] = 0.0
            if spikes.cluster_depths[cidx] < min_depth or max_depth < spikes.cluster_depths[cidx]:
                spikes.selected[cidx] = 0.0
        print('number of selected neurons: ' + str( np.sum(spikes.selected) ))

    def calc_fr_vecs(spikes, fr_start_times, fr_end_times):
        #tbin: the bin size for firing rate calculation
        num_total_clusters = len(spikes.cluster_metrics)
        total_trial_count = len(fr_start_times);
        tbins = fr_end_times - fr_start_times
        tbin_total = np.sum(tbins)
        print('total trial count: ' + str(total_trial_count) + ', total duration: ' + str(tbin_total))
        
        spikes.fr_vecs_tmp = np.zeros((total_trial_count, num_total_clusters)) #raw firing rate vector
        trial_mean_firing_rates = np.zeros(num_total_clusters)
        spikes.num_selected_clusters = 0
        for cidx in range(num_total_clusters):
            if spikes.selected[cidx] > 0.5: #selection based on neuron_selection function
                tridx = 0; 
                for st in spikes.times[cidx]:
                    while st >= fr_end_times[tridx]:
                        if tridx < total_trial_count-1:
                            tridx += 1
                        else:
                            break;
                    if fr_start_times[tridx] <= st and st < fr_end_times[tridx]:
                        spikes.fr_vecs_tmp[tridx][cidx] += 1.0/tbins[tridx]
                        trial_mean_firing_rates[cidx] += 1.0/tbin_total
                        
                if trial_mean_firing_rates[cidx] > 0.0:
                    spikes.num_selected_clusters += 1
                else:
                    spikes.selected[cidx] = 0.0 #exclude the neuron from the list of selected neurons

        spikes.fr_vecs = np.zeros((total_trial_count, spikes.num_selected_clusters)) #firing rate vector of the selected neurons
        spikes.selected_cluster_firing_rates = np.zeros((spikes.num_selected_clusters))
        scidx = -1
        for cidx in range(num_total_clusters):
            if spikes.selected[cidx] > 0.5:
                scidx += 1;
                for tridx in range(total_trial_count):
                    spikes.fr_vecs[tridx][scidx] = spikes.fr_vecs_tmp[tridx][cidx]
                spikes.selected_cluster_firing_rates[scidx] = spikes.cluster_firing_rates[cidx]

    def calc_fr_cov(spikes,fr_norm_type, fr_norm_mtd):
        num_total_clusters = len(spikes.cluster_metrics)
        num_selected_clusters = spikes.num_selected_clusters
        total_trial_count = len(spikes.fr_vecs);

        if fr_norm_type == 'trialwise':
            spikes.trialwise_mean_fr = (1.0/total_trial_count)*np.dot( np.ones((total_trial_count)), spikes.fr_vecs )
            spikes.zm_fr_vecs = spikes.fr_vecs - np.outer( np.ones((total_trial_count)), spikes.trialwise_mean_fr )        
            spikes.fr_cov = (1.0/total_trial_count)*np.dot( np.transpose(spikes.zm_fr_vecs), spikes.zm_fr_vecs )

        elif fr_norm_type == 'overall':
            #calculate the last spike time
            last_spike_times = np.zeros( (num_total_clusters) )
            for cidx in range(num_total_clusters):
                if len(spikes.times[cidx]) > 0:
                    last_spike_times[cidx] = spikes.times[cidx][-1]
            print( 'last spike time: ' + str(np.max(last_spike_times)) )
            
            #calculate overall neuron-to-neuron correlation 
            tbin = 0.5;
            total_time = int(floor( np.max( last_spike_times )/tbin ) ) + 1
            spikes.tot_frs = np.zeros( (num_selected_clusters, total_time) )
            scidx = -1
            for cidx in range(num_total_clusters):
                if spikes.selected[cidx] > 0.5:
                    scidx += 1;
                    for st in spikes.times[cidx]:
                        spikes.tot_frs[scidx][ int(floor( st/tbin )) ] += 1.0/tbin
            
            #normalized zero mean firing rate
            spikes.tot_mean_frs = (1.0/total_time)*np.dot( spikes.tot_frs, np.ones((total_time)) ),
            zm_tot_frs = spikes.tot_frs - np.outer( spikes.tot_mean_frs, np.ones((total_time)) )

            spikes.zm_fr_vecs = spikes.fr_vecs - np.outer( np.ones((total_trial_count)), spikes.tot_mean_frs )        
            spikes.fr_cov = (1.0/total_time)*np.dot( zm_tot_frs, np.transpose(zm_tot_frs) )

        if fr_norm_mtd == 'z-score':
            spikes.fr_cov = np.diag( np.diag(spikes.fr_cov) )


def probe_location(eid, probe_id, depth_range): #return the position of the probe
    probe_name = 'probe0' + str(probe_id)
    probe_trajectory = one.load(eid, dataset_types=['probes.trajectory'])[0]
    mid_depth = 0.5*(depth_range[0] + depth_range[1])
    
    trajectory_tmp = []
    for pidx in range(len(probe_trajectory)):
        if probe_trajectory[pidx]['label'] == probe_name:
             trajectory_tmp = [probe_trajectory[pidx]['x'], probe_trajectory[pidx]['y'], probe_trajectory[pidx]['z'],\
                               probe_trajectory[pidx]['phi'], probe_trajectory[pidx]['theta'], probe_trajectory[pidx]['depth']]
    return trajectory_tmp

def calc_rsa(kernel_func, trial_types, selected_trials, zm_fr_vecs, fr_cov, shuffle_cnt):
    total_trial_count = len(trial_types)
    num_selected_clusters = len(zm_fr_vecs[0])
    ttlen = 20

    if kernel_func == 'gaussian': #Gaussian kernel
        fr_simil_tmp = np.dot( zm_fr_vecs, np.dot( np.linalg.inv(fr_cov), np.transpose(zm_fr_vecs) ) )
        fr_log_similarity = (1.0/num_selected_clusters)\
                            *(- np.outer( np.diag(fr_simil_tmp), np.ones((total_trial_count)) )\
                              - np.outer( np.ones((total_trial_count)), np.diag(fr_simil_tmp) )\
                              + 2.0*fr_simil_tmp)
        fr_similarity = np.exp(fr_log_similarity)

    elif kernel_func == 'linear': #linear kernel
        fr_similarity = (1.0/num_selected_clusters)*np.dot( zm_fr_vecs, np.dot( np.linalg.inv(fr_cov), np.transpose(zm_fr_vecs) ) )

    new_trial_types = []
    for trtidx in range( len(trial_types) ):
        new_trial_types.append( trial_types[trtidx] )

    fr_mean_similarity = np.zeros((shuffle_cnt, ttlen, ttlen))
    fr_mean_similarity_cnts = np.zeros((shuffle_cnt, ttlen, ttlen))
    for k in range(shuffle_cnt):
        for tr1idx in range( total_trial_count ):
            if selected_trials[tr1idx] > 0.5:
                tr_type1 = int(new_trial_types[tr1idx])
                for tr2idx in range( total_trial_count ):
                    if selected_trials[tr2idx] > 0.5:
                        tr_type2 = int(new_trial_types[tr2idx])
                        if (tr_type1 >= 0 and tr_type2 >= 0) and abs(tr1idx - tr2idx) > 10:
                            fr_mean_similarity[k][ tr_type1 ][ tr_type2 ] += fr_similarity[tr1idx][tr2idx]
                            fr_mean_similarity_cnts[k][ tr_type1 ][ tr_type2 ] += 1.0
                                  
        for tr1idx in range(ttlen):
            for tr2idx in range(ttlen):
                if fr_mean_similarity_cnts[k][tr1idx][tr2idx] > 0.0:
                    fr_mean_similarity[k][tr1idx][tr2idx] = fr_mean_similarity[k][tr1idx][tr2idx]/fr_mean_similarity_cnts[k][tr1idx][tr2idx]

        for i in range(10000): #randomization
            tr1idx = np.random.randint(0,total_trial_count)
            tr2idx = np.random.randint(0,total_trial_count)
            if selected_trials[tr1idx] > 0.5 and selected_trials[tr2idx] > 0.5:
                tttmp = new_trial_types[tr1idx]; new_trial_types[tr1idx] = new_trial_types[tr2idx]; new_trial_types[tr2idx] = tttmp
                                  
    #plt.pcolor(np.max(fr_mean_similarity[0])*np.ones((ttlen, ttlen)) - fr_mean_similarity[0])
    #plt.colorbar()
    #plt.show()
    return fr_mean_similarity

def main(min_depth, max_depth, start_cue, end_cue, kernel_func, fr_norm_type, fr_norm_mtd, tr_id):
    eids = one.search(date_range=[time_ranges[tr_id][0], time_ranges[tr_id][1]], dataset_types=['trials.probabilityLeft','trials.choice','trials.goCue_times','trials.stimOn_times','trials.feedback_times','clusters.depths', 'spikes.times'])
    print( 'the number of total sessions: ' + str(len(eids)) )
    min_behavior_criteria_count = 0; good_behavior_and_probe_count = 0

    #behavioral selection criteria
    min_trial_count = 300
    min_average_performance = 0.67
    prob_right_range = [[0.0, 1.0], [0.0, 1.0]]
    response_time_range = [0.0, 1.5] #[s] lower/upper threshold for the response time
    dt_preStim = 0.5 #duration of prestimulation period

    #neural selection criteria
    min_good_cluster = 50 #per region
    min_trials_per_type = 2 #minimum of number trials per given trial type
    min_presence_ratio = 0.75
    firing_rate_range = [0.25, 100.0]
    depth_range = [min_depth, max_depth]

    shuffle_cnt = 10 #100 #the number of shuffle for shuffled rsa

    similarity_measure_str = '_kernel_' + str(kernel_func) + '_fr-norm_' + str(fr_norm_type) + '_' + str(fr_norm_mtd)
    depth_str = '_depth_' + str(depth_range[0]) + '-' + str(depth_range[1])
    duration_str = '_dur_' + str(start_cue) + '-to-' + str(end_cue)
                                  
    fcstr = 'data/rsa4' + str(similarity_measure_str) + str(depth_str) + str(duration_str)\
            + '_trid' + str(tr_id) + '_sfc' + str(shuffle_cnt) + '.txt'
    fwc = open(fcstr, 'w')
    fwctmp = 'min_good_cluster: ' + str(min_good_cluster) + ', min_trials_per_type: ' + str(min_trials_per_type) \
             + ', min_presence_ratio: ' + str(min_presence_ratio) + ', firing_rate_range: ' + str(firing_rate_range[0]) + ' to ' + str(firing_rate_range[1])\
             + ', response_time_range: ' + str(response_time_range[0]) + ' to ' + str(response_time_range[1]) + '\n'
    fwc.write(fwctmp)
    
    for eid in eids:
        trials = TrialData(eid)
        if trials.good_behaviour(eid, min_trial_count, min_average_performance, prob_right_range):
            min_behavior_criteria_count += 1; print(eid)

            trials.calc_trial_types()
            trials.trial_selection(response_time_range)

            probe_trajectory = one.load(eid, dataset_types=['probes.trajectory'])
            probe_trajectory = probe_trajectory[0]
            for probe_id in range( len(probe_trajectory) ):
                spikes = SpikeData(eid, probe_id)
                spikes.neuron_selection(min_presence_ratio, firing_rate_range, depth_range)
                
                num_good_clusters = int(np.sum(spikes.selected))
                print('num_good_clusters: ' + str(num_good_clusters))
                
                if num_good_clusters >= min_good_cluster and min(trials.trial_count) >= min_trials_per_type:
                    good_behavior_and_probe_count += 1; print( str(eid) + ', probe0' + str(probe_id) )
                    spikes.load_spikes(eid, probe_id)

                    wheel = WheelData(eid)
                    wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
                    wheel.calc_movement_onset_times(trials.stimOn_times)
                    
                    #start_time
                    if start_cue == 'stimOn':
                        fr_start_times = trials.stimOn_times#
                    elif start_cue == 'preStim':
                        fr_start_times = trials.stimOn_times - dt_preStim**np.ones(( len(trials.stimOn_times) ))
                    elif start_cue == 'FmoveOn':
                        fr_start_times = wheel.first_movement_onset_times
                    elif start_cue == 'LmoveOn':
                        fr_start_times = wheel.last_movement_onset_times
                    
                    for tridx in range( len(fr_start_times) ):
                        if isnan(fr_start_times[tridx]):
                            fr_start_times[tridx] = trials.stimOn_times[tridx] + 60.0
                
                    #end time
                    if end_cue == 'feedback':
                        fr_end_times = trials.feedback_times#
                    elif end_cue == 'stimOn':
                        fr_end_times = trials.stimOn_times
                    elif end_cue == 'FmoveOn':
                        fr_end_times = wheel.first_movement_onset_times
                    elif end_cue == 'LmoveOn':
                        fr_end_times = wheel.last_movement_onset_times

                    for tridx in range( len(fr_end_times) ):
                        if isnan(fr_end_times[tridx]):
                            fr_end_times[tridx] = trials.stimOn_times[tridx] + 60.0
                    
                    spikes.calc_fr_vecs(fr_start_times, fr_end_times)
                    trials.trial_reselection(spikes.fr_vecs)
                    if min(trials.new_trial_count) >= min_trials_per_type:
                        spikes.calc_fr_cov(fr_norm_type, fr_norm_mtd)
                        similarity_matrix_tmp = calc_rsa(kernel_func, trials.trial_types, trials.reselected_trials, spikes.zm_fr_vecs, spikes.fr_cov, shuffle_cnt)
                    
                        probe_xyz = probe_location(eid, probe_id, depth_range)
                        fwctmp = eid + ' probe0' + str(probe_id) + ' ' + str(probe_xyz[0]) + ' ' + str(probe_xyz[1]) + ' ' + str(probe_xyz[2]) + ' ' + str(probe_xyz[3]) + ' ' + str(probe_xyz[4]) + ' ' + str(probe_xyz[5])
                        for trtidx in range( len(trials.trial_count) ):
                            fwctmp +=  " " + str(trials.trial_count[trtidx])
                        fwc.write(fwctmp + '\n');
                     
                        for k in range(shuffle_cnt):
                            fwctmp = ""
                            for trtidx1 in range( len(trials.trial_count) ):
                                for trtidx2 in range( len(trials.trial_count) ):
                                    fwctmp += str(similarity_matrix_tmp[k][trtidx1][trtidx2]) + " "
                            fwc.write(fwctmp + '\n');
                        fwc.flush()
                del spikes
                
    print('min_behavior_criteria_count: ' + str(min_behavior_criteria_count))
    print('good_behavior_and_probe_count: ' + str(good_behavior_and_probe_count))
    
if __name__ == "__main__":
    param = sys.argv
    min_depth = int(param[1])
    max_depth = int(param[2])
    start_cue = str(param[3]) # 'preStim','stimOn', 'FmoveOn', or 'LmoveOn'
    end_cue = str(param[4]) #'stimOn', 'FmoveOn', 'LmoveOn', or 'feedback'
    kernel_func = str(param[5]) #'gaussian' or 'linear'
    fr_norm_type = str(param[6]) #'trialwise' or 'overall'
    fr_norm_mtd = str(param[7]) #'whiten' or 'z-score'
    time_range_id = int(param[8])

    main(min_depth, max_depth, start_cue, end_cue, kernel_func, fr_norm_type, fr_norm_mtd, time_range_id)
    
