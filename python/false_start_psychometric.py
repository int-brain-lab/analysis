# -*- coding: utf-8 -*-
#
# Reaction time analysis
#
# Psychometric curve estimated from wheel movement direction for early/late reaction trials
#
# Author: Naoki Hiratani (N.Hiratani@gmail.com)
#
from math import *

import sys
import alf.io
from oneibl.one import ONE
from ibllib.misc import pprint
import numpy as np
import scipy.stats as scist
from os import path
import matplotlib.pyplot as plt

one = ONE()

contrastTypes = [0.0, 0.0625, 0.125, 0.25, 1.0]
ctlen = len(contrastTypes)

class TrialData():
    def __init__(trials, eid):
        trials.goCue_times, trials.stimOn_times, trials.feedback_times, trials.contrastLeft, trials.contrastRight, trials.probLeft, trials.choice, trials.feedbackType = one.load(eid, dataset_types=['trials.goCue_times', 'trials.stimOn_times', 'trials.feedback_times', 'trials.contrastLeft', 'trials.contrastRight', 'trials.probabilityLeft', 'trials.choice', 'trials.feedbackType'])

        trials.total_trial_count = len(trials.goCue_times)
        #supplementing the effective feedback time
        for tridx in range( len(trials.feedback_times) ):
            if isnan(trials.feedback_times[tridx]):
                trials.feedback_times[tridx] = trials.stimOn_times[tridx] + 60.0

    def psychometric_curve(trials, wheel_directions, reaction_times, false_start_threshold):
        total_trial_count = trials.total_trial_count
        trials.performance = np.zeros((2,2,2*ctlen)) #[early/late response, left/right block, contrast]
        trials.fraction_choice_right = np.zeros((2,2,2*ctlen))
        trials.performance_cnts = np.zeros((2,2,2*ctlen))
        
        for tridx in range(total_trial_count):
            if trials.probLeft[tridx] != 0.5:
                FStrial = 1 if reaction_times[tridx] < false_start_threshold else 0
                Rblock = 1 if trials.probLeft[tridx] < 0.5 else 0
                Rtrial = 1 if isnan(trials.contrastLeft[tridx]) else 0
                contrast_idx = 0
                for cidx in range(ctlen):
                    if Rtrial == 1:
                        if abs(trials.contrastRight[tridx] - contrastTypes[cidx]) < 0.001:
                            contrast_idx = ctlen + cidx
                    else:
                        if abs(trials.contrastLeft[tridx] - contrastTypes[cidx]) < 0.001:
                            contrast_idx = ctlen-1 - cidx
                #Behavioural psychometric
                #trials.fraction_choice_right[FStrial][Rblock][contrast_idx] += 0.5 - trials.choice[tridx]/2.0
                #trials.performance[FStrial][Rblock][contrast_idx] += 0.5 + trials.feedbackType[tridx]/2.0
                
                #Wheel psychometric
                trials.fraction_choice_right[FStrial][Rblock][contrast_idx] += 0.5 + wheel_directions[tridx]/2.0
                trials.performance[FStrial][Rblock][contrast_idx] += 0.5 + 0.5*np.sign( (Rtrial - 0.5)*wheel_directions[tridx] )
                trials.performance_cnts[FStrial][Rblock][contrast_idx] += 1.0

class WheelData():
    def __init__(wheel, eid):
        wheel.position, wheel.timestamps = one.load(eid, dataset_types=['wheel.position', 'wheel.timestamps'])
        wheel.data_error = False
        if str(type(wheel.position)) == "<class 'pathlib.PosixPath'>" or \
            str(type(wheel.timestamps)) == "<class 'pathlib.PosixPath'>":
            wheel.data_error = True
        else:
            wheel.velocity = wheel.calc_wheel_velocity()
    
    def calc_wheel_velocity(wheel):
        wheel_velocity = []; wheel_velocity.append(0.0);
        if min(np.diff(wheel.timestamps)) <= 0.0:
            wheel.data_error = True
        else:
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
        
        wheel.movement_directions = []
        wheel.first_movement_directions = np.zeros( (wheel.total_trial_count) )
        wheel.last_movement_directions = np.zeros( (wheel.total_trial_count) )
        
        for tridx in range(len(wheel.trial_timestamps)):
            wheel.movement_onset_times.append([])
            wheel.movement_directions.append([])
            cm_dur = 0.0; #continous stationary duration
            for tpidx in range( len(wheel.trial_timestamps[tridx]) ):
                t = wheel.trial_timestamps[tridx][tpidx];
                if tpidx == 0:
                    tprev = stimOn_times[tridx] - wheel.stimOn_pre_duration
                cm_dur += (t - tprev)
                if abs(wheel.trial_velocity[tridx][tpidx]) > speed_threshold:
                    if cm_dur > duration_threshold:# and t > stimOn_times[tridx]:
                        wheel.movement_onset_times[tridx].append( t )
                        wheel.movement_directions[tridx].append( np.sign(wheel.trial_velocity[tridx][tpidx]) )
                    cm_dur = 0.0;
                tprev = t
            wheel.movement_onset_counts[tridx] = len(wheel.movement_onset_times[tridx])
            if len(wheel.movement_onset_times[tridx]) == 0: #trials with no explicit movement onset
                wheel.first_movement_onset_times[tridx] = np.NaN
                wheel.last_movement_onset_times[tridx] = np.NaN
                wheel.first_movement_directions[tridx] = 0
                wheel.last_movement_directions[tridx] = 0
            else:
                wheel.first_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][0]
                wheel.last_movement_onset_times[tridx] = wheel.movement_onset_times[tridx][-1]
                wheel.first_movement_directions[tridx] = wheel.movement_directions[tridx][0]
                wheel.last_movement_directions[tridx] = wheel.movement_directions[tridx][-1]

def main():
    false_start_threshold = 0.08 #[second], reaction time threshold for false start detection
    time_ranges = ['2020-01-01', '2020-03-31']
    eids = one.search(date_range=[time_ranges[0], time_ranges[1]], dataset_types=['trials.probabilityLeft','trials.choice','trials.goCue_times','trials.stimOn_times','trials.feedback_times','clusters.depths','spikes.times'])

    eid_count = 0; cml_RTs = []
    
    #psychometric curves for early/late response and left/right block trials, as a function of the stimulus contrast
    sum_performance = np.zeros((2,2,2*ctlen))
    sum_fraction_choice_right = np.zeros((2,2,2*ctlen))
    performance_cnts = np.zeros((2,2,2*ctlen))
    for eidx in range(len(eids)):
        eid = eids[eidx]
        trials = TrialData(eid)
        wheel = WheelData(eid)
        if wheel.data_error == False:
            wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
            wheel.calc_movement_onset_times(trials.stimOn_times)
            RTs = wheel.first_movement_onset_times - trials.stimOn_times
            for RTtmp in RTs:
                cml_RTs.append(RTtmp)
            trials.psychometric_curve(wheel.first_movement_directions, RTs, false_start_threshold)
            sum_performance += trials.performance
            sum_fraction_choice_right += trials.fraction_choice_right
            performance_cnts += trials.performance_cnts
            eid_count += 1

    performance = np.zeros((2,2,2*ctlen))
    fraction_choice_right = np.zeros((2,2,2*ctlen))
    for fsidx in range(2):
        for bidx in range(2):
            for ctidx in range(2*ctlen):
                if performance_cnts[fsidx][bidx][ctidx] > 0:
                    fraction_choice_right[fsidx][bidx][ctidx] = sum_fraction_choice_right[fsidx][bidx][ctidx]/performance_cnts[fsidx][bidx][ctidx]
                    performance[fsidx][bidx][ctidx] = sum_performance[fsidx][bidx][ctidx]/performance_cnts[fsidx][bidx][ctidx]

    print('number of sessions: ' + str(eid_count))

    svfg1 = plt.figure()
    plt.hist(cml_RTs, range=(-0.3,1.2), bins=150)
    plt.axvline(false_start_threshold, c='k', ls='--', lw=1.5)
    plt.xlim(-0.3,1.2)
    plt.show()
    svfg1.savefig('fig_false_start_psychometric_RT' + '_ephys_' + time_ranges[0] + '_to_' + time_ranges[1] + '_fsth' + str(false_start_threshold) + '.pdf')

    svfg2 = plt.figure()
    for fsidx in range(2):
        plt.subplot(1,2,fsidx+1)
        for bidx in range(2):
            plt.plot(fraction_choice_right[fsidx][bidx])
        plt.ylim(0.0,1.0)
        plt.xticks([0,2,4,5,7,9], [-1.0, -0.125, 0.0, 0.0, 0.125, 1.0])
        plt.xlabel('Contrast')
        if fsidx == 0:
            plt.ylabel('Fraction of right-ward first wheel movement')
    print(performance)
    print(fraction_choice_right)
    plt.show()
    svfg2.savefig('fig_false_start_psychometric_fraction_right' + '_ephys_' + time_ranges[0] + '_to_' + time_ranges[1] + '_fsth' + str(false_start_threshold) + '.pdf')

if __name__ == "__main__":
    param = sys.argv
    
    main()
    
