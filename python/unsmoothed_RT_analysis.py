# -*- coding: utf-8 -*-
#
# Reaction time analysis from wheel data without wheel velocity smoothing
#
# Return all significant wheel velocity changes in each trial
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

class TrialData():
    def __init__(trials, eid):
        trials.goCue_times, trials.stimOn_times, trials.feedback_times = one.load(eid, dataset_types=['trials.goCue_times', 'trials.stimOn_times', 'trials.feedback_times'])

        trials.total_trial_count = len(trials.goCue_times)
        #supplementing the effective feedback time
        for tridx in range( len(trials.feedback_times) ):
            if isnan(trials.feedback_times[tridx]):
                trials.feedback_times[tridx] = trials.stimOn_times[tridx] + 60.0
        
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

def main():
    subject_name = 'CSHL_008'
    data_ranges = ['2019-05-01', '2019-06-03']
    eids = one.search(subject=subject_name,dataset_types=['trials.goCue_times','trials.stimOn_times','trials.feedback_times','wheel.position', 'wheel.timestamps'], date_range=[data_ranges[0], data_ranges[1]])
    goCueRTs = []; stimOnRTs = []
    eid_count = 0
    for eidx in range(len(eids)):
        eid = eids[eidx]
        trials = TrialData(eid)
        wheel = WheelData(eid)
        if wheel.data_error == False:
            wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
            wheel.calc_movement_onset_times(trials.stimOn_times)
            for rtidx in range( len(wheel.first_movement_onset_times) ):
                goCueRTs.append(wheel.first_movement_onset_times[rtidx] - trials.goCue_times[rtidx])
                stimOnRTs.append(wheel.first_movement_onset_times[rtidx] - trials.stimOn_times[rtidx])
            eid_count += 1

    print('number of sessions: ' + str(eid_count))
    svfg = plt.figure()
    plt.subplot(1,2,1)
    histtmps = plt.hist(stimOnRTs, range=(-0.3,0.5), bins=100)
    median_rt = np.ma.median( np.ma.masked_invalid(np.array(stimOnRTs)) )
    plt.axvline(median_rt, ls='--', lw=1.5, color='k')
    plt.ylabel('Histgram')
    plt.xlabel('Time [s]')
    plt.title('[Movement onset] - [stimOn]')

    plt.subplot(1,2,2)
    histtmps = plt.hist(goCueRTs, range=(-0.3,0.5), bins=100)
    median_rt = np.ma.median( np.ma.masked_invalid(np.array(goCueRTs)) )
    plt.axvline(median_rt, ls='--', lw=1.5, color='k')
    plt.xlabel('Time [s]')
    plt.title('[Movement onset] - [goCue]')
    
    plt.show()
    svfg.savefig('fig_unsmoothed_RT_subject_' + subject_name + '_Nsessions' + str(eid_count) + '.pdf')

if __name__ == "__main__":
    param = sys.argv
    main()
    
