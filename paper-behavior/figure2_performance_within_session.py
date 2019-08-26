#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:47:20 2019

Quantify behavioral performance within a session

@author: guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from figure_style import seaborn_style
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavioral_analyses

# Settings
fig_path = '/home/guido/Figures/Behavior/'
csv_path = '/home/guido/Data/Behavior/'
window_size = 51  # must be an uneven number
step_size = 5

# Query list of subjects
use_subjects = (subject.Subject
                * subject.SubjectLab
                * subject.SubjectProject
                & 'subject_project = "ibl_neuropixel_brainwide_01"')
subjects = use_subjects.fetch('subject_nickname')

# Create dataframe with behavioral metrics of all mice
perf_df = pd.DataFrame(columns=['mouse', 'lab', 'perf', 'centers',
                                'num_trials', 'window_size', 'step_size'])

# Get all sessions
sess = (acquisition.Session
        * subject.Subject
        * subject.SubjectLab
        * (behavioral_analyses.SessionTrainingStatus()
            & 'training_status="trained"')
        & 'task_protocol LIKE "%training%"')
session_start_time = sess.fetch('session_start_time')

for i, ses_start in enumerate(session_start_time):
    if np.mod(i+1, 10) == 0:
        print('Loading data of session %d of %d' % (i+1, len(sess)))

    # Load in session data
    trials = behavior.TrialSet.Trial & 'session_start_time="%s"' % ses_start
    correct, contr_l, contr_r = trials.fetch('trial_feedback_type',
                                             'trial_stim_contrast_left',
                                             'trial_stim_contrast_right')

    # Calculate percentage correct for high contrast trials with sliding window
    high_contrast = np.abs(contr_l - contr_r) >= 0.5
    centers = np.arange(np.median(np.arange(window_size)),
                        np.size(correct) - np.median(np.arange(window_size)),
                        step_size,
                        dtype=int)
    perf = np.zeros(np.size(centers))
    num_trials = np.zeros(np.size(centers))
    for j, c in enumerate(centers):
        window = correct[c-int((window_size-1)/2):c+int((window_size-1)/2)]
        window = window[high_contrast[c-int((window_size-1)/2):
                        c+int((window_size-1)/2)]]
        perf[j] = np.sum(window == 1) / np.size(window)
        num_trials[j] = np.size(window)

    # Add to dataframe
    perf_df.loc[i, 'mouse'] = (sess
                               & 'session_start_time="%s"'
                               % ses_start).fetch('subject_nickname')[0]
    perf_df.loc[i, 'lab'] = (sess
                             & 'session_start_time="%s"'
                             % ses_start).fetch('lab_name')[0]
    perf_df.loc[i, 'perf'] = [perf]
    perf_df.loc[i, 'centers'] = [centers]
    perf_df.loc[i, 'num_trials'] = [num_trials]
    perf_df.loc[i, 'window_size'] = window_size
    perf_df.loc[i, 'step_size'] = step_size











