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
data_path = '/home/guido/Data/Behavior/'
window_size = 51  # must be an uneven number
step_size = 5

# Query list of subjects
use_subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject
                & 'subject_project = "ibl_neuropixel_brainwide_01"')
subjects = use_subjects.fetch('subject_nickname')

# Create dataframe with behavioral metrics of all mice
perf_df = pd.DataFrame(columns=['mouse', 'lab', 'perf', 'centers', 'num_trials', 'window_size',
                                'step_size'])

# Get all sessions
sess = (acquisition.Session * subject.Subject * subject.SubjectLab
        * (behavioral_analyses.SessionTrainingStatus() & 'training_status="trained"')
        & 'task_protocol LIKE "%training%"')
session_start_time = sess.fetch('session_start_time')

for i, ses_start in enumerate(session_start_time):
    if np.mod(i+1, 10) == 0:
        print('Loading data of session %d of %d' % (i+1, len(sess)))

    # Load in session data
    trials = behavior.TrialSet.Trial & 'session_start_time="%s"' % ses_start
    correct, contr_l, contr_r = trials.fetch('trial_feedback_type', 'trial_stim_contrast_left',
                                             'trial_stim_contrast_right')

    # Calculate percentage correct for high contrast trials with sliding window
    high_contrast = np.abs(contr_l - contr_r) >= 0.5
    centers = np.arange(np.median(np.arange(window_size)),
                        np.size(correct) - np.median(np.arange(window_size)), step_size, dtype=int)
    perf = np.zeros(np.size(centers))
    num_trials = np.zeros(np.size(centers))
    for j, c in enumerate(centers):
        window = correct[c-int((window_size-1)/2):c+int((window_size-1)/2)]
        window = window[high_contrast[c-int((window_size-1)/2):c+int((window_size-1)/2)]]
        perf[j] = np.sum(window == 1) / np.size(window)
        num_trials[j] = np.size(window)

    # Add to dataframe
    perf_df.loc[i, 'mouse'] = (sess & 'session_start_time="%s"' % ses_start).fetch(
                                                                        'subject_nickname')[0]
    perf_df.loc[i, 'lab'] = (sess & 'session_start_time="%s"' % ses_start).fetch('lab_name')[0]
    perf_df.loc[i, 'perf'] = [perf]
    perf_df.loc[i, 'centers'] = [centers]
    perf_df.loc[i, 'num_trials'] = [num_trials]
    perf_df.loc[i, 'window_size'] = window_size
    perf_df.loc[i, 'step_size'] = step_size
    perf_df.to_pickle(join(data_path, 'within_session_perf'))

perf_df = pd.read_pickle(join(data_path, 'within_session_perf'))


# Function to get average of list of lists with different lengths
def arrays_mean(list_of_arrays):
    arr_lengths = np.array([len(x) for x in list_of_arrays])
    means = np.zeros(max(arr_lengths))
    stderrs = np.zeros(max(arr_lengths))
    for j in range(max(arr_lengths)):
        means[j] = np.mean([el[j] for el in list_of_arrays[arr_lengths > j]])
        stderrs[j] = (np.std([el[j] for el in list_of_arrays[arr_lengths > j]])
                      / np.sqrt(np.sum(arr_lengths > j)))
    return means, stderrs


# Calculate per mouse the average performance of sliding window over sessions
all_perf = []
for i, mouse_id in enumerate(np.unique(perf_df['mouse'])):
    arrays = np.array(
            [x for sublist in perf_df.loc[perf_df['mouse'] == mouse_id, 'perf'] for x in sublist])
    all_perf.append(arrays_mean(arrays)[0])


# Get mean over mice
mean_perf, stderr_perf = arrays_mean(np.array(all_perf))

# Plot results
plt.figure(figsize=(8, 5))
seaborn_style()
fig = plt.gcf()
ax1 = plt.gca()
all_centers = np.array([x for sl in perf_df['centers'] for x in sl])
max_centers = all_centers[np.argmax([len(x) for x in all_centers])]
sns.lineplot(x=max_centers, y=mean_perf, color='k')
plt.fill_between(max_centers, mean_perf-(stderr_perf/2), mean_perf+(stderr_perf/2),
                 alpha=0.5, facecolor=[0.6, 0.6, 0.6])
ax1.set(xlim=[0, 1400], ylim=[0.9, 1], ylabel='Performance on easy contrasts (%)',
        yticklabels=['90', '92', '94', '96', '98', '100'],
        xlabel='Center of sliding window (trials)',
        title='n = %d mice, window size = %d trials, step size = %d trials' % (
                len(all_perf), perf_df.loc[0, 'window_size'], perf_df.loc[0, 'step_size']))

plt.tight_layout(pad=2)
plt.savefig(join(fig_path, 'figure2_panel_perf_within_session.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2_panel_perf_within_session.png'), dpi=300)
