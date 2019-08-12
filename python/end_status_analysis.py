'''
Plot various criteria for each mouse session
'''

import pandas as pd
import random
import matplotlib.pyplot as plt
import sys, os
import end_session_criteria
from collections import Counter

from ibl_pipeline import subject, behavior, acquisition
from ibl_pipeline.analyses import behavior as behavior_analysis
from training_criteria_schemas import SessionTrainingStatus

sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Python Scripts'))
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Github/analysis/python'))
import dat

def fetch_trials(subj, *args):
    if not args:
        subj, date, seq = dat.parse_ref(subj)

    #  print(subject, date, seq)
    sessions_with_date = acquisition.Session.proj('session_number', session_date='date(session_start_time)')
    trial_set = behavior.TrialSet.Trial
    trial_set = trial_set.proj(
        'trial_response_choice',
        'trial_response_choice',
        'trial_response_time',
        'trial_stim_on_time',
        'trial_start_time',
        signed_contrast='trial_stim_contrast_right \
            - trial_stim_contrast_left',
        rt='trial_response_time - trial_stim_on_time',
        correct='trial_feedback_type = 1')
    query = subject.Subject * sessions_with_date * behavior.CompleteTrialSession * trial_set \
        & 'session_date = "{}"'.format(date.strftime('%Y-%m-%d')) \
        & 'subject_nickname = "{}"'.format(subj) \
        & 'session_number = "{}"'.format(seq) \

    trials = query.fetch(
        'trial_response_choice',
        'trial_response_choice',
        'trial_response_time',
        'trial_stim_on_time',
        'trial_start_time',
        'signed_contrast',
        'rt',
        'correct', as_dict=True)
    return pd.DataFrame(trials)

trial_path = os.path.join(os.path.expanduser('~'), 'Documents/Github/analysis/python', 'trial_data')

if os.path.isfile(trial_path):
    trials = pd.read_pickle(trial_path)
else:
    query = subject.Subject * acquisition.Session * behavior.TrialSet * behavior.CompleteTrialSession * SessionTrainingStatus * \
            subject.SubjectProject * end_session_criteria.SessionEndCriteria & 'subject_project = "ibl_neuropixel_brainwide_01"'
    trials = pd.DataFrame(query.fetch('subject_nickname', 'session_start_time', 'session_number',
                                              'n_trials', 'end_status', 'end_status_index', 'training_status', as_dict=True))
    trials.to_pickle(trial_path)

trials['end_status'].value_counts()
trials['trial_end_diff'] = trials['n_trials'] - trials['end_status_index']
trials['trial_end_diff'].hist(bins=30)
print(trials['end_status'].value_counts())

#  Find refs of perf decline sessions
refs = []
proc = []
for index, row in trials.iterrows():
   if row['end_status'] == 'perf_ez<40':
       r = dat.construct_ref(row['subject_nickname'], row['session_start_time'], row['session_number'])
       refs.append(r)
       proc.append(row['training_status'])
print((Counter(proc).keys(), Counter(proc).values()))
#  (dict_keys(['intraining', 'trained_1b', 'untrainable', 'ready4ephysrig']), dict_values([94, 2, 5, 3]))

for ref in random.sample(refs,5):
    print(ref)
    trials = fetch_trials(ref)
    fig, axes = plt.subplots(1, 1)
    end_session_criteria.session_end_indices(trials, make_plot=True, ax=axes)
    fig.suptitle(ref)
plt.show()
