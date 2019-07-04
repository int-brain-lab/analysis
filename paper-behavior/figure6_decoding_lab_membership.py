#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Try to predict in which lab an animal was trained based on its behavior

@author: guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
from figure_style import seaborn_style
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Settings
path = '/home/guido/Figures/Behavior/'
iterations = 2000     # how often to decode
num_splits = 3        # n in n-fold cross validation
decoding_metrics = ['perf_easy','n_trials','threshold','bias','reaction_time','training_time']
decoding_metrics_control = ['perf_easy','n_trials','threshold','bias','reaction_time','training_time','time_zone']

# Decoding function with n-fold cross validation
def decoding(resp, labels, clf, num_splits):
    kf = KFold(n_splits=num_splits, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    return f1

# Query list of subjects
use_subjects = subject.Subject * subject.SubjectLab & 'subject_birth_date > "2018-09-01"' \
                        & 'subject_line IS NULL OR subject_line="C57BL/6J"' \
                        & 'subject_source IS NULL OR subject_source="Jax"'
subjects = use_subjects.fetch('subject_nickname')

# Create dataframe with behavioral metrics of all mice        
learning = pd.DataFrame(columns=['mouse','lab','time_zone','learned','date_learned','training_time','perf_easy','n_trials','threshold','bias','reaction_time','lapse_low','lapse_high'])
for i, nickname in enumerate(subjects):
    if np.mod(i+1,10) == 0: 
        print('Loading data of subject %d of %d'%(i+1,len(subjects)))
    
    # Gather behavioral data for subject
    subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%nickname
    behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname).proj('session_date', 'performance_easy').fetch(as_dict=True, order_by='session_date'))
    rt = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.ReactionTimeByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname)).proj('session_date', 'median_reaction_time').fetch(as_dict=True, order_by='session_date'))
    psych = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.PsychResults * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname)).proj('session_date', 'n_trials_stim','threshold','bias','lapse_low','lapse_high').fetch(as_dict=True, order_by='session_date'))
    
    # Find first session in which mouse is trained
    first_trained_session = subj.aggr(behavior_analysis.SessionTrainingStatus &	'training_status="trained"', first_trained='min(session_start_time)')
    untrainable_session = subj.aggr(behavior_analysis.SessionTrainingStatus & 'training_status="untrainable"', first_trained='min(session_start_time)')
    if len(first_trained_session) == 0 & len(untrainable_session) == 0:
        learning.loc[i,'learned'] = 'in training'
        learning.loc[i,'training_time'] = len(behav)
    elif len(first_trained_session) == 0 & len(untrainable_session) == 1:
        learning.loc[i,'learned'] = 'untrainable'
        learning.loc[i,'training_time'] = len(behav)
    else:
        first_trained_session_datetime = first_trained_session.fetch1('first_trained')    
        first_trained_session_date = first_trained_session_datetime.date()
        learning.loc[i,'learned'] = 'trained'
        learning.loc[i,'date_learned'] = first_trained_session_date
        learning.loc[i,'training_time'] = sum(behav.session_date < first_trained_session_date)
        learning.loc[i,'perf_easy'] = float(behav.performance_easy[behav.session_date == first_trained_session_date])*100
        psych['n_trials'] = n_trials = [sum(s) for s in psych.n_trials_stim]
        learning.loc[i,'n_trials'] = float(psych.n_trials[psych.session_date == first_trained_session_date])
        learning.loc[i,'threshold'] = float(psych.threshold[psych.session_date == first_trained_session_date])
        learning.loc[i,'bias'] = float(psych.bias[psych.session_date == first_trained_session_date])
        learning.loc[i,'lapse_low'] = float(psych.lapse_low[psych.session_date == first_trained_session_date])
        learning.loc[i,'lapse_high'] = float(psych.lapse_high[psych.session_date == first_trained_session_date])
        if sum(rt.session_date == first_trained_session_date) == 0:
            learning.loc[i,'reaction_time'] = float(rt.median_reaction_time[np.argmin(np.array(abs(rt.session_date - first_trained_session_date)))])*1000
        else:
            learning.loc[i,'reaction_time'] = float(rt.median_reaction_time[rt.session_date == first_trained_session_date])*1000
        
    # Add mouse and lab info to dataframe
    learning.loc[i,'mouse'] = nickname
    lab_name = subj.fetch1('lab_name')
    learning.loc[i,'lab'] = lab_name
    lab_time = reference.Lab * reference.LabLocation & 'lab_name="%s"'%lab_name
    time_zone = lab_time.fetch('time_zone')[0]
    if time_zone == ('Europe/Lisbon' or 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7
    learning.loc[i,'time_zone'] = time_zone_number
    
# Select mice that learned
learned = learning[learning['learned'] == 'trained'] 

# Merge some labs
pd.options.mode.chained_assignment = None  # deactivate warning
learned.loc[learned['lab'] == 'zadorlab','lab'] = 'churchlandlab'
learned.loc[learned['lab'] == 'mrsicflogellab','lab'] = 'cortexlab'

# Add (n = x) to lab names
for i in learned.index.values:
    learned.loc[i,'lab_n'] = learned.loc[i,'lab'] + ' (n=' + str(sum(learned['lab'] == learned.loc[i,'lab'])) + ')'

# Initialize decoders
print('\nDecoding of lab membership..')
decod = learned
clf_rf = RandomForestClassifier(n_estimators=100)

# Perform decoding of lab membership
decoding_result = pd.DataFrame(columns=['original','original_shuffled','control','control_shuffled'])
decoding_set = decod[decoding_metrics].values
control_set = decod[decoding_metrics_control].values
for i in range(iterations):
    if np.mod(i+1,100) == 0:
        print('Iteration %d of %d'%(i+1,iterations))        
    # Original dataset
    decoding_result.loc[i,'original'] = decoding(decoding_set, list(decod['lab']), clf_rf, num_splits)
    decoding_result.loc[i,'original_shuffled'] = decoding(decoding_set, list(decod['lab'].sample(frac=1)), clf_rf, num_splits)
    # Positive control dataset
    decoding_result.loc[i,'control'] = decoding(control_set, list(decod['lab']), clf_rf, num_splits)
    decoding_result.loc[i,'control_shuffled'] = decoding(control_set, list(decod['lab'].sample(frac=1)), clf_rf, num_splits)
   

# Calculate if decoder performs above chance (positive values in perc indicate above chance-level performance)
sig = np.percentile(decoding_result['original']-np.mean(decoding_result['original_shuffled']),5)
sig_control = np.percentile(decoding_result['control']-np.mean(decoding_result['control_shuffled']),5)

# Plot decoding results
seaborn_style()
plt.figure(figsize=(5,5))
fig = plt.gcf()
ax1 = plt.gca()
sns.violinplot(data=pd.concat([decoding_result['original']-decoding_result['original_shuffled'], 
                               decoding_result['control']-decoding_result['control_shuffled']], axis=1), color=[0.6,0.6,0.6], ax=ax1)
ax1.plot([-1,5],[0,0],'r--')
ax1.set(ylabel='Decoding performance over chance level\n(F1 score)', title='Random forest classifier', 
        ylim=[-0.2,0.5], xticklabels=['Decoding of\nlab membership', 'Including\ntime zone'])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)
plt.tight_layout(pad = 2)
fig.set_size_inches((5, 5), forward=False) 
plt.savefig(join(path,'figure6_panel_decoding.pdf'), dpi=300)
plt.savefig(join(path,'figure6_panel_decoding.png'), dpi=300)


