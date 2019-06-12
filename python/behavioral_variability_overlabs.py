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
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analyses
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Settings
path = '/home/guido/Figures/Behavior/'
decod_it = 2000
shuffle_it = 2000
num_splits = 3

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
#all_sub = subject.Subject * subject.SubjectLab & 'subject_birth_date between "2018-09-01" and "2019-02-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'
all_sub = subject.Subject * subject.SubjectLab & 'subject_birth_date < "2018-09-01"'
subjects = pd.DataFrame(all_sub)
        
learning = pd.DataFrame(columns=['mouse','lab','learned','date_learned','training_time','perf_easy','n_trials','threshold','bias','lapse_low','lapse_high'])
for i, nickname in enumerate(subjects['subject_nickname']):
    print('Processing subject %s'%nickname)
    subj = subject.Subject & 'subject_nickname="%s"'%nickname
    session_start, training_status = (behavior_analyses.SessionTrainingStatus & subj).fetch('session_start_time', 'training_status')
    
    # Get behavior metrics
    performance_easy, n_trials_stim, threshold, bias, lapse_low, lapse_high = (behavior_analyses.PsychResults & subj).fetch('performance_easy', 'n_trials_stim', 'threshold', 'bias', 'lapse_low', 'lapse_high')    
        
    # Find first session in which mouse is trained
    res = next((i for i, j in enumerate(training_status == 'trained') if j), None)
    res_utb = next((i for i, j in enumerate(training_status == 'untrainable') if j), None)
    if res is not None:
        """
        res_plus = 2
        if res+res_plus >= len(performance_easy):
            res_plus = np.abs(res-len(performance_easy))-1
        learning.loc[i,'learned'] = 'trained'
        learning.loc[i,'date_learned'] = session_start[res]
        learning.loc[i,'training_time'] = res
        learning.loc[i,'perf_easy'] = np.mean(performance_easy[res:res+res_plus])
        n_trials = [sum(s) for s in n_trials_stim]
        learning.loc[i,'n_trials'] = np.mean(n_trials[res:res+res_plus])
        learning.loc[i,'threshold'] = np.mean(threshold[res:res+res_plus])
        learning.loc[i,'bias'] = np.mean(bias[res:res+res_plus])
        learning.loc[i,'lapse_low'] = np.mean(lapse_low[res+res_plus])
        learning.loc[i,'lapse_high'] = np.mean(lapse_high[res:res+res_plus])
        """
        learning.loc[i,'learned'] = 'trained'
        learning.loc[i,'date_learned'] = session_start[res]
        learning.loc[i,'training_time'] = res
        learning.loc[i,'perf_easy'] = np.mean(performance_easy[res])
        n_trials = [sum(s) for s in n_trials_stim]
        learning.loc[i,'n_trials'] = np.mean(n_trials[res])
        learning.loc[i,'threshold'] = np.mean(threshold[res])
        learning.loc[i,'bias'] = np.mean(bias[res])
        learning.loc[i,'lapse_low'] = np.mean(lapse_low[res])
        learning.loc[i,'lapse_high'] = np.mean(lapse_high[res])
    elif res_utb is not None:
        learning.loc[i,'learned'] = 'untrainable'
        learning.loc[i,'date_learned'] = session_start[res_utb]
        learning.loc[i,'training_time'] = res_utb
        learning.loc[i,'perf_easy'] = performance_easy[res_utb]
        learning.loc[i,'n_trials'] = np.sum(n_trials_stim[res_utb])
        learning.loc[i,'threshold'] = threshold[res_utb]
        learning.loc[i,'bias'] = bias[res_utb]
        learning.loc[i,'lapse_low'] = lapse_low[res_utb]
        learning.loc[i,'lapse_high'] = lapse_high[res_utb]
    else:
        learning.loc[i,'learned'] = 'in training'
        learning.loc[i,'training_time'] = len(performance_easy)
        
    # Add mouse info to dataframe
    learning.loc[i,'mouse'] = nickname
    learning.iloc[i]['lab'] = subjects.iloc[i]['lab_name']
    
learned = learning[learning['learned'] == 'trained'] # Select mice that learned
#learned.loc[learned['lab'] == 'zadorlab','lab'] = 'churchlandlab'
learned = learned[learned['lab'] != 'cortexlab']
learned = learned[learned['lab'] != 'danlab']
learned = learned[learned['lab'] != 'wittenlab']
learned = learned[learned['lab'] != 'zadorlab']
for i in learned.index.values:
    learned.loc[i,'lab_n'] = learned.loc[i,'lab'][:-3] + ' lab' + '\n (n=' + str(sum(learned['lab'] == learned.loc[i,'lab'])) + ')'
for i in learning.index.values:
    learning.loc[i,'lab_n'] = learning.loc[i,'lab'][:-3] + ' lab' + '\n (n=' + str(sum(learning['lab'] == learning.loc[i,'lab'])) + ')'
    
# Initialize decoders
decod = learned
clf_rf = RandomForestClassifier(n_estimators=10)
clf_nb = GaussianNB()
clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# Perform decoding of lab membership
random_forest = []
bayes = []
lda = []
decoding_set = decod[['training_time', 'perf_easy','n_trials','threshold','bias']].values
for i in range(decod_it):
    random_forest.append(decoding(decoding_set, list(decod['lab']), clf_rf, num_splits))
    bayes.append(decoding(decoding_set, list(decod['lab']), clf_nb, num_splits))
    lda.append(decoding(decoding_set, list(decod['lab']), clf_lda, num_splits))

# Decode shuffeled labels 
shuf_rf = []
shuf_nb = []
shuf_lda = []
for i in range(shuffle_it):
    shuf_rf.append(decoding(decoding_set, list(decod['lab'].sample(frac=1)), clf_rf, num_splits))
    shuf_nb.append(decoding(decoding_set, list(decod['lab'].sample(frac=1)), clf_nb, num_splits))
    shuf_lda.append(decoding(decoding_set, list(decod['lab'].sample(frac=1)), clf_lda, num_splits))
   
perc = [np.percentile(random_forest-np.mean(shuf_rf),5), np.percentile(bayes-np.mean(shuf_nb),5), np.percentile(lda-np.mean(shuf_lda),5)]

# Set color palette
current_palette = sns.color_palette('Set1')
use_palette = [current_palette[-1]]*len(np.unique(learned['lab']))
all_color = [current_palette[5]]
use_palette = all_color + use_palette
sns.set_palette(use_palette)
sns.set(style='darkgrid', context='paper', font_scale=1.3, font='DejaVu Sans')

plt.figure()
fig = plt.gcf()
ax1 = plt.gca()
#ax1.plot([-1,5],[0,0],'--',color=[0.5,0.5,0.5])
ax1.plot([-1,5],[0,0],'r--')
sns.boxplot(x=['Random\nforest','Naive\nBayes','Linear\nDiscriminant\nAnalysis'], y=[random_forest-np.mean(shuf_rf), bayes-np.mean(shuf_nb), lda-np.mean(shuf_lda)], width=0.5, ax=ax1, color=use_palette[1])
ax1.set_ylabel('Decoding performance over chance level\n(F1 score)')
ax1.set_title('Decoding of lab membership')
ax1.set_ylim([-0.3, 0.3])
for item in ax1.get_xticklabels():
    item.set_rotation(60)
    
plt.tight_layout(pad = 2)
fig.set_size_inches((5, 6), forward=False) 
plt.savefig(join(path,'decoding_lab_membership.pdf'), dpi=300)
plt.savefig(join(path,'decoding_lab_membership.png'), dpi=300)

# Convert to float
learned['training_time'] = learned['training_time'].astype(float)
learned['perf_easy'] = learned['perf_easy'].astype(float)
learned['n_trials'] = learned['n_trials'].astype(float)
learned['threshold'] = learned['threshold'].astype(float)
learned['bias'] = learned['bias'].astype(float)
learned['lapse_low'] = learned['lapse_low'].astype(float)
learned['lapse_high'] = learned['lapse_high'].astype(float)

learend_2 = learned.copy()
learend_2['lab_n'] = 'All\n(n=%d)'%len(learned)
learned = learned.append(learend_2)
learned = learned.sort_values('lab_n')
learned['perf_easy'] = learned['perf_easy']*100

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(13,10), sharey=True)
sns.set_palette(use_palette)
  
sns.boxplot(x='perf_easy', y='lab_n', data=learned, ax=ax1)
ax1.set_title('Performance at easy contrasts (%)')
ax1.set_xlim([80, 100])
ax1.xaxis.tick_top()
ax1.set_ylabel('')
ax1.set_xlabel('')
for item in ax1.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='training_time', y='lab_n', data=learned, ax=ax2)
ax2.set_title('Time to reach trained criterion (sessions)')
ax2.xaxis.tick_top()
ax2.set_xlim([0, 30])
ax2.set_ylabel('')
ax2.set_xlabel('')
for item in ax2.get_yticklabels():
    item.set_rotation(40)
    
sns.boxplot(x='n_trials', y='lab_n', data=learned, ax=ax3)
ax3.set_title('Number of trials')
ax3.set_xlim([200, 1400])
ax3.xaxis.tick_top()
ax3.set_ylabel('')  
ax3.set_xlabel('') 
for item in ax3.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='threshold', y='lab_n', data=learned, ax=ax4)
ax4.set_title('Visual threshold (% contrast)')
ax4.set_xlim([0, 30])
ax4.xaxis.tick_top()
ax4.set_ylabel('')
ax4.set_xlabel('')
for item in ax4.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='bias', y='lab_n', data=learned, ax=ax5)
ax5.set_title('Bias (% contrast)')
ax5.xaxis.tick_top()
ax5.set_xlim([-25, 25])
ax5.set_ylabel('')  
ax5.set_xlabel('')  
for item in ax5.get_yticklabels():
    item.set_rotation(40)

plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((12, 8), forward=False)
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()

plt.savefig(join(path, 'learning_params_vertical.pdf'), dpi=300)
plt.savefig(join(path, 'learning_params_vertical.png'), dpi=300)

"""

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(13,10), sharex=True)
  
sns.boxplot(x='lab_n', y='perf_easy', data=learned, ax=ax1)
ax1.set_ylabel('Performance at easy contrast (%)')
for item in ax1.get_xticklabels():
    item.set_rotation(60)
ax1.set_xlabel('')
ax1.set_ylim([80,100])

sns.boxplot(x='lab_n', y='training_time', data=learned, ax=ax2)
ax2.set_ylabel('Time to reach trained criterion (days)')
for item in ax2.get_xticklabels():
    item.set_rotation(60)
ax2.set_xlabel('')
ax2.set_ylim([0,30])
    
sns.boxplot(x='lab_n', y='n_trials', data=learned, ax=ax3)
ax3.set_ylabel('Number of trials')
for item in ax3.get_xticklabels():
    item.set_rotation(60)
ax3.set_xlabel('')
ax3.set_ylim([0,1600])
    
sns.boxplot(x='lab_n', y='threshold', data=learned, ax=ax4)
ax4.set_ylabel('Visual threshold (% contrast)')
for item in ax4.get_xticklabels():
    item.set_rotation(60)
ax4.set_xlabel('')
ax4.set_ylim([0,30])
    
sns.boxplot(x='lab_n', y='bias', data=learned, ax=ax5)
ax5.set_ylabel('Bias')
for item in ax5.get_xticklabels():
    item.set_rotation(60)
ax5.set_xlabel('')
ax5.set_ylim([-30,30])
    
sns.boxplot(x='lab_n', y='lapse_high', data=learned, ax=ax6)
ax6.set_ylabel('Lapse rate')
for item in ax6.get_xticklabels():
    item.set_rotation(60)
ax6.set_xlabel('')
ax6.set_ylim([0,0.25])
    
plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((12, 8), forward=False)
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()

plt.savefig(join(path, 'learning_params.pdf'), dpi=300)
plt.savefig(join(path, 'learning_params.png'), dpi=300)

"""

