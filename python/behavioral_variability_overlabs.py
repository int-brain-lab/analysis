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
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
path = '/home/guido/Figures/Behavior/'

# Query list of subjects
all_sub = subject.Subject * subject.SubjectLab & 'subject_birth_date > "2018-09-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'
#all_sub = subject.Subject * subject.SubjectLab & 'subject_nickname = "ZM_1742"'
subjects = pd.DataFrame(all_sub)
        
learning = pd.DataFrame(columns=['mouse','lab','learned','date_learned','training_time','perf_easy','n_trials','threshold','bias','reaction_time','lapse_low','lapse_high'])
for i, nickname in enumerate(subjects['subject_nickname']):
    print('Processing subject %s'%nickname)
    
    # Gather behavioral data for subject
    subj = subject.Subject & 'subject_nickname="%s"'%nickname
    behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname).proj('session_date', 'performance_easy').fetch(as_dict=True, order_by='session_date'))
    rt = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.ReactionTimeByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname)).proj('session_date', 'median_reaction_time').fetch(as_dict=True, order_by='session_date'))
    psych = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.PsychResults * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname)).proj('session_date', 'n_trials_stim','threshold','bias','lapse_low','lapse_high').fetch(as_dict=True, order_by='session_date'))
    
    # Find first session in which mouse is trained
    first_trained_session = subj.aggr(behavior_analysis.SessionTrainingStatus &	'training_status="trained"', first_trained='DATE(min(session_start_time))')
    untrainable_session = subj.aggr(behavior_analysis.SessionTrainingStatus & 'training_status="untrainable"', first_trained='DATE(min(session_start_time))')
    ephys_session = subj.aggr(behavior_analysis.SessionTrainingStatus & 'training_status="ready for ephys"', first_trained='DATE(min(session_start_time))')
    if len(first_trained_session) == 0 & len(untrainable_session) == 0:
        learning.loc[i,'learned'] = 'in training'
        learning.loc[i,'training_time'] = len(behav)
    elif len(first_trained_session) == 0 & len(untrainable_session) == 1:
        learning.loc[i,'learned'] = 'untrainable'
        learning.loc[i,'training_time'] = len(behav)
    else:
        first_trained_session_time = first_trained_session.fetch1('first_trained')    
        learning.loc[i,'learned'] = 'trained'
        learning.loc[i,'date_learned'] = first_trained_session_time
        learning.loc[i,'training_time'] = sum(behav.session_date < first_trained_session_time)
        learning.loc[i,'perf_easy'] = float(behav.performance_easy[behav.session_date == first_trained_session_time])*100
        psych['n_trials'] = n_trials = [sum(s) for s in psych.n_trials_stim]
        learning.loc[i,'n_trials'] = float(psych.n_trials[psych.session_date == first_trained_session_time])
        learning.loc[i,'threshold'] = float(psych.threshold[psych.session_date == first_trained_session_time])
        learning.loc[i,'bias'] = float(psych.bias[psych.session_date == first_trained_session_time])
        learning.loc[i,'lapse_low'] = float(psych.lapse_low[psych.session_date == first_trained_session_time])
        learning.loc[i,'lapse_high'] = float(psych.lapse_high[psych.session_date == first_trained_session_time])
        if sum(rt.session_date == first_trained_session_time) == 0:
            learning.loc[i,'reaction_time'] = float(rt.median_reaction_time[np.argmin(np.array(abs(rt.session_date - first_trained_session_time)))])*1000
        else:
            learning.loc[i,'reaction_time'] = float(rt.median_reaction_time[rt.session_date == first_trained_session_time])*1000
    if len(ephys_session) > 0:
        first_ephys_session_time = ephys_session.fetch1('first_trained')  
        learning.loc[i,'learned'] = 'ephys'
        learning.loc[i,'date_ephys'] = first_ephys_session_time
        learning.loc[i,'days_trained_ephys'] = sum((behav.session_date > first_trained_session_time) & (behav.session_date < first_ephys_session_time))
        
    # Add mouse info to dataframe
    learning.loc[i,'mouse'] = nickname
    learning.iloc[i]['lab'] = subjects.iloc[i]['lab_name']
    
# Select mice that learned
learned = learning[learning['learned'] == 'trained']
learned = learned.append(learning[learning['learned'] == 'ephys'])

# Merge some labs
learned.loc[learned['lab'] == 'zadorlab','lab'] = 'churchlandlab'
learned.loc[learned['lab'] == 'mrsicflogellab','lab'] = 'cortexlab'

# Rename labs
learned.loc[learned['lab'] == 'angelakilab','lab'] = 'NYU'
learned.loc[learned['lab'] == 'churchlandlab','lab'] = 'CSHL'
learned.loc[learned['lab'] == 'cortexlab','lab'] = 'UCL'
learned.loc[learned['lab'] == 'danlab','lab'] = 'Berkeley'
learned.loc[learned['lab'] == 'mainenlab','lab'] = 'CCU'
learned.loc[learned['lab'] == 'wittenlab','lab'] = 'Princeton'

for i in learned.index.values:
    learned.loc[i,'lab_n'] = learned.loc[i,'lab'] + ' (n=' + str(sum(learned['lab'] == learned.loc[i,'lab'])) + ')'

# Convert to float
learned['training_time'] = learned['training_time'].astype(float)
learned['perf_easy'] = learned['perf_easy'].astype(float)
learned['n_trials'] = learned['n_trials'].astype(float)
learned['threshold'] = learned['threshold'].astype(float)
learned['bias'] = learned['bias'].astype(float)
learned['lapse_low'] = learned['lapse_low'].astype(float)
learned['lapse_high'] = learned['lapse_high'].astype(float)
learned['reaction_time'] = learned['reaction_time'].astype(float)

# Set color palette
current_palette = sns.color_palette('Set1')
use_palette = [current_palette[-1]]*len(np.unique(learned['lab']))
all_color = [current_palette[5]]
use_palette = all_color + use_palette
sns.set_palette(use_palette)
sns.set(style='darkgrid', context='paper', font_scale=1.3, font='DejaVu Sans')

# Plot metrics over all mice
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12,8))
sns.boxplot(y=learned.perf_easy, ax=ax1, color=use_palette[1], width=0.5)
ax1.set(ylabel='Perf. at easy contrasts (%)', xlabel='', ylim=[80, 101], yticks=[80,85,90,95,100])
sns.boxplot(y=learned.training_time, ax=ax2, color=use_palette[1], width=0.5)
ax2.set(ylabel='Training duration (sessions)', xlabel='', ylim=[0, 60])
sns.boxplot(y=learned.n_trials, ax=ax3, color=use_palette[1], width=0.5)
ax3.set(ylabel='Number of trials', xlabel='', ylim=[0, 1600], yticks=np.arange(0,1601,400))
sns.boxplot(y=learned.threshold, ax=ax4, color=use_palette[1], width=0.5)
ax4.set(ylabel='Visual threshold (% contrast)', xlabel='', ylim=[0, 30])
sns.boxplot(y=learned.bias, ax=ax5, color=use_palette[1], width=0.5)
ax5.set(ylabel='Bias (% contrast)', xlabel='', ylim=[-20, 20])
sns.boxplot(y=learned.reaction_time, ax=ax6, color=use_palette[1], width=0.5)
ax6.set(ylabel='Reaction time (ms)', xlabel='', ylim=[0, 1000])

plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((10, 6), forward=False)
plt.savefig(join(path, 'learning_params_all.pdf'), dpi=300)
plt.savefig(join(path, 'learning_params_all.png'), dpi=300)

# Add all mice to dataframe seperately for plotting
learned_2 = learned.copy()
learned_2['lab_n'] = 'All (n=%d)'%len(learned)
learned_2 = learned.append(learned_2)
learned_2 = learned_2.sort_values('lab_n')

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(13,10), sharey=True)
sns.set_palette(use_palette)
  
sns.boxplot(x='perf_easy', y='lab_n', data=learned_2, ax=ax1)
ax1.set_title('Performance at easy contrasts (%)')
ax1.set_xlim([80, 101])
ax1.xaxis.tick_top()
ax1.set_ylabel('')
ax1.set_xlabel('')
for item in ax1.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='training_time', y='lab_n', data=learned_2, ax=ax2)
ax2.set_title('Time to reach trained criterion (sessions)')
ax2.xaxis.tick_top()
ax2.set_xlim([0, 60])
ax2.set_ylabel('')
ax2.set_xlabel('')
for item in ax2.get_yticklabels():
    item.set_rotation(40)
    
sns.boxplot(x='n_trials', y='lab_n', data=learned_2, ax=ax3)
ax3.set_title('Number of trials')
ax3.set_xlim([0, 1600])
ax3.xaxis.tick_top()
ax3.set_ylabel('')  
ax3.set_xlabel('') 
for item in ax3.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='threshold', y='lab_n', data=learned_2, ax=ax4)
ax4.set_title('Visual threshold (% contrast)')
ax4.set_xlim([0, 40])
ax4.xaxis.tick_top()
ax4.set_ylabel('')
ax4.set_xlabel('')
for item in ax4.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='bias', y='lab_n', data=learned_2, ax=ax5)
ax5.set_title('Bias (% contrast)')
ax5.xaxis.tick_top()
ax5.set_xlim([-30, 30])
ax5.set_ylabel('')  
ax5.set_xlabel('')  
for item in ax5.get_yticklabels():
    item.set_rotation(40)

sns.boxplot(x='reaction_time', y='lab_n', data=learned_2, ax=ax6)
ax6.set_title('Reaction time (ms)')
ax6.xaxis.tick_top()
ax6.set_xlim([0, 1000])
ax6.set_ylabel('')  
ax6.set_xlabel('')  
for item in ax6.get_yticklabels():
    item.set_rotation(40)

plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((12, 8), forward=False)
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()

plt.savefig(join(path, 'learning_params.pdf'), dpi=300)
plt.savefig(join(path, 'learning_params.png'), dpi=300)

# Z-score data
learned_zs = pd.DataFrame()
learned_zs['lab_n'] = learned['lab_n']
learned_zs['Training time'] = stats.zscore(learned['training_time'])
learned_zs['Performance'] = stats.zscore(learned['perf_easy'])
learned_zs['# of trials'] = stats.zscore(learned['n_trials'])
learned_zs['Threshold'] = stats.zscore(learned['threshold'])
learned_zs['Bias'] = stats.zscore(learned['bias'])
learned_zs['Reaction time'] = stats.zscore(learned['reaction_time'])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), sharey=True)
sns.heatmap(data=learned_zs.groupby('lab_n').mean(), vmin=-1, vmax=1, cmap=sns.color_palette("coolwarm", 100), 
            cbar_kws={"ticks":[-1,-0.5,0,0.5,1]}, ax=ax1)
            #cbar_kws={'label':'z-scored mean', "ticks":[-1,-0.5,0,0.5,1]}, ax=ax1)
ax1.set(ylabel='', title='')
ax1.set(ylabel='', title='Mean per lab (z-scored)')
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=50, ha="right" )

    
sns.heatmap(data=learned_zs.groupby('lab_n').std(), vmin=0, vmax=2, cmap=sns.color_palette("coolwarm", 100), 
            cbar_kws={"ticks":[0,0.5,1,1.5,2]}, ax=ax2)
            #cbar_kws={'label':'z-scored std', "ticks":[0,0.5,1,1.5,2]}, ax=ax2)
ax2.set(ylabel='', title='Variance per lab (z-scored)')
plt.setp(ax2.yaxis.get_majorticklabels(), rotation=40)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=50, ha="right" )

plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((10,5), forward=False)
plt.savefig(join(path, 'variability_heatmap.pdf'), dpi=300)
plt.savefig(join(path, 'variability_heatmap.png'), dpi=300)




