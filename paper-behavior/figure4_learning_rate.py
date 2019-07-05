#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:11:01 2019

Fit cumulative gaussian to learning curves

@author: guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figure_style import seaborn_style
from os.path import join
import datajoint as dj
from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analysis
import sys
sys.path.insert(0, '../python')
from fit_learning_curves import fit_learningcurve, plot_learningcurve

# Set path for figure 
path = '/home/guido/Figures/Behavior/'

# Query list of subjects
use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
subjects = use_subjects.fetch('subject_nickname')

# Create dataframe with behavioral metrics of all mice        
df_learning = pd.DataFrame()
for i, nickname in enumerate(subjects):
    if np.mod(i+1,10) == 0: 
        print('Loading data of subject %d of %d'%(i+1,len(subjects)))
    # Gather subject info
    subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%nickname
    behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname).proj('session_date', 'performance_easy','subject_nickname').fetch(as_dict=True, order_by='session_date'))
    
    # Fit learning curve
    behav['session_day'] = behav.index.array+1
    fitted_curve = fit_learningcurve(behav, nickname)
    
        
    # Get whether and when mouse is considered trained
    first_trained_session = subj.aggr(behavior_analysis.SessionTrainingStatus &	'training_status="trained"', first_trained='DATE(min(session_start_time))')
    if len(first_trained_session) == 0:
        fitted_curve.loc[nickname, 'day_trained'] = np.nan
    else:
        first_trained_session_date = first_trained_session.fetch1('first_trained') 
        fitted_curve.loc[nickname, 'day_trained'] = sum(behav.session_date < first_trained_session_date)

    df_learning = df_learning.append(fitted_curve)

# Select trained mice    
df_learned = df_learning[df_learning['day_trained'].notnull()]

# Get example mouse data
example_mouse = 'ZM_1091'
#example_mouse = 'NYU-06'
subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%example_mouse
behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%example_mouse).proj('session_date', 'performance_easy','subject_nickname').fetch(as_dict=True, order_by='session_date'))
behav['session_day'] = behav.index.array+1

# Plotting
seaborn_style()
f, (ax1, ax2) = plt.subplots(1,2,figsize=[9,5])
plot_learningcurve(behav['session_day'], behav['performance_easy'], example_mouse, ax1)
ax1.set(ylabel='Performance at easy trials (%)', xlabel='Training sessions', xlim=[0,20], ylim=[0.4, 1], xticks=np.arange(0,21,5))

sns.boxplot(data=df_learned[['delay', 'rise', 'asymp', 'day_trained']], color=[0.6,0.6,0.6], ax=ax2)
ax2.set(ylabel='Training sessions', ylim=[0,50], xticklabels=['Delay', 'Rise', 'Asymptotic\nperformance','Reached trained\ncriterium'])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad = 3)
fig = plt.gcf()
fig.set_size_inches((9,5), forward=False)
plt.savefig(join(path, 'figure4_learning_rate.pdf'), dpi=300)
plt.savefig(join(path, 'figure4_learning_rate.png'), dpi=300)










    