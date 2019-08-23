#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:28:01 2019

@author: ibladmin
"""

# @alejandropan 2019

#Import some general packages

import time, re, datetime, os, glob
from datetime import timedelta
import seaborn as sns
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import embed as shell
from scipy import stats

## CONNECT TO datajoint

import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
sys.path.insert(0, '/Users/alex/Documents/PYTHON/analysis/python/')
sys.path.insert(0, '/Users/alex/Documents/PYTHON/analysis/paper-behavior/')
sys.path.insert(0, '/Users/alex/Documents/PYTHON/analysis/paper-behavior/figure8/')

from dj_tools import *

import datajoint as dj
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from alexfigs_datajoint_functions import *  # this has all plotting functions


#Collect all all data 
use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project="ibl_neuropixel_brainwide_01"'
sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%trainingChoiceWorld%"') \
	* use_subjects
b 		= (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat 	= pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
allsubjects 	= dj2pandas(bdat)





#Add learning rate columns
allsubjects['training_status'] =np.nan
allsubjects['days_to_trained'] = np.nan
allsubjects['trials_to_trained'] = np.nan
allsubjects['days_to_ephys'] = np.nan
allsubjects['trials_to_ephys'] = np.nan


#Add bias (level2) columns
allsubjects['average_bias08'] =np.nan
allsubjects['average_bias02'] =np.nan
allsubjects['average_threshold'] =np.nan
allsubjects['average_lapse_high'] =np.nan
allsubjects['average_lapse_low'] =np.nan



users  =  allsubjects['lab_name'].unique()

for labname in users:
    for mouse in allsubjects['subject_nickname']:
            try:
                # TRIAL COUNTS AND SESSION DURATION
                behav = get_behavior(mouse, labname)
                # check whether the subject is trained based the the lastest session
                subj = subject.Subject & 'subject_nickname="{}"'.format(mouse)
                last_session = subj.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')
                training_status = (behavior_analysis.SessionTrainingStatus & last_session).fetch1('training_status')
                
                if training_status in ['trained', 'ready for ephys']:
                    first_trained_session = subj.aggr(behavior_analysis.SessionTrainingStatus & 'training_status="trained"', first_trained='min(session_start_time)')
                    first_trained_session_time = first_trained_session.fetch1('first_trained')
                    # convert to timestamp
                    trained_date = pd.DatetimeIndex([first_trained_session_time])[0]
                    # how many days to training?
                    days_to_trained = sum(behav['date'].unique() < trained_date.to_datetime64())
                    # how many trials to trained?
                    trials_to_trained = sum(behav['date'] < trained_date.to_datetime64())
                       
                    #average threshold
                    pars = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate.PsychResults * subject.Subject * subject.SubjectLab & \
                                         'subject_nickname="%s"'%mouse & 'lab_name="%s"'%labname).fetch(as_dict=True))
                    average_threshold  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'threshold'].mean()
                    average_lapse_high  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'lapse_high'].mean()
                    average_lapse_low  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'lapse_low'].mean()
                else:   
                    days_to_trained = np.nan
                    trials_to_trained = np.nan
                    average_threshold = np.nan
                    average_lapse_high = np.nan
                    average_lapse_low = np.nan
    
                if training_status == 'ready for ephys':
                    #Only counting from ready to ephys status
                    first_ephystrained_session = subj.aggr(behavior_analysis.SessionTrainingStatus & \
                                                           'training_status="ready for ephys"', first_ephystrained='min(session_start_time)')
                    first_ephystrained_session_time = first_ephystrained_session.fetch1('first_ephystrained')
                    # trials to ready for ephys
                    ephys_date = pd.DatetimeIndex([first_ephystrained_session_time])[0]
                    days_to_ephys = sum((behav['date'].unique() < ephys_date.to_datetime64()) & (behav['date'].unique() > trained_date.to_datetime64()))
                    trials_to_ephys = sum((behav['date'] < ephys_date.to_datetime64()) & (behav['date'] > trained_date.to_datetime64()))
                    
                    #Bias analysis
                
                    pars = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate.PsychResults * \
                                         subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%mouse & \
                                         'lab_name="%s"'%labname).fetch(as_dict=True))
                    average_bias_08  = pars.loc[(pars['prob_left'] == 0.8) & (pars['session_date'] \
                                                >= first_ephystrained_session_time.date()), 'bias'].mean()
                    average_bias_02  = pars.loc[(pars['prob_left'] == 0.2) & (pars['session_date'] \
                                                >= first_ephystrained_session_time.date()), 'bias'].mean()
                    
                else:
                    average_bias_08 = np.nan
                    average_bias_02= np.nan
                    days_to_ephys = np.nan
                    trials_to_ephys= np.nan
                    
                # keep track
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['days_to_trained']] = days_to_trained
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['trials_to_trained']] = trials_to_trained
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['days_to_ephys']] = days_to_ephys
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['trials_to_ephys']] = trials_to_ephys
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['training_status']] = training_status
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_threshold']] = average_threshold
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_lapse_high']] = average_lapse_high
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_lapse_low']] = average_lapse_low
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_bias08']] = average_bias_08
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_bias02']] = average_bias_02
                
            except:
                pass
        
#Star plotting
#Make sublist with labs that have trained males and female
#TODO dectect this condition automatically
allsubjects['sex of the experimenter'] = "F"
allsubjects.loc[((allsubjects['lab_name']== 'cortexlab') | (allsubjects['lab_name']== 'wittenlab')|(allsubjects['lab_name']=='angelakilab') | (allsubjects['lab_name']=='danlab')), ['sex of the experimenter']] = "M"
#labs that have trained males and females (TODO: detect this automatically)
subjects_mixed = allsubjects.loc[((allsubjects['lab_name']== 'churchlandlab')|(allsubjects['lab_name']=='wittenlab') | (allsubjects['lab_name']=='angelakilab') | (allsubjects['lab_name']=='cortexlab'))]


##Plots per session
#Total - day/trials
sns.set()
total_day = plt.figure(figsize=(20,20))
total_day.add_subplot(221)
sns.boxplot(x="sex", y="days_to_trained", data=allsubjects )
sns.swarmplot(x="sex", y="days_to_trained", data=allsubjects,hue="lab_name", edgecolor="white")
plt.ylabel('sessions to trained (sessions)')
total_day.add_subplot(222)
sns.boxplot(x="sex", y="days_to_ephys", data=allsubjects )
sns.swarmplot(x="sex", y="days_to_ephys", data=allsubjects,hue="lab_name", edgecolor="white")
plt.ylabel('sessions to ephys from trained (sessions)')


total_day.add_subplot(223)
sns.boxplot(x="sex", y="trials_to_trained", data=allsubjects )
sns.swarmplot(x="sex", y="trials_to_trained", data=allsubjects,hue="lab_name", edgecolor="white")
plt.ylabel('trials to trained (trials)')
total_day.add_subplot(224)
sns.boxplot(x="sex", y="trials_to_trained", data=allsubjects )
sns.swarmplot(x="sex", y="trials_to_trained", data=allsubjects,hue="lab_name", edgecolor="white")
plt.ylabel('trials to trained (sessions)')

total_day.savefig("total_day.pdf", bbox_inches='tight')
total_day.savefig("total_day.png", bbox_inches='tight')

#Per Lab - day/trials trained
figtrained = plt.figure(figsize=(20,20))
figtrained.add_subplot(221)
sns.boxplot(x="lab_name", y="days_to_trained", hue="sex",
            data=allsubjects)
plt.xticks(rotation=90)
plt.ylabel('sessions to trained (sessions)')
plt.xlabel('')
figtrained.add_subplot(222)
sns.boxplot(x="lab_name", y="trials_to_trained", hue="sex",
            data=allsubjects)
plt.xticks(rotation=90)
plt.ylabel('trials  to trained (trials)')


#Per Lab - day/trials ephys
figtrained.add_subplot(223)
sns.boxplot(x="lab_name", y="days_to_ephys", hue="sex",
            data=allsubjects)
plt.xticks(rotation=90)
plt.ylabel('sessions from trained to ephys (sessions)')
plt.xlabel('')
figtrained.add_subplot(224)
sns.boxplot(x="lab_name", y="trials_to_ephys", hue="sex",
            data=allsubjects)
plt.xticks(rotation=90)
plt.ylabel('trials from trained to ephys (trials)')


figtrained.savefig("figdaystrained.pdf", bbox_inches='tight')
figtrained.savefig("figdaystrained.png", bbox_inches='tight')


#sessions to trained pooled

sns.set()
figtrained1 = plt.figure(figsize=(10,10))
figtrained1.add_subplot()
sns.swarmplot(x="lab_name", y="days_to_ephys",
            data=allsubjects, color = 'black')
sns.boxplot(x="lab_name", y="days_to_ephys",
            data=allsubjects)
plt.xticks(rotation=90)
plt.ylabel('sessions from trained to ephys (sessions)')
plt.xlabel('')

figtrained1.savefig("figdaystrained_pooled.pdf", bbox_inches='tight')



#Interaction - day
interaction_trial = plt.figure(figsize=(10,6))
interaction_trial.add_subplot(211)
sns.catplot(x="sex", y="days_to_trained",col='sex of the experimenter', hue="lab_name",  data=subjects_mixed)
interaction_trial.add_subplot(212)
sns.catplot(x="sex", y="days_to_ephys",col='sex of the experimenter',  hue="lab_name",data=subjects_mixed)

interaction_trial.savefig("figdaystrained.pdf", bbox_inches='tight')
interaction_trial.savefig("figdaystrained.png", bbox_inches='tight')