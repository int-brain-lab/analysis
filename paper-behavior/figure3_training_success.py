#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:00:42 2019

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
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = '/home/guido/Figures/Behavior/'
data_path = '/home/guido/Data/Behavior/'

# Query list of subjects
use_subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject * subject.Death
                & 'subject_project = "ibl_neuropixel_brainwide_01"')
# use_subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject
#                & 'subject_project = "ibl_neuropixel_brainwide_01"')
subjects = use_subjects.fetch('subject_nickname')

trained_df = pd.DataFrame(columns=['mouse', 'lab', 'trained'])
for i, nickname in enumerate(subjects):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects)))

    training = (behavior_analysis.SessionTrainingStatus * subject.Subject
                & 'subject_nickname="%s"' % nickname).fetch('training_status')
    if len(training) == 0 or np.sum(training == 'trained') == 0:
        trained_df.loc[i, 'trained'] = False
    else:
        trained_df.loc[i, 'trained'] = True
    trained_df.loc[i, 'mouse'] = nickname
    trained_df.loc[i, 'lab'] = (use_subjects & 'subject_nickname="%s"' % nickname).fetch(
                                                                                    'lab_name')[0]
