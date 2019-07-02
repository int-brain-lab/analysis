#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:11:01 2019

@author: guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datajoint as dj
from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analysis
import sys
sys.path.insert(0, '../python')
from fit_learning_curves import fit_learningcurve, plot_learningcurve


# Query list of subjects
all_sub = subject.Subject * subject.SubjectLab & 'subject_birth_date > "2018-09-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'
subjects = all_sub.fetch('subject_nickname')

# Create dataframe with behavioral metrics of all mice        
fitted_curve = pd.DataFrame()
for i, nickname in enumerate(subjects):
    if np.mod(i+1,10) == 0: 
        print('Loading data of subject %d of %d'%(i+1,len(subjects)))
    
    subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%nickname
    behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%nickname).proj('session_date', 'performance_easy','subject_nickname').fetch(as_dict=True, order_by='session_date'))
    behav['session_day'] = behav.index.array+1
    fitted_curve = fitted_curve.append(fit_learningcurve(behav, nickname))
    
    