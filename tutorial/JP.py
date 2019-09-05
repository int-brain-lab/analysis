"""
Created on Sept 5th

I have no idea yet, this is just practice

@author: Jean-Paul Noel
"""
# NOT SPECIFIC TO IBL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# IBL SPECIFIC STUFF
import datajoint as dj
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as ba

# APPENDING TO THE PATH WHERE TUTORIAL ARE
tutorial_path = os.path.join(os.path.expanduser('~'), 'Documents', 'IBL-JP-CODE', 'analysis')
sys.path.append(tutorial_path)

# FOR PLOTTING
NROW = 1
NCOL = 7
counter = -1
xvec  = np.arange(-100, 100)

# GETTING ALL THE ANGELAKI LAB ANIMALS
use_subjects = pd.DataFrame(((subject.Subject * subject.SubjectLab * subject.SubjectProject) & 'lab_name = "angelakilab"' & 'sex!="U"' & 'subject_project = "ibl_behaviour_pilot_matlabrig"').fetch())
subjects = use_subjects['subject_nickname']

# INITIALIZING A FIGURE
fig1, axs1 = plt.subplots(NROW, NCOL, constrained_layout=True); fig1.suptitle('Easy Performance')
fig2, axs2 = plt.subplots(NROW, NCOL, constrained_layout=True); fig2.suptitle('Threshold')
fig3, axs3 = plt.subplots(NROW, NCOL, constrained_layout=True); fig3.suptitle('Bias')
fig4, axs4 = plt.subplots(NROW, NCOL, constrained_layout=True); fig4.suptitle('Lapse Low')
fig5, axs5 = plt.subplots(NROW, NCOL, constrained_layout=True); fig5.suptitle('Lapse High')
fig6, axs6 = plt.subplots(NROW, NCOL, constrained_layout=True); fig6.suptitle('Psychometric')


for i, mouse in enumerate(subjects):
    print('Loading data of subject %s'%(mouse))

    # Gather subject info
    subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%mouse
    behav = pd.DataFrame((ba.BehavioralSummaryByDate * subject.Subject * subject.SubjectLab &
       'subject_nickname="%s"'%mouse).proj('session_date', 'performance_easy', 'subject_nickname', 'lab_name').fetch(as_dict=True, order_by='session_date'))
    
    fit = pd.DataFrame(subject.Subject * subject.SubjectLab * ba.PsychResults & 'subject_nickname="%s"'%mouse)

    # fit.signed_contrasts  
    # fit.session_start_time 

    # hint to JP the noob: print(df.columns.values) shows all the columns

    if behav.empty:
        continue
    
    # PLOTTING PERFORMANCE ON EASY TRIALS
    counter += 1
    axs1[counter].plot(behav['performance_easy']); axs1[counter].set_title(mouse)
    axs2[counter].plot(fit['threshold']); axs2[counter].set_title(mouse)
    axs3[counter].plot(fit['bias']); axs3[counter].set_title(mouse)
    axs4[counter].plot(fit['lapse_low']); axs4[counter].set_title(mouse)
    axs5[counter].plot(fit['lapse_high']); axs5[counter].set_title(mouse)

    
    psycho = pd.DataFrame({'signed_contrasts':xvec, \
        'choice':psy.erf_psycho_2gammas([fit['bias'].iloc[-1],fit['threshold'].iloc[-1],fit['lapse_low'].iloc[-1],fit['lapse_high'].iloc[-1]], xvec)})

    axs6[counter].plot(psycho.choice); axs6[counter].set_title(mouse)
    axs6[counter].scatter(fit['signed_contrasts'].iloc[-1]*100+100, fit['prob_choose_right'].iloc[-1]) 

print(0)
