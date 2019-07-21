"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell # for debugging
from scipy.special import erf # for psychometric functions
import datetime

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Arial')
sns.set(style="darkgrid", context="paper")
sns.set(style="darkgrid", context="paper", font_scale=1.3)

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *
new_criteria = dj.create_virtual_module('analyses', 'group_shared_anneurai_analyses')

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# 1. get training status from original DJ table
# ================================= #

use_subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
sess = behavioral_analyses.SessionTrainingStatus() \
 * use_subjects * subject.SubjectLab * subject.Subject.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')

df = pd.DataFrame(sess.fetch(as_dict=True))
df2 = df.groupby(['training_status'])['subject_uuid'].count().reset_index()
# remove the erroneous mice that were wrongly trained on biasedChoiceWorld already
# df2 = df2[~df2.training_status.str.contains("wrong session type run")]
df2 = df2.replace({'over40days': 'untrainable'})
df2 = df2.sort_values('training_status')
print(df2)

# QUICK PIE PLOT
sns.set_palette("husl")
fig, ax = plt.subplots(1, 2, figsize=(13,13))

# ================================= #
# v0
# ================================= #

sess = behavioral_analyses.SessionTrainingStatus() \
 * use_subjects * subject.SubjectLab * subject.Subject.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')
df = pd.DataFrame(sess.fetch(as_dict=True))
df2 = df.groupby(['training_status'])['subject_uuid'].count().reset_index()
df2.index = df2.training_status
df2 = df2.reindex(['ready for ephys', 'trained', 'training in progress', 'over40days'])

print(df2)

ax[0].pie(df2['subject_uuid'], autopct='%1.2f%%', labels=df2['training_status'])
ax[0].set_title('Original criteria (v0), n = %d'%df2['subject_uuid'].sum())

# ================================= #
# v1
# ================================= #

sess = new_criteria.SessionTrainingStatus() \
 * use_subjects * subject.SubjectLab * subject.Subject.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')
df = pd.DataFrame(sess.fetch(as_dict=True))
df2 = df.groupby(['training_status'])['subject_uuid'].count().reset_index()
df2.index = df2.training_status
df2 = df2.reindex(['ready4ephysrig', 'trained_1b', 'trained_1a', 'intraining', 'untrainable'])
print(df2)

ax[1].pie(df2['subject_uuid'], autopct='%1.2f%%', labels=df2['training_status'])
ax[1].set_title('Alternative criteria (v1), n = %d'%df2['subject_uuid'].sum())

# ================================= #

fig.savefig(os.path.join(figpath, "training_success.pdf"))
fig.savefig(os.path.join(figpath, "training_success.png"), dpi=300)
plt.close('all')
