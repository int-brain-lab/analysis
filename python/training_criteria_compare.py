"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import time, os
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell # for debugging

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Arial')
sns.set(style="darkgrid", context="paper")
sns.set(style="darkgrid", context="paper", font_scale=1.3)

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *
new_criteria = dj.create_virtual_module('analyses', 'user_anneurai_analyses')

figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# 1. get training status from original DJ table
# ================================= #

use_subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
use_sessions = use_subjects * subject.SubjectLab * subject.Subject.aggr(behavior.TrialSet,
                                                          session_start_time='max(session_start_time)')

# QUICK PIE PLOT
plt.close('all')
sns.set_palette("cubehelix")
fig, ax = plt.subplots(1, 2, figsize=(13, 13))

# ================================= #
# v0
# ================================= #

sess = behavioral_analyses.SessionTrainingStatus() * use_sessions
df1 = pd.DataFrame(sess.fetch(as_dict=True))

df2 = df1.groupby(['training_status'])['subject_uuid'].count().reset_index()
df2.index = df2.training_status
df2 = df2.reindex(['-', 'over40days', 'training in progress', 'trained', '.', 'ready for ephys', '..'])
df2.fillna(0, inplace=True)
original = df2.copy()
print(df2)

ax[0].pie(df2['subject_uuid'], autopct='%1.2f%%', labels=df2.index)
ax[0].set_title('Original criteria (v0), n = %d'%df2['subject_uuid'].sum())

# ================================= #
# v1
# ================================= #

# sns.set_palette("Set2")

sess = new_criteria.SessionTrainingStatus() * use_sessions
df3 = pd.DataFrame(sess.fetch(as_dict=True))

df4 = df3.groupby(['training_status'])['subject_uuid'].count().reset_index()
df4.index = df4.training_status
df4 = df4.reindex(['unbiasable', 'untrainable', 'in_training', 'trained_1a', 'trained_1b',
                   'ready4ephysrig', 'ready4recording'])
df4.fillna(0, inplace=True)
print(df4)
new = df4.copy()

ax[1].pie(df4['subject_uuid'], autopct='%1.2f%%', labels=df4.index)
ax[1].set_title('Alternative criteria (v1), n = %d'%df4['subject_uuid'].sum())

fig.savefig(os.path.join(figpath, "training_success.pdf"))
fig.savefig(os.path.join(figpath, "training_success.png"), dpi=300)
plt.close('all')

# ================================= #
# COMPARE PER LAB
# ================================= #

# WRITE A SUMMARY DOCUMENT
df5 = pd.merge(df1, df3, on='subject_nickname')
df5 = df5[['subject_nickname', 'lab_name_x', 'training_status_x', 'training_status_y']]
df5 = df5.sort_values(by=['lab_name_x', 'subject_nickname'])
df5.to_csv(os.path.join(figpath, "criteria_comparison.csv"))

# ================================= #
# COMPARE PER LAB
# ================================= #

sys.path.insert(0, '/Users/urai/Documents/code/postdoc-analyses/')
import aging_tables

progression_old = pd.DataFrame((behavioral_analyses.SessionTrainingStatus() \
 * (subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"') \
 * subject.Subject * subject.SubjectLab * aging_tables.TrainingDay).fetch('subject_nickname', 'lab_name',
                                                                          'training_day', 'training_status',
                                                                          as_dict=True))
progression_old['stage'] = progression_old['training_status'].replace({'training in progress': 1,
                                                                       'trained': 2,
                                                                       'ready for ephys': 3,
                                                                       'over40days': 0,
                                                                       'wrong session type run': np.nan})
# plot these
fig = sns.FacetGrid(progression_old, col="lab_name", col_wrap=3, hue="subject_nickname")
fig.map(sns.lineplot, "training_day", "stage", estimator=None, units=progression_old.subject_nickname)
fig.savefig(os.path.join(figpath, "training_progression.pdf"))


progression_new = pd.DataFrame((new_criteria.SessionTrainingStatus() \
 * (subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"') \
 * subject.Subject * subject.SubjectLab * aging_tables.TrainingDay).fetch('subject_nickname', 'lab_name',
                                                                          'training_day', 'training_status',
                                                                          as_dict=True))
progression_new['stage'] = progression_new['training_status'].replace({'in_training': 1,
                                                                       'trained_1a': 2,
                                                                       'trained_1b': 3,
                                                                       'ready4ephysrig': 4,
                                                                       'ready4recording': 5,
                                                                       'untrainable': 0,
                                                                       'unbiasable': 0})
# plot these
fig = sns.FacetGrid(progression_new, col="lab_name", col_wrap=3, hue="subject_nickname")
fig.map(sns.lineplot, "training_day", "stage", estimator=None, units=progression_new.subject_nickname)
fig.savefig(os.path.join(figpath, "training_progression_new.pdf"))

# ================================================================== #
# FOR EACH ANIMAL, COMPUTE TIME TO REACH EACH TRAINING CRITERION
# ================================================================== #

training_time_old = progression_old.groupby(['lab_name', 'subject_nickname', 'training_status']).min().reset_index()
training_time_new = progression_new.groupby(['lab_name', 'subject_nickname', 'training_status']).min().reset_index()

plt.close('all')
sns.set_palette("cubehelix")
fig, ax = plt.subplots(1, 2, figsize=(13,8), sharey=True)

sns.swarmplot(x='stage', y='training_day', hue="lab_name", data=training_time_old, ax=ax[0])
sns.boxplot(x='stage', y='training_day', data=training_time_old, ax=ax[0], boxprops={'facecolor':'None'})
ax[0].set_title("Original criteria")
ax[0].set_xlim([0.5, 5.5])
ax[0].set_xticks([1,2,3])
ax[0].set_xticklabels(['in training', 'trained', 'ready4ephys'],
                  size='small', rotation=45)

sns.swarmplot(x='stage', y='training_day', hue="lab_name", data=training_time_new, ax=ax[1])
sns.boxplot(x='stage', y='training_day', data=training_time_new, ax=ax[1], boxprops={'facecolor':'None'})
ax[1].set_title("New criteria")
ax[1].set_xlim([0.5, 5.5])
ax[1].set_xticks([1,2,3,4,5])
ax[1].set_xticklabels(['in_training', 'trained_1a', 'trained_1b', 'ready4ephysrig', 'ready4recording'],
                  size='small', rotation=45)
ax[1].get_legend().remove()

fig.savefig(os.path.join(figpath, "training_times.pdf"))
fig.savefig(os.path.join(figpath, "training_times.png"), dpi=300)
plt.close('all')