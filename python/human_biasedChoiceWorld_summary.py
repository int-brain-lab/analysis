"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1.4)

import datajoint as dj
from dj_tools import *

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses
sfn_subject = dj.create_virtual_module('subject', 'group_shared_sfndata')
figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# SUMMARY OF DEMOGRAPHICS
# ================================= #

subj_query = (sfn_subject.Subject & 'subject_nickname LIKE "human%"') \
             * (acquisition.Session.proj(session_date='date(session_start_time)'))
subj_data = (subj_query * behavior.TrialSet).fetch(format='frame').reset_index()
subj_data.sex.fillna(value='U', inplace=True)

# compute some simple things
subj_data['performance'] = subj_data.n_correct_trials / subj_data.n_trials
subj_data['trial_duration'] = subj_data.trials_end_time / subj_data.n_trials
subj_data.fillna(value=pd.np.nan, inplace=True)

g = sns.pairplot(subj_data, vars=['age', 'task_knowledge', 'aq_score', 'performance', 'n_trials'],
                 hue='sex', diag_kind='hist', diag_kws={'histtype':'step'})
g.savefig(os.path.join(figpath, "human_sfn_overview.png"), dpi=300)
plt.close('all')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

subj_query = (sfn_subject.Subject & 'subject_nickname LIKE "human%"') \
             * (acquisition.Session.proj(session_date='date(session_start_time)'))
behav = (subj_query * behavior.TrialSet.Trial).fetch(format='frame').reset_index()
assert(not behav.empty)
behav = dj2pandas(behav)

print(behav.describe())
print(behav.subject_nickname.unique())

# ================================= #
# PLOT
# ================================= #

#PSYCHOMETRIC
fig = sns.FacetGrid(behav, hue="task_knowledge", hue_order=[1,2,3,4],
                    palette=sns.color_palette("cubehelix", 4))
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "human_sfn_psychfuncs_taskknowledge.png"), dpi=300)

# CHRONOMETRIC
fig = sns.FacetGrid(behav,
                    hue="task_knowledge", hue_order=[1,2,3,4],
                    palette=sns.color_palette("cubehelix", 4))
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Trial duration (s)')
fig.savefig(os.path.join(figpath, "human_sfn_chronfuncs_taskknowledge.png"), dpi=300)


# BY BIASED BLOCKS
fig = sns.FacetGrid(behav,
                    hue="probabilityLeft",
                    palette=sns.diverging_palette(20, 220, n=2, center="dark"))
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "human_sfn_psychfuncs_biased.png"), dpi=300)

# BY BIASED BLOCKS
fig = sns.FacetGrid(behav,
                    hue="probabilityLeft",
                    palette=sns.diverging_palette(20, 220, n=2, center="dark"))
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.savefig(os.path.join(figpath, "human_sfn_chronfuncs_biased.png"), dpi=300)

# ================================= #
# LEARNING CURVES
# ================================= #

fig = sns.FacetGrid(behav, hue="task_knowledge", hue_order=[1,2,3,4],
                    palette=sns.color_palette("cubehelix", 4))
fig.map(sns.lineplot, "trial_id", "correct")
fig.savefig(os.path.join(figpath, "human_sfn_learning_taskknowledge.pdf"))

# plot one curve for each person
fig = sns.FacetGrid(behav,
                    col="subject_nickname", col_wrap=12, sharex=True, sharey=True,
                    hue="task_knowledge")
fig.map(plot_psychometric, "signed_contrast", "choice_right",
        "subject_nickname")
fig.set_titles("{col_name}")
fig.set_axis_labels('Signed contrast (%)', 'Trial duration (s)')
fig.savefig(os.path.join(figpath, "human_sfn_psychfuncs_allsj.pdf"))
