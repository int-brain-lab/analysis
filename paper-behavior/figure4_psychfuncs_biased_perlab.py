"""
PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS IN BIASED BLOCKS
Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
from figure_style import seaborn_style
import datajoint as dj
from IPython import embed as shell # for debugging
from scipy.special import erf # for psychometric functions

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '/Users/urai/Documents/code/analysis_IBL/python')
from dj_tools import *

## INITIALIZE A FEW THINGS
seaborn_style()
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# use_subjects = (subject.Subject() & 'subject_birth_date > "2018-09-01"' \
# 			   & 'subject_line IS NULL OR subject_line="C57BL/6J"') * subject.SubjectLab()
use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
criterion = behavioral_analyses.SessionTrainingStatus()
sess = ((acquisition.Session & 'task_protocol LIKE "%biasedChoiceWorld%"') \
		* (criterion & 'training_status="ready for ephys"')) * use_subjects

b = (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav = dj2pandas(bdat)

lab_names = {'carandinilab':'UCL', 'mainenlab':'CCU', 'churchlandlab':'CSHL',
			 'wittenlab':'Princeton', 'angelakilab':'NYU', 'mrsicflogellab':'SWC', 'danlab':'Berkeley'}

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# ================================= #

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
	sharex=True, sharey=True, aspect=1, hue="probabilityLeft")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4c_psychfuncs_biased_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_psychfuncs_biased_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="subject_nickname", col_wrap=8, hue="probabilityLeft",
					sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4c_psychfuncs_biased_permouse.pdf"))
plt.close('all')

# ================================= #
# CHRONOMETRIC FUNCTIONS
# ================================= #

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
	sharex=True, sharey=True, aspect=1, hue="probabilityLeft")
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4d_chronfuncs_biased_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure4d_chronfuncs_biased_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="subject_nickname", col_wrap=8, hue="probabilityLeft",
					sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4d_chronfuncs_biased_permouse.pdf"))
plt.close('all')