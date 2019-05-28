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

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Arial')
sns.set(style="darkgrid", context="paper")
sns.set(style="darkgrid", context="paper", font_scale=1.3)

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

sess = ((acquisition.Session & 'task_protocol LIKE "%trainingchoice%"') * \
 (behavioral_analyses.SessionTrainingStatus() & 'training_status="trained"'))

# sess = (acquisition.Session * \
#  (behavioral_analyses.SessionTrainingStatus() & 'training_status="trained"'))

s = pd.DataFrame.from_dict(sess.fetch(as_dict=True))
labs = list(s['session_lab'].unique())
labs.append('zadorlab')
print(labs)

# hack to get around SQL limit
for lidx, lab in enumerate(labs):

	print(lab)
	b = (behavior.TrialSet.Trial & (subject.SubjectLab() & 'lab_name="%s"'%lab)) \
		* sess.proj('session_uuid','task_protocol') \
		* subject.SubjectLab.proj('lab_name') \
		* subject.Subject() & 'subject_birth_date between "2018-09-01" and "2019-02-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'

	bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
	print(bdat['subject_nickname'].unique())

	if lidx == 0:
		behav = bdat.copy()
	else:
		behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

# ================================= #
# for now, manually add the cortexlab matlab animals
# ================================= #

ucl_mice = ['KS001', 'MW003', 'MW001', 'MW002', 'LEW008', 'LEW009', 'LEW010']
ucl_trained_dates = ['2019-02-25', '2018-12-10', '2019-02-11', '2019-01-14', '2018-10-04', '2018-10-04', 'LEW010']

for midx, mouse in enumerate(ucl_mice):

	print(mouse)
	sess = (acquisition.Session.proj('session_start_time', 'task_protocol', 'session_uuid', session_date='DATE(session_start_time)') & \
		'session_date > "%s"'%ucl_trained_dates[midx]) * behavioral_analyses.SessionTrainingStatus()
	b = ((behavior.TrialSet.Trial & (subject.SubjectLab() & 'lab_name="cortexlab"')) \
		* subject.SubjectLab.proj('lab_name') \
		* subject.Subject() & 'subject_nickname="%s"'%mouse) \
		* sess

	bdat  = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
	behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

# ================================= #
# convert
# ================================= #

behav = dj2pandas(behav)
behav['lab_name'] = behav['lab_name'].str.replace('zadorlab', 'churchlandlab')

# ================================= #
# ONLY BLACK/neutral CURVES!
# ================================= #

sns.set_palette("gist_gray")  # palette for water types
fig = sns.FacetGrid(behav, 
	col="lab_name", col_order=['cortexlab', 'churchlandlab', 'mainenlab', 'angelakilab', 'danlab', 'wittenlab'], col_wrap=3,
	palette="gist_gray", sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")

#fig.map(plot_chronometric, "signed_contrast", "rt")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')

# FOR PARIS MEETING, ADD N IN TITLE
# titles = behav.groupby('lab_name').agg({'subject_nickname':'nunique', 'choice':
	# 'count'}).reset_index()      
# titles2 = ['%s | %d mice, %d trials'%(df['lab_name'], df['subject_nickname'], df['choice']) for i, df in titles.iterrows()]
# titles2 = ['%s lab'%(str.title(df['lab_name'][:-3])) for i, df in titles.iterrows()]
# # titles2 = [t.replace('Churchland lab', 'Churchland & Zador labs') for t in titles2]
# titles2 = [t.replace('Cortex lab', 'Carandini-Harris lab') for t in titles2]
titles2 = ['Carandini-Harris lab', 'Churchland & Zador labs', 'Mainen lab', 'Angelaki lab', 'Dan lab', 'Witten lab']

for ax, title in zip(fig.axes.flat, titles2):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_summary.pdf"))
fig.savefig(os.path.join(figpath, "psychfuncs_summary.png"), dpi=600)
plt.close('all')

# ================================= #
## ALSO ADD WITH individual subjects
# ================================= #

fig = sns.FacetGrid(behav, hue='subject_nickname',
	col="lab_name", col_order=sorted(behav.lab_name.unique()), col_wrap=3,
	palette="colorblind", sharex=True, sharey=True, aspect=.8)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_summary_permouse.pdf"))
fig.savefig(os.path.join(figpath, "psychfuncs_summary_permouse.png"), dpi=600)

# ONE PANEL PER MOUSE
fig = sns.FacetGrid(behav[behav.init_unbiased == True], 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_permouse_black.pdf"))
fig.savefig(os.path.join(figpath, "psychfuncs_permouse_black.png"), dpi=600)
plt.close('all')

# ALSO CHRONOMETRIC FUNCTIONS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0})
fig = sns.FacetGrid(behav[behav.init_unbiased == True], 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=True, sharey=True)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "chrono_permouse_black.pdf"))


# ================================= #
# ALSO CHRONOMETRIC FUNCTIONS
behav['abs_contrast'] = np.abs(behav.signed_contrast)
fig = sns.FacetGrid(behav[behav.init_unbiased == True], 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=True, sharey=True)
fig.map(plot_chronometric, "abs_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Contrast (%)', 'RT (s)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "chrono_abs_permouse_black.pdf"))


fig = sns.FacetGrid(behav, hue='subject_nickname',
	col="lab_name", col_order=sorted(behav.lab_name.unique()), col_wrap=2,
	palette="colorblind", sharex=True, sharey=True, aspect=.8)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "chrono_summary_permouse.pdf"))


# ABSOLUTE CHRONO, PER LAB
fig = sns.FacetGrid(behav, 
	col="lab_name", col_order=['cortexlab', 'churchlandlab', 'mainenlab', 'angelakilab', 'danlab', 'wittenlab'], col_wrap=3,
	palette="gist_gray", sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "abs_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Contrast (%)', 'RT (s)')
for ax, title in zip(fig.axes.flat, titles2):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "chrono_abs.pdf"))

