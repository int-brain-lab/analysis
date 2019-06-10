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
import training_criteria_schemas as criteria_urai 

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

criterion = criteria_urai.SessionTrainingStatus_v0()
sess = ((acquisition.Session) * \
 (criterion & 'training_status="trained"')) \
 * subject.SubjectLab * subject.Subject

s = pd.DataFrame.from_dict(sess.fetch(as_dict=True))
labs = list(s['lab_name'].unique())
print(labs)
# labs = list(filter(None, labs)) # remove None lab

# hack to get around SQL limit
for lidx, lab in enumerate(labs):

	print(lab)
	subjects = s[s['lab_name'].str.contains(lab)].reset_index()

	for midx, mousename in enumerate(subjects['subject_nickname'].unique()):

        # ============================================= #
        # check whether the subject is trained based the the lastest session
        # ============================================= #

		subj = subject.Subject & 'subject_nickname="{}"'.format(mousename)
		last_session = subj.aggr(
		behavior.TrialSet, session_start_time='max(session_start_time)')
		training_status = \
		(criterion & last_session).fetch1(
		'training_status')

		if training_status in ['trained', 'ready for ephys']:
			first_trained_session = subj.aggr(
			criterion &
			'training_status="trained"',
			first_trained='min(session_start_time)')
			first_trained_session_time = first_trained_session.fetch1(
			'first_trained')
			# convert to timestamp
			trained_date = pd.DatetimeIndex([first_trained_session_time])[0]
		else:
			print('WARNING: THIS MOUSE WAS NOT TRAINED!')
			continue

		# now get the sessions that went into this
		# https://github.com/shenshan/IBL-pipeline/blob/master/ibl_pipeline/analyses/behavior.py#L390
		sessions = (behavior.TrialSet & subj &
			(acquisition.Session) &
			'session_start_time <= "{}"'.format(
			trained_date.strftime(
			'%Y-%m-%d %H:%M:%S')
			)).fetch('KEY')

		# if not more than 3 biased sessions, keep status trained
		sessions_rel = sessions[-3:]

		b = (behavior.TrialSet.Trial & sessions_rel) \
			* (subject.SubjectLab & 'lab_name="%s"'%lab) \
			* (subject.Subject & 'subject_nickname="%s"'%mousename)

		bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
		print(bdat['subject_nickname'].unique())
		print(trained_date)
		print(bdat['session_start_time'].unique())

		# APPEND
		if not 'behav' in locals():
			behav = bdat.copy()
		else:
			behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

# ================================================================== #
# for now, manually add the cortexlab matlab animals
# ================================================================== #

# ucl_mice = ['KS001', 'MW003', 'MW001', 'MW002', 'LEW008', 'LEW009', 'LEW010']
# ucl_trained_dates = ['2019-02-25', '2018-12-10', '2019-02-11', '2019-01-14', '2018-10-04', '2018-10-04', 'LEW010']

# for midx, mouse in enumerate(ucl_mice):

# 	print(mouse)
# 	sess = (acquisition.Session.proj('session_start_time', 'task_protocol', 'session_uuid', session_date='DATE(session_start_time)') & \
# 		'session_date > "%s"'%ucl_trained_dates[midx]) * behavioral_analyses.SessionTrainingStatus()
# 	b = ((behavior.TrialSet.Trial & (subject.SubjectLab() & 'lab_name="cortexlab"')) \
# 		* subject.SubjectLab.proj('lab_name') \
# 		* subject.Subject() & 'subject_nickname="%s"'%mouse) \
# 		* sess

# 	bdat  = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
# 	behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

# ================================= #
# convert
# ================================= #

behav = dj2pandas(behav)
behav['lab_name'] = behav['lab_name'].str.replace('zadorlab', 'churchlandlab')
print(behav.describe())

# ================================= #
# ONE PANEL PER MOUSE
# ================================= #

fig = sns.FacetGrid(behav, 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname", color='k').add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_permouse_black_v0.pdf"))

shell()

fig.savefig(os.path.join(figpath, "psychfuncs_permouse_black.png"), dpi=600)
plt.close('all')

# ALSO CHRONOMETRIC FUNCTIONS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0})
fig = sns.FacetGrid(behav, 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=True, sharey=True)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "chrono_permouse_black.pdf"))
shell()

# ================================= #
# RT ACROSS ALL CONTRASTS, PER LAB
# ================================= #

median_rt = behav.groupby(['subject_nickname', 'lab_name'])['rt'].median().reset_index()
fig = sns.swarmplot(x="lab_name", y="rt", data=median_rt)
fig.set_title('RT median')
plt.savefig(os.path.join(figpath, "rt_median.pdf"))
plt.close('all')

mean_rt = behav.groupby(['subject_nickname', 'lab_name'])['rt'].mean().reset_index()
fig = sns.swarmplot(x="lab_name", y="rt", data=mean_rt)
fig.set_title('RT mean')
plt.savefig(os.path.join(figpath, "rt_mean.pdf"))
plt.close('all')

# RT DISTRIBUTIONS - ONE PANEL PER MOUSE
fig = sns.FacetGrid(behav[behav.init_unbiased == True], 
	col="subject_nickname", col_wrap=7, 
	palette="gist_gray", sharex=False, sharey=False, xlim=[-1, 5])
fig.map(sns.distplot, "rt", bins=500).add_legend()
# fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
#fig.set_xlim([-0.1, 3])
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "rt_dist_permouse.pdf"))
plt.close('all')

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

# ================================= #
# ALSO CHRONOMETRIC FUNCTIONS
# ================================= #

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
