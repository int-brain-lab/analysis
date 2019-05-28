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
sns.set(style="darkgrid", context="paper", font='Helvetica')
sns.set(style="darkgrid", context="paper", palette="colorblind")

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

sess = (acquisition.Session & 'task_protocol LIKE "%biased%"')
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
		* subject.Subject() & 'subject_line IS NULL OR subject_line="C57BL/6J"'
		#* subject.Subject() & 'subject_birth_date between "2018-09-01" and "2019-02-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'

	bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
	print(bdat['subject_nickname'].unique())

	if lidx == 0:
		behav = bdat.copy()
	else:
		behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

behav = dj2pandas(behav)

# ================================= #
# choice history
# ================================= #

# code for history
behav['previous_choice'] = behav.choice.shift(1)
behav.loc[behav.previous_choice == 0, 'previous_choice'] = np.nan
behav['previous_outcome'] = behav.trial_feedback_type.shift(1)
behav.loc[behav.previous_outcome == 0, 'previous_outcome'] = np.nan
behav['previous_contrast'] = np.abs(behav.signed_contrast.shift(1))
behav['previous_choice_name'] = behav['previous_choice'].map({-1:'left', 1:'right'})

#  ONLY TAKE UNBIASED BLOCKS
behav = behav.loc[behav.probabilityLeft == 50, :]

# ================================= #
# choice history
# ================================= #

fig = sns.FacetGrid(behav,
	row="lab_name", col="previous_choice", hue="previous_outcome")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "previous_outcome.pdf"))
print('done')
plt.close("all")

## PREVIOUS CHOICE
fig = sns.FacetGrid(behav, col="lab_name", col_wrap=3, col_order=sorted(
	behav.lab_name.unique()),
	hue="previous_choice_name")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
titles = behav.groupby('lab_name').agg({'subject_nickname':'nunique', 'choice':
	'count'}).reset_index()      

# nicer lab name titles
titles2 = ['%s lab'%(str.capitalize(df['lab_name'][:-3])) for i, df in titles.iterrows()]
titles2 = [t.replace('Churchland lab', 'Churchland & Zador labs') for t in titles2]
titles2 = [t.replace('Cortex lab', 'Carandini-Harris lab') for t in titles2]
for ax, title in zip(fig.axes.flat, titles2):
    ax.set_title(title)

fig.despine(trim=True)
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "previous_choice.pdf"))
fig.savefig(os.path.join(figpath, "previous_choice.png"), dpi=600)
print('done')
plt.close("all")

## PREVIOUS CHOICE - SUMMARY PLOT
fig = sns.FacetGrid(behav, hue="previous_choice_name")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "previous_choice_onepanel.pdf"))
fig.savefig(os.path.join(figpath, "previous_choice_onepanel.png"), dpi=600)
print('done')

# ================================= #
# biased blocks - plot curves
# ================================= #

print('fitting psychometric per session...')
pars = behav.groupby(['lab_name', 'subject_nickname', 'previous_choice', 'previous_outcome']).apply(fit_psychfunc).reset_index()

# check if these fits worked as expected
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

def compute_biasshift_posterror(x):

	xax = np.arange(-100, 100)
	x_right = x.loc[np.isclose(x['previous_choice'], 1) & np.isclose(x['previous_outcome'], -1)]
	x_left = x.loc[np.isclose(x['previous_choice'], -1) & np.isclose(x['previous_outcome'], -1)]

	y_right = psy.erf_psycho_2gammas([x_right['bias'].item(), 
		x_right['threshold'].item(), x_right['lapselow'].item(), x_right['lapsehigh'].item()], xax)
	y_left = psy.erf_psycho_2gammas([x_left['bias'].item(), 
		x_left['threshold'].item(), x_left['lapselow'].item(), x_left['lapsehigh'].item()], xax)

	shift_posterror = (y_right[xax == 0] - y_left[xax == 0])
	return shift_posterror

def compute_biasshift_postcorrect(x):

	xax = np.arange(-100, 100)
	x_right = x.loc[np.isclose(x['previous_choice'], 1) & np.isclose(x['previous_outcome'], 1)]
	x_left = x.loc[np.isclose(x['previous_choice'], -1) & np.isclose(x['previous_outcome'], 1)]

	y_right = psy.erf_psycho_2gammas([x_right['bias'].item(), 
		x_right['threshold'].item(), x_right['lapselow'].item(), x_right['lapsehigh'].item()], xax)
	y_left = psy.erf_psycho_2gammas([x_left['bias'].item(), 
		x_left['threshold'].item(), x_left['lapselow'].item(), x_left['lapsehigh'].item()], xax)

	shift_postcorrect = (y_right[xax == 0] - y_left[xax == 0])
	return shift_postcorrect

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_postcorrect).reset_index()
biasshift = biasshift.rename(columns={0: 'history_postcorrect'})

biasshift2 = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_posterror).reset_index()
biasshift2 = biasshift2.rename(columns={0: 'history_posterror'})
biasshift['history_posterror'] = biasshift2.history_posterror.copy()

# ================================= #
# PLOT AN OVERVIEW SCHEMATIC
# ================================= #

plt.close("all")
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
sns.scatterplot(x="history_postcorrect", y="history_posterror", hue="lab_name", data=biasshift, legend=False, ax=ax)
plt.text(0.5, 0.5, 'stay', horizontalalignment='center',verticalalignment='center')
plt.text(0.5, -0.5, 'win stay'+'\n'+'lose switch', horizontalalignment='center',verticalalignment='center')
plt.text(-0.5, -0.5, 'switch', horizontalalignment='center',verticalalignment='center')
plt.text(-0.5, 0.5, 'win switch'+'\n'+'lose stay', horizontalalignment='center',verticalalignment='center')

# ax.fig.text(0, 1,'Left the plot', fontsize=20, rotation=90)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])

ax.set_xlabel("History shift, after correct")
ax.set_ylabel("History shift, after error")
fig.tight_layout()
fig.savefig(os.path.join(figpath, "history_strategy.pdf"))
fig.savefig(os.path.join(figpath, "history_strategy.png"), dpi=600)

