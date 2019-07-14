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
from math import ceil

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Helvetica')
sns.set(style="darkgrid", context="paper", palette="colorblind")

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '/Users/urai/Documents/code/analysis_IBL/python')
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# only data when the animal is trained,
# but hasn't seen biased blocks yet
# ================================= #

# use_subjects = (subject.Subject() & 'subject_birth_date > "2018-09-01"' \
# 			   & 'subject_line IS NULL OR subject_line="C57BL/6J"') * subject.SubjectLab()
use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
criterion = behavioral_analyses.SessionTrainingStatus()
# sess = ((acquisition.Session & 'task_protocol LIKE "%trainingChoiceWorld%"') \
# 		* (criterion & 'training_status="trained"')) * use_subjects

# take all trials that include 0% contrast, instead of those where the animal is trained
sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%trainingChoiceWorld%"') \
	* use_subjects

b = (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav = dj2pandas(bdat)

# ================================= #
# choice history
# ================================= #

# code for history
behav['previous_choice'] 		= behav.choice.shift(1)
behav.loc[behav.previous_choice == 0, 'previous_choice'] = np.nan
behav['previous_outcome'] 		= behav.trial_feedback_type.shift(1)
behav.loc[behav.previous_outcome == 0, 'previous_outcome'] = np.nan
behav['previous_contrast'] 		= np.abs(behav.signed_contrast.shift(1))
behav['previous_choice_name'] 	= behav['previous_choice'].map({-1:'left', 1:'right'})
behav['previous_outcome_name']	= behav['previous_outcome'].map({-1:'post-error', 1:'post-correct'})

# ================================= #
# choice history
# ================================= #

# PREVIOUS CHOICE - SUMMARY PLOT
fig = sns.FacetGrid(behav, col="previous_outcome_name", hue="previous_choice_name", aspect=1, sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.set_titles("{col_name}")
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "history_allmice.pdf"))
fig.savefig(os.path.join(figpath, "history_allmice.png"), dpi=600)
print('done')

# ================================= #
# FIT PSYCHOMETRIC FUNCTIONS
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
# STRATEGY SPACE
# ================================= #

plt.close("all")
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
sns.scatterplot(x="history_postcorrect", y="history_posterror", hue="lab_name", data=biasshift, ax=ax, legend=False)

axlim = ceil(np.max([biasshift.history_postcorrect.max(), biasshift.history_posterror.max()]) * 10) / 10
plt.text(axlim/2, axlim/2, 'stay', horizontalalignment='center',verticalalignment='center')
plt.text(axlim/2, -axlim/2, 'win stay'+'\n'+'lose switch', horizontalalignment='center',verticalalignment='center')
plt.text(-axlim/2, -axlim/2, 'switch', horizontalalignment='center',verticalalignment='center')
plt.text(-axlim/2, axlim/2, 'win switch'+'\n'+'lose stay', horizontalalignment='center',verticalalignment='center')

ax.set_xlim([-axlim,axlim])
ax.set_ylim([-axlim,axlim])
ax.set_xticks([-axlim,0,axlim])
ax.set_yticks([-axlim,0,axlim])

ax.set_xlabel("History shift, after correct")
ax.set_ylabel("History shift, after error")
fig.tight_layout()
fig.savefig(os.path.join(figpath, "history_strategy.pdf"))
fig.savefig(os.path.join(figpath, "history_strategy.png"), dpi=600)
plt.close("all")

# ================================= #
# NOW ADD A SPLIT PER PREVIOUS CONTRAST
# ================================= #

print('fitting psychometric per previous contrast...')
pars = behav.groupby(['lab_name', 'subject_nickname', 'previous_choice', 'previous_outcome', 'previous_contrast'
					  ]).apply(fit_psychfunc).reset_index()
# keep only those fits with a minimum number of trials
pars = pars[pars.ntrials > 150]

def compute_biasshift(x):

	xax = np.arange(-100, 100)
	x_right = x.loc[np.isclose(x['previous_choice'], 1)]
	x_left  = x.loc[np.isclose(x['previous_choice'], -1)]

	if x_right.empty or x_left.empty:
		shift = np.nan
	else:
		y_right = psy.erf_psycho_2gammas([x_right['bias'].item(),
			x_right['threshold'].item(), x_right['lapselow'].item(), x_right['lapsehigh'].item()], xax)
		y_left  = psy.erf_psycho_2gammas([x_left['bias'].item(),
			x_left['threshold'].item(), x_left['lapselow'].item(), x_left['lapsehigh'].item()], xax)
		shift = (y_right[xax == 0] - y_left[xax == 0]).item()
	return shift

# compute a 'bias shift' per animal
biasshift_con = pars.groupby(['lab_name', 'subject_nickname', 'previous_contrast',
							  'previous_outcome']).apply(compute_biasshift).reset_index()
biasshift_con = biasshift_con.rename(columns={0: 'history_shift'})

# PLOT 
shell()

biasshift_con['previous_contrast'] = biasshift_con['previous_contrast'].replace(100, 35) # move 100 closer in
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
sns.scatterplot(x="previous_contrast", y="history_shift",marker='.', legend=False,
			 hue="previous_outcome", palette={1:"palegreen", -1:"lightcoral"}, 
			 data=biasshift_con, ax=ax)
sns.lineplot(x="previous_contrast", y="history_shift", err_style="bars", marker='o', legend=False,
			 hue="previous_outcome", palette={1:"forestgreen", -1:"firebrick"}, 
			 data=biasshift_con, ax=ax)
ax.set_xlabel("Previous contrast (%)")
ax.set_ylabel("History shift (%)")
ax.set(xticks=[0,6,12,25,35], xticklabels=['0', '6', '12', '25', '100'])
fig.tight_layout()
fig.savefig(os.path.join(figpath, "history_prevcontrast.pdf"))
fig.savefig(os.path.join(figpath, "history_prevcontrast.png"), dpi=600)
plt.close("all")

# ================================= #
# ADD A SPLIT BY PREVIOUS RT
# ================================= #

# 1st, is RT a good proxy for confidence here?
rts 	= behav.groupby(['lab_name', 'subject_nickname', 'trial_feedback_type', 'signed_contrast'])['rt'].median().reset_index()
rt_bins = behav.groupby(['lab_name', 'subject_nickname', 'signed_contrast'])['rt'].apply(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop')).reset_index()
behav['rt_bin'] = rt_bins.rt

fig, ax = plt.subplots(1, 3, figsize=(13,4))
sns.lineplot(data=rts, x='signed_contrast', y='rt', hue='trial_feedback_type', marker='o', ax=ax[0], ci=None, estimator=np.median, palette={-1:"firebrick", 1:"forestgreen"}, legend=False)
ax[0].set(title='Vevaiometric curve')
sns.regplot(data=behav, x='rt', y='correct', x_bins=10, color='k', marker='o', ax=ax[1], ci=None)
ax[1].set(title='Confidence calibration', ylim=[0.5, 1])
sns.lineplot(data=behav, x='signed_contrast', y='correct', hue='rt_bin', marker='o', ci=None, ax=ax[2], palette={0:"silver", 1:"dimgrey", 2:"black"}, legend=False)
ax[2].set(title='Conditional psychometric', ylim=[0.5, 1])

fig.tight_layout()
fig.savefig(os.path.join(figpath, "rt_confidence.pdf"))
plt.close("all")


