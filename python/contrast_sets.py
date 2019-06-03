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
sns.set_style("darkgrid")
sns.set_context(context="paper")

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================================================== #
# COMPARE DIFFERENT CONTRAST SETS
# ================================================================== #

labs = ['churchlandlab', 'mainenlab']
# hack to get around SQL limit
for lidx, lab in enumerate(labs):

	print(lab)

	if 'churchlandlab' in lab:

		# old mice, latest sessions (to check different contrast sets)
		b = (behavior.TrialSet.Trial & (subject.SubjectLab() & 'lab_name="%s"'%lab)) \
			* (acquisition.Session() & 'session_start_time > "2019-03-01"'  & 'task_protocol LIKE "%biased%"') \
			* (subject.Subject() & 'subject_birth_date < "2018-08-01"') * subject.SubjectLab.proj('lab_name')
	elif 'mainenlab' in lab:
		b = (behavior.TrialSet.Trial & (subject.SubjectLab() & 'lab_name="%s"'%lab)) \
			* (acquisition.Session() & 'session_start_time > "2019-04-01"'  & 'task_protocol LIKE "%biased%"') \
			* (subject.Subject() & 'subject_nickname in ("ZM_1369", "ZM_1371", "ZM_1372")') * subject.SubjectLab.proj('lab_name')

	bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
	print(bdat['subject_nickname'].unique())

	if lidx == 0:
		behav = bdat.copy()
	else:
		behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

# recode
behav = dj2pandas(behav)

# CODE FOR WHETHER A LOW CONTRAST IS PRESENT OR NOT
behav['trial_stim_contrast_right'] = behav['trial_stim_contrast_right'] * 100
behav['trial_stim_contrast_right'] = behav.trial_stim_contrast_right.astype(int)
behav['trial_stim_contrast_right'] = behav['trial_stim_contrast_right'].replace(0, 100)

# indicate the contrast set for each session
behav['lower_contrasts'] = behav.groupby(['subject_nickname', 'session_start_time', 
	'lab_name'])['trial_stim_contrast_right'].transform(lambda x: np.any(x < 5))
behav.groupby(['subject_nickname', 'session_start_time'])['lower_contrasts'].unique()

# PLOT SEPARATELY PER CONTRAST SET
cmap = sns.diverging_palette(20, 220, n=len(behav['probabilityLeft'].unique()), center="dark")
fig = sns.FacetGrid(behav, hue="probabilityLeft", row="subject_nickname", col="lower_contrasts", palette=cmap)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "contrast_sets.pdf"))
print('done')


# DO STATS - IS THE BIAS SHIFT DEPENDENT ON THE INCLUSION OF LOWER CONTRASTS?
print('fitting psychometric per session...')
pars = behav.groupby(['lab_name', 'subject_nickname', 'lower_contrasts',
	'probabilityLeft']).apply(fit_psychfunc).reset_index()
# check if these fits worked as expected
print(pars.describe())
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

def compute_biasshift(x):

	# shift in the intercept parameter, along x-axis
	shift = x.loc[x['probabilityLeft'] == 80, 'bias'].item() - x.loc[x['probabilityLeft'] == 20, 'bias'].item()

	# also read out the y-shift, in choice probability
	xax = np.arange(-100, 100)
	y_80 = psy.erf_psycho_2gammas([x.loc[x['probabilityLeft'] == 80, 'bias'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'threshold'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'lapselow'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'lapsehigh'].item()], xax)
	y_20 = psy.erf_psycho_2gammas([x.loc[x['probabilityLeft'] == 20, 'bias'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'threshold'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'lapselow'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'lapsehigh'].item()], xax)
	print(y_20[xax == 0])
	print(y_80[xax == 0])
	
	yshift = 0.5 + (y_20[xax == 0] - y_80[xax == 0]) / 2 # from Nick Roy, 23 April

	return yshift

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname', 'lower_contrasts']).apply(compute_biasshift)
biasshift = biasshift.reset_index()
biasshift = biasshift.rename(columns={0: 'bias_shift'})

# make an overview plot of all bias shifts, one line for each subject
plt.close("all")	
fig, ax = plt.subplots()
sns.pointplot(x="lower_contrasts", y="bias_shift", hue='subject_nickname',
	markers='None', color="0.9", join=True, data=biasshift, legend=False)
ax = sns.pointplot(x="lower_contrasts", y="bias_shift", hue='lab_name', 
	dodge=True, join=True, data=biasshift, legend=False)
ax.set_ylabel('Bias shift (% choice)')
plt.setp(ax.lines, zorder=1000)
# improve the legend
handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[-2:], labels[-2:], frameon=True)
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "contrast_sets_stats.pdf"))



