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
sns.set(style="darkgrid", context="paper", font_scale=1.4)
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0})

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Documents/IBL/analysis/Figures/')

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
		* subject.Subject() & 'subject_birth_date between "2018-09-01" and "2019-02-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'
		# * subject.Subject() & 'subject_line IS NULL OR subject_line="C57BL/6J"'

	bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
	print(bdat['subject_nickname'].unique())

	if lidx == 0:
		behav = bdat.copy()
	else:
		behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

behav = dj2pandas(behav)
behav['lab_name'] = behav['lab_name'].str.replace('zadorlab', 'churchlandlab')

# ================================= #
# choice history
# ================================= #

# code for history
behav['previous_choice'] = behav.choice.shift(1)
behav.loc[behav.previous_choice == 0, 'previous_choice'] = np.nan
behav['previous_outcome'] = behav.trial_feedback_type.shift(1)
behav.loc[behav.previous_outcome == 0, 'previous_outcome'] = np.nan
behav['previous_contrast'] = np.abs(behav.signed_contrast.shift(1))

# fig = sns.FacetGrid(behav.loc[behav.probabilityLeft == 50, :], 
# 	row="lab_name", col="previous_choice", hue="previous_outcome")
# fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "previous_outcome.pdf"))
# print('done')
# plt.close("all")

## PREVIOUS CHOICE
behav['previous_choice_name'] = behav['previous_choice'].map({-1:'left', 1:'right'})
fig = sns.FacetGrid(behav.loc[behav.probabilityLeft == 50, :], col="lab_name", col_wrap=3, col_order=sorted(
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

# replicate Mani's finding - split by contrast of the previous trial
palette = sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=False)
fig = sns.FacetGrid(behav.loc[behav.probabilityLeft == 50, :], col_wrap=2,
	col="lab_name", hue="previous_contrast", palette=palette)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "previous_contrasts.pdf"))
plt.close("all")

# ================================= #
# EXAMPLE STIMULUS SEQUENCE
# ================================= #

# cmap = sns.diverging_palette(20, 220, n=len(behav['probabilityLeft'].unique()), center="dark")
# behav['probability_left_block'] = (behav.probabilityLeft - 50) * 2
# behav['stimulus_side'] = np.sign(behav.signed_contrast)
# behav.loc[behav.stimulus_side == 0, 'stimulus_side'] = np.nan

# fig = sns.FacetGrid(behav, 
# 	col="probabilityLeft", hue="probabilityLeft", col_wrap=3, col_order=[50, 20, 80],
# 	palette=cmap, sharex=True, sharey=True, aspect=0.6, height=2.2)
# fig.map(sns.distplot, "stimulus_side", kde=False, norm_hist=True, bins=2, hist_kws={'rwidth':1})
# fig.set_axis_labels(' ', 'Probability')
# fig.set(xticks=[-1,1], xticklabels=['L', 'R'], xlim=[-1.5,1.5], ylim=[0,1], yticks=[0,0.5, 1])
# for ax, title in zip(fig.axes.flat, ['P(Right) = 50%', 'P(Right) = 80%', 'P(Right) = 20%']):
#     ax.set_title(title)
# fig.savefig(os.path.join(figpath, "block_distribution.png"), dpi=600)
# fig.savefig(os.path.join(figpath, "block_distribution.pdf"))

# cmap = sns.diverging_palette(20, 220, n=len(behav['probabilityLeft'].unique()), center="dark")
# dat = behav[behav.subject_nickname.str.contains('ibl_witten_06') & 
# 	(behav.session_start_time > '2019-04-15') & (behav.session_start_time < '2019-04-19')]
# fig = sns.FacetGrid(dat, 
# 	col="session_start_time", col_wrap=1, 
# 	palette=cmap, sharex=True, sharey=True, aspect=3)
# fig.map(sns.lineplot, "trial_id", "probability_left_block", color='k')
# fig.map(sns.lineplot, "trial_id", "signed_contrast", hue=np.sign(dat.signed_contrast), palette=cmap, linewidth=0, marker='.', mec=None)
# for ax, title in zip(fig.axes.flat, titles2):
#     ax.set_title(' ')
# # fig.map(sns.lineplot, "trial_id", "probabilityLeft", marker='o')  
# fig.set(xlim=[0, 1000]) 
# fig.set_axis_labels('Trial number', 'Signed contrast (%)')
# fig.savefig(os.path.join(figpath, "session_course.png"), dpi=600)

# ================================= #
# biased blocks - plot curves
# ================================= #

cmap = sns.diverging_palette(220, 20, n=len(behav['probabilityLeft'].unique()), center="dark")
behav['prob_left_flip'] = 100 - behav.probabilityLeft
fig = sns.FacetGrid(behav[behav.init_unbiased == True], hue="prob_left_flip",
	col="lab_name", col_wrap=3, col_order=['cortexlab', 'churchlandlab', 'mainenlab', 'angelakilab', 'danlab', 'wittenlab'],
	palette=cmap, sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig._legend.set_title('P(Right) (%)')
titles2 = ['Carandini-Harris lab', 'Churchland & Zador labs', 'Mainen lab', 'Angelaki lab', 'Dan lab', 'Witten lab']

for ax, title in zip(fig.axes.flat, titles2):
    ax.set_title(title)
fig.savefig(os.path.join(figpath, "psychfuncs_biased_blocks_summary.pdf"))
fig.savefig(os.path.join(figpath, "biased_blocks_summary.png"), dpi=600)


fig = sns.FacetGrid(behav[behav.init_unbiased == True], hue="probabilityLeft",
	col="subject_nickname", col_wrap=6, 
	palette=cmap, sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig._legend.set_title('P(Right) (%)')
fig.set_titles("{col_name}")
fig.savefig(os.path.join(figpath, "psychfuncs_biased_blocks_permouse.pdf"))
fig.savefig(os.path.join(figpath, "biased_blocks_permouse.png"), dpi=600)

# LEAVE OUT THE 50/50 BLOCKS
fig = sns.FacetGrid(behav[behav.probabilityLeft != 50], hue="probabilityLeft",
	col="subject_nickname", col_wrap=6, 
	palette=sns.diverging_palette(20, 220, n=2, center="dark"), sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig._legend.set_title('P(Right) (%)')
fig.set_titles("{col_name}")
fig.savefig(os.path.join(figpath, "psychfuncs_biased_blocks_permouse_no5050.pdf"))

# ================================ #

# ALSO CHRONOMETRIC FUNCTION
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0})
fig = sns.FacetGrid(behav[behav.probabilityLeft != 50], hue="probabilityLeft",
	col="subject_nickname", col_wrap=6, 
	palette=sns.diverging_palette(20, 220, n=2, center="dark"), sharex=True, sharey=True)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.despine(trim=True)
fig._legend.set_title('P(Right) (%)')
fig.set_titles("{col_name}")
fig.savefig(os.path.join(figpath, "chrono_biased_blocks_permouse_no5050.pdf"))
plt.close("all")

fig = sns.FacetGrid(behav, hue="probabilityLeft",
	col="subject_nickname", col_wrap=6, 
	palette=cmap, sharex=True, sharey=True)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.despine(trim=True)
fig._legend.set_title('P(Right) (%)')
fig.set_titles("{col_name}")
fig.savefig(os.path.join(figpath, "chrono_biased_blocks_permouse.pdf"))
plt.close("all")

fig = sns.FacetGrid(behav, hue="probabilityLeft",
	col="lab_name", col_wrap=3, col_order=['cortexlab', 'churchlandlab', 'mainenlab', 'angelakilab', 'danlab', 'wittenlab'],
	palette=cmap, sharex=True, sharey=True)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
fig.despine(trim=True)
for ax, title in zip(fig.axes.flat, titles2):
    ax.set_title(title)
fig._legend.set_title('P(Right) (%)')
fig.savefig(os.path.join(figpath, "chrono_biased_blocks.pdf"))
plt.close("all")


# # SEPARATELY FOR INIT_UNBIASED
# fig = sns.FacetGrid(behav, hue="probabilityLeft", row="lab_name", col="init_unbiased", palette=cmap)
# fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "biased_blocks.pdf"))

# fig = sns.FacetGrid(behav, hue="probabilityLeft", col="subject_nickname", col_wrap=5, palette=cmap)
# fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
# fig.despine(trim=True)
# fig.savefig(os.path.join(figpath, "biased_blocks_permouse.pdf"))
# plt.close("all")

# ================================= #
# biased blocks - plot curves
# ================================= #

#shell()
print('fitting psychometric per session...')

pars = behav.groupby(['lab_name', 'subject_nickname', 
	'probabilityLeft']).apply(fit_psychfunc).reset_index()
# check if these fits worked as expected
print(pars.describe())
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

def compute_biasshift(x):
	# print(x.describe())
	shift = x.loc[x['probabilityLeft'] == 80, 'bias'].item() - x.loc[x['probabilityLeft'] == 20, 'bias'].item()

	xax = np.arange(-100, 100)
	y_80 = psy.erf_psycho_2gammas([x.loc[x['probabilityLeft'] == 80, 'bias'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'threshold'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'lapselow'].item(), 
		x.loc[x['probabilityLeft'] == 80, 'lapsehigh'].item()], xax)
	y_20 = psy.erf_psycho_2gammas([x.loc[x['probabilityLeft'] == 20, 'bias'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'threshold'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'lapselow'].item(), 
		x.loc[x['probabilityLeft'] == 20, 'lapsehigh'].item()], xax)

	yshift = 100 * (y_20[xax == 0] - y_80[xax == 0])
	yshift = 0.5 + (y_20[xax == 0] - y_80[xax == 0]) / 2 # from Nick Roy, 23 April

	return yshift

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift)
biasshift = biasshift.reset_index()
biasshift = biasshift.rename(columns={0: 'bias_shift'})

print(biasshift.describe())

biasshift = biasshift[biasshift['subject_nickname'].str.contains("DY_002")==False]

# make an overview plot of all bias shifts, one line for each subject
plt.close("all")
fig, ax = plt.subplots(figsize=(2,3))
# sns.swarmplot(x="lab_name", y="bias_shift", hue="lab_name", data=biasshift, ax=ax, zorder=0)
sns.pointplot(x=biasshift.lab_name, y=100 * biasshift.bias_shift, legend=False, join=False, ax=ax, color='k')
ax.set_ylim([50, 75])
ax.set_xlabel('')
ax.set_ylabel('P(match) (%)')
ax.set_xticklabels(titles2, rotation=90)
fig.tight_layout()
fig.savefig(os.path.join(figpath, "pmatch_stats.pdf"))
fig.savefig(os.path.join(figpath, "pmatch_stats.png"), dpi=600)
print(biasshift['bias_shift'].mean())


# ================================= #
# ALSO GET PARAMETERS from the psychometric function fit
# ================================= #

print('fitting psychometric per session...')
# triple check, does this return consistent parameter estimates??
# TODO: GET THESE FROM PSYCHRESULTS??

pars = behav.groupby(['lab_name', 'subject_nickname', 'init_unbiased',
	'probabilityLeft']).apply(fit_psychfunc).reset_index()
# check if these fits worked as expected
print(pars.describe())
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname', 'init_unbiased']).apply(compute_biasshift)
biasshift = biasshift.reset_index()
biasshift = biasshift.rename(columns={0: 'bias_shift'})

# make an overview plot of all bias shifts, one line for each subject
plt.close("all")
fig, ax = plt.subplots()
sns.pointplot(x="init_unbiased", y="bias_shift", hue='subject_nickname', 
	markers='None', color="0.9", join=True, data=biasshift, legend=False)
ax = sns.pointplot(x="init_unbiased", y="bias_shift", hue='lab_name', dodge=True,
 	join=True, data=biasshift, legend=False)
ax.set_ylabel('Bias shift (%% contrast)')
plt.setp(ax.lines, zorder=1000)
# improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-2:], labels[-2:], frameon=True)
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "biased_blocks_stats.pdf"))

# ================================= #
# re-express in % response
# ================================= #

zerotrials = behav.loc[behav.signed_contrast == 0, :]
choicebias = zerotrials.groupby(['lab_name', 'subject_nickname', 
	'probabilityLeft'])['choice_right'].mean().reset_index()

# make an overview plot of all bias shifts, one line for each subject
plt.close("all")
fig, ax = plt.subplots()
sns.pointplot(x="probabilityLeft", y="choice_right", hue='subject_nickname', 
	markers='None', color="0.9", join=True, data=choicebias, legend=False)
ax = sns.pointplot(x="probabilityLeft", y="choice_right", dodge=True,
 	join=True, data=choicebias, legend=False)
ax.set_ylabel('Bias shift (P(choice == "right") at 0% contrast)')
# improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-1:], labels[-1:], frameon=True)
sns.despine(trim=True)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
fig.savefig(os.path.join(figpath, "biased_blocks_response_stats.pdf"))

# ================================================================== #
# SEPARATELY FOR LONG AND SHORT BIAS BLOCKS
# ================================================================== #

behav['trialsinblock'] = behav.groupby(['lab_name', 'subject_nickname', 
	'session_start_time', 'probabilityLeft']).cumcount() + 1
# divide into early and late trials
#behav['trials_early'] = behav['trialsinblock'] <= behav['trialsinblock'].median()
behav['trials_early'] = behav.groupby(['subject_nickname', 'lab_name', 'probabilityLeft'])['trialsinblock'].transform(lambda x: x < 50)

# plot
fig = sns.FacetGrid(behav, hue="probabilityLeft", row="trials_early", col="init_unbiased", palette=cmap)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "biased_blocks_shortlong.pdf"))

# recompute fits
pars = behav.groupby(['lab_name', 'subject_nickname', 'init_unbiased',
	'probabilityLeft', 'trials_early']).apply(fit_psychfunc).reset_index()

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname', 'init_unbiased', 'trials_early']).apply(compute_biasshift)
biasshift = biasshift.reset_index()
biasshift = biasshift.rename(columns={0: 'bias_shift'})

# plot summary of those, separately for early and late trials
plt.close("all")
fig = sns.catplot(x="init_unbiased", y="bias_shift", col="lab_name", hue='trials_early', 
	kind='point', join=True, dodge=True, data=biasshift)
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "biased_blocks_stats_shortlong.pdf"))

