# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import time, re, datetime, os, glob
from datetime import timedelta
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from IPython import embed as shell

# IBL stuff
from oneibl.one import ONE
import psychofit as psy # https://github.com/cortex-lab/psychofit

# loading and plotting functions
from define_paths import fig_path
from os.path import join as join
from behavior_plots import *
from load_mouse_data import * # this has all plotting functions


## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0 } )
sns.set_context(context="paper")

## CONNECT TO ONE
one = ONE() # initialize

# get folder to save plots
path = fig_path()
if not os.path.exists(path):
    os.mkdir(path)

users = ['valeria', 'ines', 'miles']

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

for lidx, lab in enumerate(users):

	subjects 	= pd.DataFrame(one.alyx.get('/subjects?water_restricted=True&alive=True&responsible_user=%s'%lab))
	fig, axes = plt.subplots(ncols=5, nrows=4, constrained_layout=False,
        gridspec_kw=dict(width_ratios=[2,2,1,1,1], height_ratios=[1,1,1,1]), figsize=(11.69, 8.27))
	sns.set_palette("colorblind") # palette for water types
	axes = axes.flatten() # to enable 1d indexing

	for i, mouse in enumerate(subjects['nickname']):

		try:

			# ============================================= #
			# GENERAL METADATA
			# ============================================= #

			fig.suptitle('Mouse %s (%s), born %s, user %s (%s) \nstrain %s, cage %s, %s' %(subjects['nickname'][i],
			 subjects['sex'][i], subjects['birth_date'][i],
			 subjects['responsible_user'][i], subjects['lab'][i],
			 subjects['strain'][i], subjects['litter'][i], subjects['description'][i]))

			# ============================================= #
			# PERFORMANCE AND MEDIAN RT
			# ============================================= #

			# performance on easy trials
			ax = axes[2,0]
			behav['correct_easy'] = behav.correct
			behav.loc[np.abs(behav['signedContrast']) < 50, 'correct_easy'] = np.NaN
			correct_easy = behav.groupby(['date'])['correct_easy'].mean().reset_index()

			sns.lineplot(x="date", y="correct_easy", marker='o', color=".15", data=correct_easy, ax=ax)
			ax.set(xlabel='', ylabel="Performance (easy trials)",
				xlim=xlims, yticks=[0.5, 0.75, 1], ylim=[0.4, 1.01])
			# ax.yaxis.label.set_color("black")

			# RTs on right y-axis
			trialcounts = behav.groupby(['date'])['rt'].median().reset_index()
			righty = ax.twinx()
			sns.lineplot(x="date", y="rt", marker='o', color="firebrick", data=trialcounts, ax=righty)

			righty.yaxis.label.set_color("firebrick")
			righty.tick_params(axis='y', colors='firebrick')
			righty.set(xlabel='', ylabel="RT (s)", ylim=[0.1,10], xlim=xlims)
			righty.set_yscale("log")

			righty.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
			righty.grid(False)
			fix_date_axis(righty)
			fix_date_axis(ax)

			# ============================================= #
			# CONTRAST/CHOICE HEATMAP
			# ============================================= #

			ax = axes[3,0]
			import copy; cmap=copy.copy(plt.get_cmap('vlag'))
			cmap.set_bad(color="w") # remove those squares

	        # TODO: only take the mean when there is more than 1 trial (to remove bug in early sessions)
			pp  = behav.groupby(['signedContrast', 'days']).agg({'choice2':'mean'}).reset_index()
			pp2 = pp.pivot("signedContrast", "days",  "choice2").sort_values(by='signedContrast', ascending=False)
			pp2 = pp2.reindex([-100, -50, -25, -12, -6, 0, 6, 12, 25, 50, 100])

			# inset axes for colorbar, to the right of plot
			axins1 = inset_axes(ax, width="5%", height="90%", loc='right',
				bbox_to_anchor=(0.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0,)
			# now heatmap
			sns.heatmap(pp2, linewidths=.5, ax=ax, vmin=0, vmax=1, cmap=cmap, cbar=True,
				cbar_ax=axins1,
				cbar_kws={'label': 'Choose right (%)', 'shrink': 0.8, 'ticks': []})
			ax.set(ylabel="Contrast (%)")

			# fix the date axis
			dates  = behav.date.unique()
			xpos   = np.arange(len(dates)) + 0.5 # the tick locations for each day
			xticks = [i for i, dt in enumerate(dates) if pd.to_datetime(dt).weekday() is 0]
			ax.set_xticks(np.array(xticks) + 0.5)

			xticklabels = [pd.to_datetime(dt).strftime('%b-%d') for i, dt in enumerate(dates) if pd.to_datetime(dt).weekday() is 0]
			ax.set_xticklabels(xticklabels)
			for item in ax.get_xticklabels():
				item.set_rotation(60)
			ax.set(xlabel='')


	print("%s failed to run" %mouse)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.savefig(join(path + '%s_overview.pdf'%lab))


