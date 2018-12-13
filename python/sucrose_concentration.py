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
# from ibllib.time import isostr2date
import psychofit as psy # https://github.com/cortex-lab/psychofit

# loading and plotting functions
from define_paths import fig_path
from os.path import join as join
from load_mouse_data import * # this has all plotting functions

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0 } )
sns.set_context(context="paper")

## CONNECT TO ONE
one = ONE() # initialize
path = fig_path()

# get a list of all mice at cshl
subjects 	= pd.DataFrame(one.alyx.get('/subjects?alive=True&water_restricted=True&responsible_user=valeria'))
print(subjects['nickname'].unique())
fig, axes 	= plt.subplots(ncols=int(np.ceil(np.sqrt(len(subjects)))),
	nrows=int(np.ceil(np.sqrt(len(subjects)))),
	figsize=(11.69, 8.27), sharex=True, sharey=True)
axes = axes.flatten() # to enable 1d indexing

for i, mouse in enumerate(subjects['nickname']):
	
	print(mouse)
	weight_water, baseline 	= get_water_weight(mouse)

	# HACK TO RESTRICT TO TUES, WED, THU IN BOTH WEEKS
	behav_1stwk  = get_behavior(mouse, date_range=['2018-12-04', '2018-12-06'])
	behav_2ndwk  = get_behavior(mouse, date_range=['2018-12-11', '2018-12-13'])
	behav = pd.concat([behav_1stwk, behav_2ndwk])

	trialcounts 			= behav.groupby(['date'])['trial'].count().reset_index()

	# combine into a table that has trial counts, weights, water type 
	df = pd.merge(weight_water, trialcounts, on="date", how='outer')
	df.dropna(inplace=True)
	df = df[df['water_type'].str.contains("Water")] # subselect those days where some sucrose was given
	# assert(len(df['water_type'].unique()) > 2)
	df['concentration'] = df['water_type'].map({'Water': '0%', 'Water 10% Sucrose': '10%', 'Water 15% Sucrose': '15%'})

	# remove duplicate dates
	df.drop_duplicates(subset=['date', 'trial'], inplace=True)

	# show what's in here
	print(df.head(n=20))

	# plot their trial counts, errorbar on top of swarm
	sns.catplot(x="concentration", y="trial", kind="swarm", order=['0%', '10%', '15%'],
	            data=df, ax=axes[i], zorder=1);
	sns.pointplot(x="concentration", y="trial", color="k", order=['0%', '10%', '15%'],
	              data=df, ax=axes[i], join=False, zorder=100)
	axes[i].set(xlabel='', title=mouse)
	fig.savefig(join(path + 'sucrose_concentration.pdf'))

	# save into larger dataset
	if not 'all_data' in locals():
		all_data = df.groupby(['concentration'])['trial'].mean().reset_index()
	else:
		all_data = all_data.append(df.groupby(['concentration'])['trial'].mean().reset_index())

# ============================================= #
# ADD A GRAND AVERAGE PANEL
# ============================================= #

sns.catplot(x="concentration", y="trial", kind="swarm", order=['0%', '10%', '15%'],
		            data=all_data, ax=axes[i+1], zorder=1);
sns.pointplot(x="concentration", y="trial", color="k",  order=['0%', '10%', '15%'],
		              data=all_data, ax=axes[i+1], join=False, zorder=100)
axes[i+1].set(xlabel='', ylabel="Trial count", title='Group')

# save
plt.tight_layout()
fig.savefig(join(path + 'sucrose_concentration.pdf'))
plt.close(fig)

