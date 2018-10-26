# Anne Urai, CSHL, 2018

# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import time, re, datetime, os, glob 
from datetime import timedelta
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from oneibl.one import ONE
from ibllib.time import isostr2date

## INITIALIZE A FEW THINGS
sns.set()
sns.set_context(context="paper")
one = ONE() # initialize

# get a list of all mice that are currently training
subjects 	= one._alyxClient.get('/subjects?alive=True')
names 		= [d['nickname'] for d in subjects]

# select only those belonging to ibl projects
projects 	= [d['projects'] for d in subjects]
projects 	= [item for sublist in projects for item in sublist]
indices 	= [i for i, s in enumerate(projects) if 'ibl' in s]
mice 		= [names[i] for i in indices]
print(mice)

def get_weights(mousename):
	wei = one._alyxClient.get('/weighings?nickname=%s' %mousename)
	wei = pd.DataFrame(wei)
	wei['date_time'] = pd.to_datetime(wei.date_time)
	wei.sort_values('date_time', inplace=True)
	wei.reset_index(drop=True, inplace=True)
	wei['date'] = wei['date_time'].dt.round('D')  

	wei['days'] = wei.date - wei.date[0]
	wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

	return wei

def get_water(mousename):
	wei = one._alyxClient.get('/water-administrations?nickname=%s' %mousename)
	wei = pd.DataFrame(wei)
	wei['date_time'] = pd.to_datetime(wei.date_time)

	# for w in wei:
	# wei['date_time'] = isostr2date(wei['date_time'])
	wei.sort_values('date_time', inplace=True)
	wei.reset_index(drop=True, inplace=True)
	wei['date'] = wei['date_time'].dt.round('D')  

	wei['days'] = wei.date - wei.date[0]
	wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

	wei = wei.set_index('date')
	wei.index = pd.to_datetime(wei.index)

	wa_unstacked = wei.pivot_table(index='date', 
		columns='water_type', values='water_administered', aggfunc='sum').reset_index()
	# wa_unstacked = wa_unstacked.set_index('date')
	# wa_unstacked.index = pd.to_datetime(wa_unstacked.index)

	wa_unstacked['date'] = pd.to_datetime(wa_unstacked.date)
	wa_unstacked.set_index('date', inplace=True)

	return wa_unstacked

def fix_date_axis(ax):
	# deal with date axis and make nice looking 
	ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
	for item in ax.get_xticklabels():
		item.set_rotation(60)

#def get_behavior(mousename):




# ## START BIG OVERVIEW PLOT
# # grab all mice that are currently trainig

for i, mouse in enumerate(mice):

	try:
		# MAKE THE FIGURE
		print(mouse)
		fig, ax = plt.subplots(4,3, figsize=(15,10))

		# ============================================= #
		# 1.  first subplot: weight and water curve
		# ============================================= #

		wei = get_weights(mouse)
		sns.lineplot(x="date_time", y="weight", color="black", data=wei, ax=ax[0,0])
		g = sns.scatterplot(x="date_time", y="weight", color="black", data=wei, ax=ax[0,0])
		ax[0,0].set(xlabel='', ylabel="Weight (g)", 
			xlim=[wei.date_time.min()-timedelta(days=1), wei.date_time.max()+timedelta(days=1)])
		fix_date_axis(ax[0,0])

		# ============================================= #
		# 2. water intake
		# ============================================= #

		# use pandas plot for a stacked bar
		wa = get_water(mouse)
		ax[1, 0].bar(wa.index, wa['Water'])
		if 'Hydrogel' in wa:
			ax[1, 0].bar(wa.index, wa['Hydrogel'], bottom=wa['Water'])
		ax[1, 0].set(ylabel="Water intake (mL)", xlabel='')
		fix_date_axis(ax[1,0])

		# 3. trial counts

		# 4. performance on easy trials


		fig.suptitle("Mouse %s, collected by %s" %(mouse, wei['user'].unique()))
		plt.tight_layout()
		fig.savefig('/Users/urai/Google Drive/Rig building WG/DataFigures/BehaviourData_Weekly/AlyxPlots/%s_overview.png' %mouse)
	except:
		pass

# behavioral data
# mice = one.search(subjects=mouse)

