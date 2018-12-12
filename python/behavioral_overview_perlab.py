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

# loading and plotting functions
from define_paths import fig_path
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

users = ['miles', 'ines', 'valeria']
users = ['mainenlab', 'cortexlab', 'zadorlab']

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

for lidx, lab in enumerate(users):

	subjects = pd.DataFrame(one.alyx.get('/subjects?water_restricted=True&alive=True&lab=%s'%lab))

	# group by batches: mice that were born on the same day
	batches = subjects.birth_date.unique()

	for birth_date in batches:

		mice = subjects.loc[subjects['birth_date'] == birth_date]['nickname']
		print(mice)
		fig, axes = plt.subplots(ncols=max([len(mice), 4]), nrows=4, constrained_layout=False, figsize=(11.69, 8.27))
		sns.set_palette("colorblind") # palette for water types

		for i, mouse in enumerate(mice):
			print(mouse)

			try:

				# WEIGHT CURVE AND WATER INTAKE
				t = time.time()
				axes[3,i].set_xlabel("Mouse " + mouse, fontweight="bold")

				weight_water, baseline = get_water_weight(mouse)

				# determine x limits
				xlims = [weight_water.date.min()-timedelta(days=2), weight_water.date.max()+timedelta(days=2)]
				plot_water_weight_curve(weight_water, baseline, axes[0,i])

				# TRIAL COUNTS AND SESSION DURATION
				behav 	= get_behavior(mouse)
				plot_trialcounts_sessionlength(behav, axes[1,i], xlims)
				fix_date_axis(axes[1,i])

				# PERFORMANCE AND MEDIAN RT
				plot_performance_rt(behav, axes[2,i], xlims)
				fix_date_axis(axes[2,i])

				# CONTRAST/CHOICE HEATMAP
				plot_contrast_heatmap(behav, axes[3,i])
				axes[3,i].set_xlabel("Mouse " + mouse, fontweight="bold")

				elapsed = time.time() - t
				print( "Elapsed time: %f seconds.\n" %elapsed )

			except:
				pass

		# SAVE PER BATCH, MAX 5
		fig.suptitle('Mice born on %s, user %s' %(birth_date, lab))
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig.savefig(os.path.join(path + '%s_overview_batch_%s.pdf'%(lab, birth_date)))
		plt.close(fig)

