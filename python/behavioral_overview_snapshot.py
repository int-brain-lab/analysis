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

from oneibl.one import ONE
from ibllib.time import isostr2date
from psychofit import psychofit as psy # https://github.com/cortex-lab/psychofit

# loading and plotting functions
from behavior_plots import *
from load_mouse_data import * # this has all plotting functions

def fit_psychfunc(df):

	# reshape data
	choicedat = df.groupby('signedContrast').agg({'trial':'max', 'choice2':'mean'}).reset_index()
	# print(choicedat)

	# if size(np.abs(df['signedContrast'])) > 6:
	pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas', parstart=np.array([choicedat['signedContrast'].mean(), 20., 0.05, 0.05]), parmin=np.array([choicedat['signedContrast'].min(), 0., 0., 0.]), parmax=np.array([choicedat['signedContrast'].max(), 100., 1, 1]))

	df2 = {'bias':pars[0],'threshold':pars[1], 'lapselow':pars[2], 'lapsehigh':pars[3]}
	return pd.DataFrame(df2, index=[0])

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True} )
sns.set_context(context="paper")
current_palette = sns.color_palette()

# set a new palette for biased blocks: black, purple, orange
one = ONE() # initialize

# get a list of all mice that are currently training
subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?water_restricted=True'))
# subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?nickname=IBL_1'))
print(subjects['nickname'].unique())

for i, mouse in enumerate(subjects['nickname']):

	try:

		# MAKE THE FIGURE, divide subplots using gridspec
		print(mouse)
		wei 	= get_weights(mouse)
		behav 	= get_behavior(mouse)

		fig, axes = plt.subplots(ncols=5, nrows=4, constrained_layout=False,
	        gridspec_kw=dict(width_ratios=[2,2,1,1,1], height_ratios=[1,1,1,1]), figsize=(11.69, 8.27))

		# ============================================= #
		# GENERAL METADATA
		# ============================================= #

		# write mouse info
		# TODO: MOVE THIS INTO SUPTITLE
		# ax = axes[0,0]
		# ax.annotate('Mouse %s (%s), DoB %s \n user %s, %s\n strain %s, cage %s \n %s' %(subjects['nickname'][i],
		#  subjects['sex'][i], subjects['birth_date'][i], 
		#  subjects['responsible_user'][i], subjects['lab'][i],
		#  subjects['strain'][i], subjects['litter'][i], subjects['description'][i]),
		#  xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center')
		# ax.axis('off')

		# ============================================= #
		# WEIGHT CURVE AND WATER INTAKE
		# ============================================= #

		# weight on top
		ax = axes[0,0]
		sns.lineplot(x="date_time", y="weight", color="black", markers=True, data=wei, ax=ax)
		sns.scatterplot(x="date_time", y="weight", color="black", data=wei, ax=ax)
		ax.set(xlabel='', ylabel="Weight (g)", 
			xlim=[wei.date_time.min()-timedelta(days=1), wei.date_time.max()+timedelta(days=1)])
		fix_date_axis(ax)

		# use pandas plot for a stacked bar - water types
		ax = axes[1,0]
		sns.set_palette("colorblind") # palette for water
		try:
			wa_unstacked, wa 	= get_water(mouse)
			wa_unstacked.loc[:,['Water','Hydrogel']].plot.bar(stacked=True, ax=ax)
			l = ax.legend()
			l.set_title('')
			ax.set(ylabel="Water intake (mL)", xlabel='')

			# fix dates, known to be an issue in pandas/matplotlib
			ax.set_xticklabels([dt.strftime('%b-%d') if dt.weekday() is 1 else "" for dt in wa_unstacked.index.to_pydatetime()])
			for item in ax.get_xticklabels():
				item.set_rotation(60)
		except:
			pass

		# ============================================= #
		# TRIAL COUNTS AND PERFORMANCE
		# ============================================= #

		# performance on easy trials
		ax = axes[2,0]
		behav['correct_easy'] = behav.correct
		behav.loc[np.abs(behav['signedContrast']) < 50, 'correct_easy'] = np.NaN
		correct_easy = behav.groupby(['start_time'])['correct_easy'].mean().reset_index()
		
		sns.lineplot(x="start_time", y="correct_easy", markers=True, color="black", data=correct_easy, ax=ax)
		sns.scatterplot(x="start_time", y="correct_easy", color="black", data=correct_easy, ax=ax)
		ax.set(xlabel='', ylabel="Performance on easy trials", 
			xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=2)],
			yticks=[0.5, 0.75, 1], ylim=[0.4, 1.01])
		ax.yaxis.label.set_color("black")

		# overlay trial counts
		trialcounts = behav.groupby(['start_time'])['trial'].max().reset_index()
		righty = ax.twinx()
		sns.lineplot(x="start_time", y="trial", markers=True, color="firebrick", data=trialcounts, ax=righty)
		sns.scatterplot(x="start_time", y="trial", color="firebrick", data=trialcounts, ax=righty)
		righty.yaxis.label.set_color("firebrick")
		righty.grid(False)
		fix_date_axis(righty)
		fix_date_axis(ax)
		righty.set(xlabel='', ylabel="Trial count", 
			xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=2)])

		# ============================================= #
		# CONTRAST/CHOICE HEATMAP
		# ============================================= #

		ax = axes[3,0]
		plot_perf_heatmap(behav, ax=ax)
		ax.set_xticklabels([dt.strftime('%b-%d') if dt.weekday() is 1 else "" for dt in wa_unstacked.index.to_pydatetime()])
		for item in ax.get_xticklabels():
			item.set_rotation(60)

		# ============================================= #
		# PSYCHOMETRIC FUNCTION FITS OVER TIME
		# ============================================= #

		# fit psychfunc on choice fraction, rather than identity
		pars = behav.groupby(['start_time', 'probabilityLeft']).apply(fit_psychfunc).reset_index()
		parsdict = {'threshold': r'Threshold $(\sigma)$', 'bias': r'Bias $(\mu)$', 
			'lapselow': r'Lapse low $(\gamma)$', 'lapsehigh': r'Lapse high $(\lambda)$'}
		ylims = [[-5, 105], [-105, 105], [-0.05, 1.05], [-0.05, 1.05]]

		# pick a good-looking diverging colormap with black in the middle
		cmap = sns.diverging_palette(220, 20, n=len(behav['probabilityLeft'].unique()), center="dark")
		if len(behav['probabilityLeft'].unique()) == 1:
			cmap = "gist_gray"
		sns.set_palette(cmap)

		for pidx, (var, labelname) in enumerate(parsdict.items()):
			ax = axes[pidx,1]
			sns.lineplot(x="start_time", y=var, hue="probabilityLeft", palette=cmap, data=pars, legend=None, ax=ax)
			sns.scatterplot(x="start_time", y=var, hue="probabilityLeft", palette=cmap, data=pars, legend=None, ax=ax)
			ax.set(xlabel='', ylabel=labelname, ylim=ylims[pidx], xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=1)])

			fix_date_axis(ax)
			if pidx == 0:
				ax.set(title=r'$\gamma + (1 -\gamma-\lambda)  (erf(\frac{x-\mu}{\sigma} + 1)/2$')
			if pidx < 3:
				ax.set(xticklabels=[])

		# ============================================= #
		# LAST THREE SESSIONS
		# ============================================= #

		didx = 1
		sorteddays = behav['days'].sort_values(ascending=True).unique()
		for day in behav['days'].unique():

			# use only the last three days
			if day < sorteddays[-3]:
				continue

			dat = behav.loc[behav['days'] == day, :]
			# print(dat['date'].unique())
			didx += 1

			# PSYCHOMETRIC FUNCTION
			ax = axes[0, didx]
			cmap = sns.diverging_palette(220, 20, n=len(dat['probabilityLeft'].unique()), center="dark")
			if len(dat['probabilityLeft'].unique()) == 1:
				cmap = [np.array([0,0,0,1])]

			for ix, probLeft in enumerate(dat['probabilityLeft'].sort_values().unique()):
				plot_psychometric(dat.loc[dat['probabilityLeft'] == probLeft, :], ax=ax, color=cmap[ix])
			ax.set(title=pd.to_datetime(dat['start_time'].unique()[0]).strftime('%b-%d, %A'))

			# CHRONOMETRIC FUNCTION
			ax = axes[1, didx]
			for ix, probLeft in enumerate(dat['probabilityLeft'].sort_values().unique()):
				plot_chronometric(dat.loc[dat['probabilityLeft'] == probLeft, :], ax, cmap[ix])

			# RTS THROUGHOUT SESSION
			ax = axes[2, didx]
			sns.scatterplot(x='trial', y='rt', hue='correct', 
				palette={1:"forestgreen", 0:"crimson"},
				alpha=.5, data=dat, ax=ax, legend=False)
			sns.lineplot(x='trial', y='rt', color='black', ci=None, 
				data=dat[['trial', 'rt']].rolling(10).median(), ax=ax) 
			ax.set(xlabel="Trial number", ylabel="RT (s)")
			ax.set_yscale("log")
			ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

			# # WHEEL ANALYSIS
			# thisdate = dat.loc[dat.index[0], 'date'].strftime('%Y-%m-%d')
			# eid = one.search(subjects=mouse, date_range=[thisdate, thisdate])
			# t, wheelpos, wheelvel = one.load(eid[0], 
			# 	dataset_types=['_ibl_wheel.timestamps', '_ibl_wheel.position', '_ibl_wheel.velocity'])
			# wheeltimes = np.interp(np.arange(0,len(wheelpos)), t[:,0], t[:,1])
		 #    #times = np.interp(np.arange(0,len(wheelPos)), t[:,0], t[:,1])
			# wheel = pd.DataFrame.from_dict({'position':wheelpos, 'velocity':wheelvel, 'times':wheeltimes})

			# ax = axes[3, didx]
			# sns.lineplot(x=wheeltimes, y=wheelpos, ax=ax)

		for i in range(3):
			axes[i,3].set(ylabel='')
			axes[i,4].set(ylabel='')
		righty.set(ylabel='Trial count')

		plt.tight_layout()
		fig.savefig('/Users/urai/Google Drive/Rig building WG/DataFigures/BehaviourData_Weekly/AlyxPlots/%s_overview.pdf' %mouse)

	except:
		print("%s failed to run" %mouse)
		pass

	
