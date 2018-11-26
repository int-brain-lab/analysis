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
from behavior_plots import *
from load_mouse_data import * # this has all plotting functions

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0 } )
sns.set_context(context="paper")

## CONNECT TO ONE
one = ONE() # initialize

# get folder to save plots
path = fig_path()

# get a list of all mice that are currently training
subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?water_restricted=True&alive=True'))
# subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?responsible_user=ines'))
# subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?nickname=ZM_329'))
# subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?nickname=ALK082'))

print(subjects['nickname'].unique())

for i, mouse in enumerate(subjects['nickname']):

	try:

		# MAKE THE FIGURE, divide subplots using gridspec
		print(mouse)
		fig, axes = plt.subplots(ncols=5, nrows=4, constrained_layout=False,
	        gridspec_kw=dict(width_ratios=[2,2,1,1,1], height_ratios=[1,1,1,1]), figsize=(11.69, 8.27))
		sns.set_palette("colorblind") # palette for water types

		# ============================================= #
		# GENERAL METADATA
		# ============================================= #

		fig.suptitle('Mouse %s (%s), DoB %s, user %s (%s), strain %s, cage %s, %s' %(subjects['nickname'][i],
		 subjects['sex'][i], subjects['birth_date'][i],
		 subjects['responsible_user'][i], subjects['lab'][i],
		 subjects['strain'][i], subjects['litter'][i], subjects['description'][i]))

		# ============================================= #
		# WEIGHT CURVE AND WATER INTAKE
		# ============================================= #

		ax = axes[0,0]
		# get all the weights and water aligned in 1 table
		weight_water = get_water_weight(mouse)

		# use pandas plot for a stacked bar - water types
		wa_unstacked = weight_water.pivot_table(index='days',
	    	columns='water_type', values='water_administered', aggfunc='sum').reset_index()

		# shorten names for legend
		wa_unstacked.columns = wa_unstacked.columns.str.replace("Water", "Wa")
		wa_unstacked.columns = wa_unstacked.columns.str.replace("Sucrose", "Sucr")
		wa_unstacked.columns = wa_unstacked.columns.str.replace("Citric Acid", "CA")
		wa_unstacked.columns = wa_unstacked.columns.str.replace("Hydrogel", "Hdrg")

	    # mark the citric acid columns to indicate adlib amount
		for ic, c in enumerate(wa_unstacked.columns):
			if 'CA' in c:
				wa_unstacked[c].replace({0:2}, inplace=True)

		# https://stackoverflow.com/questions/44250445/pandas-bar-plot-with-continuous-x-axis
		plotvar 	  = wa_unstacked
		plotvar.index = plotvar.days
		plotvar.drop(columns='days', inplace=True)
		plotvar = plotvar.reindex(np.arange(weight_water.days.min(), weight_water.days.max()+2))

		# sort the columns by possible water types
		plotvar = plotvar[sorted(list(plotvar.columns.values), reverse=True)]
		plotvar.plot(kind='bar', style='.', stacked=True, ax=ax, edgecolor="none")
		l = ax.legend(loc='lower left', prop={'size': 'xx-small'},
			bbox_to_anchor=(0., 1.02, 1., .102),
			ncol=2, mode="expand", borderaxespad=0., frameon=False)
		l.set_title('')
		ax.set(ylabel="Water intake (mL)", xlabel='')
		ax.yaxis.label.set_color("#0072B2")

		# overlay the weight curve
		weight_water2 = weight_water.groupby('days').mean().reset_index()
		weight_water2 = weight_water2.dropna(subset=['weight'])
		righty = ax.twinx()
		sns.lineplot(x=weight_water2.days, y=weight_water2.weight, ax=righty, color='.15', marker='o')
		righty.set(xlabel='', ylabel="Weight (g)",
			xlim=[weight_water.days.min()-2, weight_water.days.max()+2])
		righty.grid(False)

		# correct the ticks to show dates, not days
		# also indicate Mondays by grid lines
		ax.set_xticks([weight_water.days[i] for i, dt in enumerate(weight_water.date) if dt.weekday() is 0])
		ax.set_xticklabels([weight_water.date[i].strftime('%b-%d') for i, dt in enumerate(weight_water.date) if dt.weekday() is 0])
		for item in ax.get_xticklabels():
			item.set_rotation(60)

		# ============================================= #
		# TRIAL COUNTS AND SESSION DURATION
		# ============================================= #

		behav 	= get_behavior(mouse)

		ax = axes[1,0]
		trialcounts = behav.groupby(['date'])['trial'].max().reset_index()
		sns.lineplot(x="date", y="trial", marker='o', color=".15", data=trialcounts, ax=ax)
		ax.set(xlabel='', ylabel="Trial count",
			xlim=[weight_water.date.min()-timedelta(days=2), behav.date.max()+timedelta(days=2)])

		# compute the length of each session
		behav['sessionlength'] = (behav.end_time - behav.start_time)
		behav['sessionlength'] = behav.sessionlength.dt.total_seconds() / 60
		sessionlength = behav.groupby(['date'])['sessionlength'].mean().reset_index()

		righty = ax.twinx()
		sns.lineplot(x="date", y="sessionlength", marker='o', color="firebrick", data=sessionlength, ax=righty)
		righty.yaxis.label.set_color("firebrick")
		righty.tick_params(axis='y', colors='firebrick')
		righty.set(xlabel='', ylabel="Session (min)", ylim=[0,80],
				xlim=[weight_water.date.min()-timedelta(days=2), behav.date.max()+timedelta(days=2)])

		righty.grid(False)
		fix_date_axis(righty)
		fix_date_axis(ax)

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
			xlim=[weight_water.date.min()-timedelta(days=2), behav.date.max()+timedelta(days=2)],
			yticks=[0.5, 0.75, 1], ylim=[0.4, 1.01])
		# ax.yaxis.label.set_color("black")

		# RTs on right y-axis
		trialcounts = behav.groupby(['date'])['rt'].median().reset_index()
		righty = ax.twinx()
		sns.lineplot(x="date", y="rt", marker='o', color="firebrick", data=trialcounts, ax=righty)

		righty.yaxis.label.set_color("firebrick")
		righty.tick_params(axis='y', colors='firebrick')
		righty.set(xlabel='', ylabel="RT (s)", ylim=[0.1,10],
			xlim=[weight_water.date.min()-timedelta(days=2), behav.date.max()+timedelta(days=2)])
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

		# ============================================= #
		# PSYCHOMETRIC FUNCTION FITS OVER TIME
		# ============================================= #

		# fit psychfunc on choice fraction, rather than identity
		pars = behav.groupby(['date', 'probabilityLeft']).apply(fit_psychfunc).reset_index()
		parsdict = {'threshold': r'Threshold $(\sigma)$', 'bias': r'Bias $(\mu)$',
			'lapselow': r'Lapse low $(\gamma)$', 'lapsehigh': r'Lapse high $(\lambda)$'}
		ylims = [[-5, 105], [-105, 105], [-0.05, 1.05], [-0.05, 1.05]]

		# pick a good-looking diverging colormap with black in the middle
		cmap = sns.diverging_palette(220, 20, n=len(behav['probabilityLeft'].unique()), center="dark")
		if len(behav['probabilityLeft'].unique()) == 1:
			cmap = "gist_gray"
		sns.set_palette(cmap)

		# plot the fitted parameters
		for pidx, (var, labelname) in enumerate(parsdict.items()):
			ax = axes[pidx,1]
			sns.lineplot(x="date", y=var, marker='o', hue="probabilityLeft",
				palette=cmap, data=pars, legend=None, ax=ax)
			ax.set(xlabel='', ylabel=labelname, ylim=ylims[pidx],
				xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=1)])

			fix_date_axis(ax)
			if pidx == 0:
				ax.set(title=r'$\gamma + (1 -\gamma-\lambda)  (erf(\frac{x-\mu}{\sigma} + 1)/2$')

		# ============================================= #
		# LAST THREE SESSIONS
		# ============================================= #

		didx = 1
		sorteddays = behav['days'].sort_values(ascending=True).unique()
		for day in behav['days'].unique():

			# use only the last three days
			if day < sorteddays[-3]:
				continue

			# grab only that day
			dat = behav.loc[behav['days'] == day, :]
			print(dat['date'].unique())
			didx += 1

			# colormap for the asymmetric blocks
			cmap = sns.diverging_palette(220, 20, n=len(dat['probabilityLeft'].unique()), center="dark")
			if len(dat['probabilityLeft'].unique()) == 1:
				cmap = [np.array([0,0,0,1])]

			# PSYCHOMETRIC FUNCTION
			ax = axes[0, didx]
			for ix, probLeft in enumerate(dat['probabilityLeft'].sort_values().unique()):
				plot_psychometric(dat.loc[dat['probabilityLeft'] == probLeft, :], ax=ax, color=cmap[ix])
			ax.set(xlabel="Contrast (%)", ylabel="Choose right (%)")
			ax.set(title=pd.to_datetime(dat['start_time'].unique()[0]).strftime('%b-%d, %A'))

			# CHRONOMETRIC FUNCTION
			ax = axes[1, didx]
			for ix, probLeft in enumerate(dat['probabilityLeft'].sort_values().unique()):
				plot_chronometric(dat.loc[dat['probabilityLeft'] == probLeft, :], ax, cmap[ix])
			ax.set(ylim=[0.1,1])
			ax.set_yscale("log")
			ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

			# RTS THROUGHOUT SESSION
			ax = axes[2, didx]
			sns.scatterplot(x='trial', y='rt', style='correct', hue='correct',
				palette={1:"#009E73", 0:"#D55E00"}, # from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/seaborn-colorblind.mplstyle
				markers={1:'o', 0:'X'}, s=10, linewidths=0, edgecolors='none',
				alpha=.5, data=dat, ax=ax, legend=False)
			# running median overlaid
			sns.lineplot(x='trial', y='rt', color='black', ci=None,
				data=dat[['trial', 'rt']].rolling(10).median(), ax=ax)
			ax.set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.02, 60])
			ax.set_yscale("log")
			ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
				('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

			# ============================ #
			# WHEEL ANALYSIS
			# ============================ #

			plotWheel = False
			if plotWheel:
				# FIRST CREATE A PANDAS DATAFRAME WITH THE FULL WHEEL TRACE DURING THE SESSION
				thisdate = dat.loc[dat.index[0], 'date'].strftime('%Y-%m-%d')
				eid = one.search(subjects=mouse, date_range=[thisdate, thisdate])
				t, wheelpos, wheelvel = one.load(eid[0],
					dataset_types=['_ibl_wheel.timestamps', '_ibl_wheel.position', '_ibl_wheel.velocity'])
				wheel = pd.DataFrame.from_dict({'position':wheelpos[0], 'velocity':np.transpose(wheelvel)[0]})
				wheel['time'] = pd.to_timedelta(np.linspace(t[0,0], t[1,1], len(wheelpos[0])), unit='s')
				wheel.set_index(wheel['time'], inplace=True)
				wheel = wheel.resample('10ms', on='time').mean().reset_index() # to do analyses more quickly, RESAMPLE to 10ms

				# ADD A FEW SECONDS WITH NANS AT THE BEGINNING AND END
				wheel = pd.concat([ pd.DataFrame.from_dict({'time': pd.to_timedelta(np.arange(-10, 0, 0.1), 's'), 
					'position': np.full((100,), np.nan), 'velocity':  np.full((100,), np.nan)}),
					 wheel,
					 pd.DataFrame.from_dict({'time': pd.to_timedelta(np.arange(wheel.time.max().total_seconds(), 
					 	wheel.time.max().total_seconds()+10, 0.1), 's'), 
					'position': np.full((100,), np.nan), 'velocity':  np.full((100,), np.nan)})])
				wheel.index = wheel['time']

				# round to have the same sampling rate as wheeltimes
				stimonset_times = pd.to_timedelta(np.round(dat['stimOn_times'], 2), 's') # express in timedelta

				# THEN EPOCH BY LOCKING TO THE STIMULUS ONSET TIMES
				prestim 		= pd.to_timedelta(0.2, 's')
				poststim 		= pd.to_timedelta(dat.rt.median(), 's') + pd.to_timedelta(1, 's')
				
				signal = []; time = []
				for i, stimonset in enumerate(stimonset_times):
					sliceidx = (wheel.index > (stimonset - prestim)) & (wheel.index < (stimonset + poststim))
					signal.append(wheel['position'][sliceidx].values)

					# also append the time axis to alignment in seaborn plot
					if i == 0:
						timeaxis = np.linspace(-prestim.total_seconds(), poststim.total_seconds(), len(wheel['position'][sliceidx].values))
					time.append(timeaxis)

				# also baseline correct at zero
				zeroindex = np.argmin(np.abs(timeaxis))
				signal_blcorr = []
				for i, item in enumerate(signal):
					signal_blcorr.append(item - item[zeroindex])

				# MAKE INTO A PANDAS DATAFRAME AGAIN, append all relevant columns
				wheel = pd.DataFrame.from_dict({'time': np.hstack(time), 'position': np.hstack(signal), 
					'position_blcorr': np.hstack(signal_blcorr), 
					'choice': np.repeat(dat['choice'], len(timeaxis)), 
					'correct': np.repeat(dat['correct'], len(timeaxis)),
					'signedContrast': np.repeat(dat['signedContrast'], len(timeaxis))})
				
				ax = axes[3, didx]
				sns.lineplot(x='time', y='position_blcorr', ci=None, hue='signedContrast', 
					style='correct', data=wheel, ax=ax, legend=None)
				ax.set(xlabel='Time from stim (s)', ylabel='Wheel position (deg)')
			else:
				ax = axes[3, didx]

		# clean up layout
		for i in range(3):
			axes[i,3].set(ylabel='')
			axes[i,4].set(ylabel='')

		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig.savefig(join(path + '%s_overview.pdf'%mouse))
		plt.close(fig)

	except:
		print("%s failed to run" %mouse)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig.savefig(join(path + '%s_overview.pdf'%mouse))

		raise


